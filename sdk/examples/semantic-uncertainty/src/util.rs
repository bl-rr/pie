use inferlet::{Context, Result};
use rand::{Error, RngCore};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::time::Duration;

pub const DEFAULT_BRIEF: &str = "Answer the following question as briefly as possible.\n";
pub const CHAT_BRIEF: &str =
    "Answer the following question in a single brief but complete sentence.\n";

#[derive(Deserialize, Clone, Debug, Default)]
pub struct GenerationConfig {
    pub model_name: Option<String>,
    pub max_new_tokens: Option<usize>,
    pub num_generations: Option<usize>,
    pub temperature: Option<f32>,
    pub low_temperature: Option<f32>,
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct EntailmentConfig {
    pub max_new_tokens: Option<usize>,
    pub temperature: Option<f32>,
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct DataConfig {
    pub dataset: Option<String>,
    pub dataset_path: Option<String>,
    pub num_samples: Option<usize>,
    pub num_few_shot: Option<usize>,
    pub use_context: Option<bool>,
    pub random_seed: Option<u64>,
    pub brief_prompt: Option<String>,
    pub brief_always: Option<bool>,
    pub enable_brief: Option<bool>,
    pub metric: Option<String>,
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct UncertaintyConfig {
    pub strict_entailment: Option<bool>,
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct RootConfig {
    pub generation: Option<GenerationConfig>,
    pub entailment: Option<EntailmentConfig>,
    pub data: Option<DataConfig>,
    pub uncertainty: Option<UncertaintyConfig>,
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct Answers {
    pub text: Vec<String>,
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct Reference {
    pub answers: Answers,
    #[serde(default)]
    #[allow(dead_code)]
    pub id: Option<String>,
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct QAExample {
    pub question: String,
    #[serde(default)]
    pub context: Option<String>,
    #[serde(default)]
    pub answers: Option<Answers>,
    #[serde(default)]
    pub reference: Option<Reference>,
    #[serde(default)]
    #[allow(dead_code)]
    pub id: Option<String>,
}

impl QAExample {
    pub fn answer_texts(&self) -> Vec<String> {
        if let Some(ans) = &self.answers
            && !ans.text.is_empty()
        {
            return ans.text.clone();
        }
        if let Some(reference) = &self.reference {
            return reference.answers.text.clone();
        }
        Vec::new()
    }
}

#[derive(Debug)]
pub struct DatasetSplits {
    pub train: Vec<QAExample>,
    pub validation: Vec<QAExample>,
}

pub struct PromptState {
    pub ctx: Context,
    pub seed_ids: Vec<u32>,
    pub seed_probs: Vec<f32>,
    pub ttft: Duration,
}

#[derive(Debug)]
pub struct GenerationOutcome {
    pub text: String,
    pub logprobs: Vec<f32>, // base log-probs (temperature=1) of sampled tokens
}

pub struct EntailmentJudge<'a> {
    pub model: &'a inferlet::Model,
    pub max_new_tokens: usize,
    pub strict: bool,
    pub cache: HashMap<(String, String), u8>,
    /// Global prefix context shared across ALL questions.
    /// Caches the invariant part: "We are evaluating answers to the question "
    pub global_prefix_ctx: Option<Context>,
    /// Cached prompt state for the current question's prefix.
    /// This avoids re-prefilling the common part of the entailment prompt.
    pub question_prefix_cache: Option<(String, Context)>,
}

pub fn load_dataset_splits(dataset_path: &str, dataset_name: &str) -> Result<DatasetSplits> {
    let raw = fs::read_to_string(dataset_path)?;
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) {
        if value.is_array() {
            let data: Vec<QAExample> = serde_json::from_value(value)?;
            return Ok(DatasetSplits {
                train: data.clone(),
                validation: data,
            });
        }
        if let Some(ds) = parse_dataset_object(&value)? {
            return Ok(ds);
        }
    }

    Err(inferlet::Error::msg(format!(
        "Could not load dataset '{}' from path {:?}",
        dataset_name, dataset_path
    )))
}

pub fn parse_dataset_object(value: &serde_json::Value) -> Result<Option<DatasetSplits>> {
    if !value.is_object() {
        return Ok(None);
    }

    let mut train = None;
    let mut validation = None;

    let keys = [
        "train",
        "training",
        "validation",
        "val",
        "test",
        "examples",
        "data",
    ];
    for key in keys {
        if let Some(arr) = value.get(key)
            && arr.is_array()
        {
            let parsed: Vec<QAExample> = serde_json::from_value(arr.clone())?;
            match key {
                "train" | "training" => train = Some(parsed),
                "validation" | "val" | "test" => validation = Some(parsed),
                _ => {
                    if validation.is_none() {
                        validation = Some(parsed.clone());
                    }
                    if train.is_none() {
                        train = Some(parsed);
                    }
                }
            }
        }
    }

    if let Some(val) = validation {
        let tr = train.unwrap_or_else(|| val.clone());
        return Ok(Some(DatasetSplits {
            train: tr,
            validation: val,
        }));
    }

    Ok(None)
}

// --- SplitMix64 (Used for seeding) ---
// This is what rand::SeedableRng uses to turn a u64 seed into 256 bits of state
struct SplitMix64 {
    x: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { x: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.x = self.x.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.x;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

// --- Xoshiro256** (The Generator) ---
pub struct Xoshiro256StarStar {
    s: [u64; 4],
}

// --- THE FIX: Implement RngCore ---
impl RngCore for Xoshiro256StarStar {
    fn next_u32(&mut self) -> u32 {
        self.next_u64_custom() as u32
    }

    fn next_u64(&mut self) -> u64 {
        self.next_u64_custom()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        // 1. Create an iterator that gives us mutable chunks of 8 bytes
        let mut chunks = dest.chunks_exact_mut(8);

        // 2. Fill all full 8-byte chunks
        for chunk in chunks.by_ref() {
            let n = self.next_u64_custom();
            chunk.copy_from_slice(&n.to_le_bytes());
        }

        // 3. Handle the leftover bytes (if any)
        let remainder = chunks.into_remainder();
        if !remainder.is_empty() {
            let n = self.next_u64_custom();
            // Only copy as many bytes as we need for the remainder
            remainder.copy_from_slice(&n.to_le_bytes()[..remainder.len()]);
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl Xoshiro256StarStar {
    // Matches Xoshiro256StarStar::seed_from_u64(seed)
    pub fn new(seed: u64) -> Self {
        let mut sm = SplitMix64::new(seed);
        Self {
            s: [sm.next_u64(), sm.next_u64(), sm.next_u64(), sm.next_u64()],
        }
    }

    #[inline]
    fn rotl(x: u64, k: u32) -> u64 {
        (x << k) | (x >> (64 - k))
    }

    // Your CORE logic (the source of truth)
    fn next_u64_custom(&mut self) -> u64 {
        let result = Self::rotl(self.s[1].wrapping_mul(5), 7).wrapping_mul(9);
        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = Self::rotl(self.s[3], 45);

        result
    }

    // Matches rand::Rng::gen_range using Lemire's method
    pub fn gen_range(&mut self, low: u64, high: u64) -> u64 {
        let range = high - low;
        let zone = (0u64.wrapping_sub(range)) % range;

        loop {
            let v = self.next_u64();
            // We cast to u128 to perform the multiplication without overflow
            // This is the standard way to implement Lemire's method in Rust
            let m = (v as u128) * (range as u128);
            let l = m as u64; // Keep only lower 64 bits

            if l >= zone {
                return low + (m >> 64) as u64;
            }
        }
    }
}

/// select count from candidates
pub fn pick_indices(
    candidates: &[usize],
    count: usize,
    rng: &mut Xoshiro256StarStar,
) -> Vec<usize> {
    let mut indices = Vec::with_capacity(count);
    let pool = candidates.to_vec();
    let n = pool.len();
    for i in 0..count {
        let idx = rng.gen_range(i as u64, n as u64);
        println!("{}", idx);

        indices.push(pool[idx as usize]);
    }
    indices
}
