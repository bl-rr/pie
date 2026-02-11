//! Utilities for Chain-of-Verification inferlet.
//!
//! Includes config structures, RNG, HTTP search, and metrics.

use inferlet::wstd::http::{Client, Method, Request};
use inferlet::wstd::io::{empty, AsyncRead};
use inferlet::Result;
use rand::{Error, RngCore};
use serde::Deserialize;
use std::fs;

// ============================================================================
// Configuration (all fields required - panics if missing)
// ============================================================================

#[derive(Deserialize, Clone, Debug)]
pub struct GenerationConfig {
    pub model_name: String,
    pub model_id: String,
    pub tokenizer_id: String,
    pub torch_dtype: String,
    pub max_new_tokens: usize,
    pub temperature: f32,
}

#[derive(Deserialize, Clone, Debug)]
pub struct CoVeConfig {
    pub use_search: bool,
    pub max_search_results: usize,
    pub search_timeout: f32,
    pub max_tokens_baseline: usize,
    pub max_tokens_verification: usize,
    pub max_tokens_final: usize,
    pub temperature_routing: f32,
    pub temperature_baseline: f32,
    pub temperature_verification: f32,
    pub temperature_final: f32,
    pub max_verification_questions: usize,
}

#[derive(Deserialize, Clone, Debug)]
pub struct DataConfig {
    pub dataset: String,
    pub dataset_path: String,
    pub num_samples: usize,
    pub random_seed: u64,
    pub metric: String,
}

#[derive(Deserialize, Clone, Debug)]
pub struct RootConfig {
    pub generation: GenerationConfig,
    pub cove: CoVeConfig,
    pub data: DataConfig,
}

pub fn load_config(path: &str) -> RootConfig {
    let content = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read config file '{}': {}", path, e));
    toml::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse config file '{}': {}", path, e))
}

// ============================================================================
// Dataset Structures
// ============================================================================

#[derive(Deserialize, Clone, Debug, Default)]
pub struct Answers {
    pub text: Vec<String>,
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct QAExample {
    pub question: String,
    #[serde(default)]
    pub context: Option<String>,
    #[serde(default)]
    pub answers: Option<Answers>,
    #[serde(default)]
    pub id: Option<String>,
}

impl QAExample {
    pub fn answer_texts(&self) -> Vec<String> {
        self.answers
            .as_ref()
            .map(|a| a.text.clone())
            .unwrap_or_default()
    }
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct DatasetFile {
    #[serde(default)]
    pub dataset: Option<String>,
    #[serde(default)]
    pub chain_type: Option<String>,
    #[serde(default)]
    pub train: Vec<QAExample>,
    #[serde(default)]
    pub validation: Vec<QAExample>,
}

pub fn load_dataset(path: &str) -> DatasetFile {
    let content = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read dataset file '{}': {}", path, e));

    // Try parsing as DatasetFile first
    if let Ok(ds) = serde_json::from_str::<DatasetFile>(&content) {
        return ds;
    }

    // Try parsing as array of examples
    if let Ok(examples) = serde_json::from_str::<Vec<QAExample>>(&content) {
        return DatasetFile {
            dataset: None,
            chain_type: None,
            train: examples.clone(),
            validation: examples,
        };
    }

    panic!(
        "Failed to parse dataset file '{}' as DatasetFile or Vec<QAExample>",
        path
    );
}

// ============================================================================
// DuckDuckGo HTTP Search
// ============================================================================

/// Search DuckDuckGo and return formatted results.
pub async fn search_ddg(query: &str, max_results: usize) -> Result<String> {
    let client = Client::new();

    // Use DuckDuckGo instant answer API
    let encoded_query = urlencoding::encode(query);
    let url = format!(
        "https://api.duckduckgo.com/?q={}&format=json&no_html=1&skip_disambig=1",
        encoded_query
    );

    let request = Request::builder()
        .uri(&url)
        .method(Method::GET)
        .body(empty())?;

    let response = client.send(request).await?;
    let mut body = response.into_body();
    let mut buf = Vec::new();
    body.read_to_end(&mut buf).await?;

    let text = String::from_utf8_lossy(&buf).to_string();

    // Parse JSON and extract relevant info
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
        let mut results = Vec::new();

        // Abstract (main answer)
        if let Some(abstract_text) = json.get("AbstractText").and_then(|v| v.as_str()) {
            if !abstract_text.is_empty() {
                results.push(format!("Summary: {}", abstract_text));
            }
        }

        // Related topics
        if let Some(topics) = json.get("RelatedTopics").and_then(|v| v.as_array()) {
            for (i, topic) in topics.iter().take(max_results).enumerate() {
                if let Some(text) = topic.get("Text").and_then(|v| v.as_str()) {
                    results.push(format!("{}. {}", i + 1, text));
                }
            }
        }

        if !results.is_empty() {
            return Ok(results.join("\n"));
        }
    }

    Ok("No search results found.".to_string())
}

// ============================================================================
// Metrics
// ============================================================================

/// Calculate FACTSCORE: fraction of reference entities found in prediction.
pub fn calculate_factscore(prediction: &str, references: &[String]) -> f64 {
    if references.is_empty() {
        return 0.0;
    }

    let pred_lower = prediction.to_lowercase();
    let matches = references
        .iter()
        .filter(|r| {
            let ref_lower = r.to_lowercase();
            !ref_lower.is_empty() && pred_lower.contains(&ref_lower)
        })
        .count();

    matches as f64 / references.len() as f64
}

// ============================================================================
// Xoshiro256StarStar RNG (matches Python implementation)
// ============================================================================

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

pub struct Xoshiro256StarStar {
    s: [u64; 4],
}

impl RngCore for Xoshiro256StarStar {
    fn next_u32(&mut self) -> u32 {
        self.next_u64_custom() as u32
    }

    fn next_u64(&mut self) -> u64 {
        self.next_u64_custom()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut chunks = dest.chunks_exact_mut(8);
        for chunk in chunks.by_ref() {
            let n = self.next_u64_custom();
            chunk.copy_from_slice(&n.to_le_bytes());
        }
        let remainder = chunks.into_remainder();
        if !remainder.is_empty() {
            let n = self.next_u64_custom();
            remainder.copy_from_slice(&n.to_le_bytes()[..remainder.len()]);
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl Xoshiro256StarStar {
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

    /// Generate random number in range [low, high) using Lemire's method.
    pub fn gen_range(&mut self, low: u64, high: u64) -> u64 {
        let range = high - low;
        let zone = (0u64.wrapping_sub(range)) % range;

        loop {
            let v = self.next_u64();
            let m = (v as u128) * (range as u128);
            let l = m as u64;

            if l >= zone {
                return low + (m >> 64) as u64;
            }
        }
    }
}

/// Pick random indices from candidates.
pub fn pick_indices(
    candidates: &[usize],
    count: usize,
    rng: &mut Xoshiro256StarStar,
) -> Vec<usize> {
    let mut pool = candidates.to_vec();
    let n = pool.len();
    let mut indices = Vec::with_capacity(count);

    for i in 0..count.min(n) {
        let idx = rng.gen_range(i as u64, n as u64) as usize;
        pool.swap(i, idx);
        indices.push(pool[i]);
    }

    indices
}
