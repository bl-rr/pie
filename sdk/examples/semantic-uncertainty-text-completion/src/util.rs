//! Utilities for semantic uncertainty text completion.
//! Adapted from the semantic-uncertainty example.

use futures::future::join_all;
use inferlet::stop_condition::{StopCondition, ends_with_any, max_len};
use inferlet::{Context, Sampler};
use rand::{Error, RngCore};
use std::collections::HashMap;

// --- SplitMix64 (Used for seeding) ---
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
}

/// Entailment judge for semantic similarity checks.
pub struct EntailmentJudge<'a> {
    pub model: &'a inferlet::Model,
    pub max_new_tokens: usize,
    pub strict: bool,
    pub cache: HashMap<(String, String), u8>,
    pub global_prefix_ctx: Option<Context>,
    pub question_prefix_cache: Option<(String, Context)>,
}

impl<'a> EntailmentJudge<'a> {
    pub fn new(model: &'a inferlet::Model, max_new_tokens: usize, strict: bool) -> Self {
        Self {
            model,
            max_new_tokens,
            strict,
            cache: HashMap::new(),
            global_prefix_ctx: None,
            question_prefix_cache: None,
        }
    }

    async fn init_global_prefix(&mut self) {
        if self.global_prefix_ctx.is_some() {
            return;
        }
        const GLOBAL_PREFIX: &str = "We are evaluating answers to the question ";
        let mut ctx = self.model.create_context();
        ctx.fill(GLOBAL_PREFIX);
        ctx.flush().await;
        self.global_prefix_ctx = Some(ctx);
    }

    async fn get_or_prepare_prefix_context(&mut self, question: &str) -> Context {
        if let Some((ref cached_question, ref cached_ctx)) = self.question_prefix_cache
            && cached_question == question
        {
            return cached_ctx.fork();
        }

        self.init_global_prefix().await;

        let mut ctx = self.global_prefix_ctx.as_ref().unwrap().fork();
        let suffix = format!(
            "\"{}\"\nHere are two possible answers, does Possible Answer 1 semantically entail Possible Answer 2? Respond only with entailment, contradiction, or neutral.\nPossible Answer 1: ",
            question
        );
        ctx.fill(&suffix);
        ctx.flush().await;

        self.question_prefix_cache = Some((question.to_string(), ctx.fork()));
        ctx
    }

    pub async fn check_implications_batch(
        &mut self,
        pairs: &[(usize, usize)],
        strings_list: &[String],
        question: &str,
        entailment_temperature: f32,
    ) -> Vec<((usize, usize), u8)> {
        if pairs.is_empty() {
            return Vec::new();
        }

        let mut results: Vec<((usize, usize), u8)> = Vec::new();
        let mut uncached_pairs: Vec<(usize, usize)> = Vec::new();

        for &(i, j) in pairs {
            let text1 = &strings_list[i];
            let text2 = &strings_list[j];
            if text1 == text2 {
                results.push(((i, j), 2u8));
                continue;
            }
            if let Some(&val) = self.cache.get(&(text1.clone(), text2.clone())) {
                results.push(((i, j), val));
            } else {
                uncached_pairs.push((i, j));
            }
        }

        if uncached_pairs.is_empty() {
            return results;
        }

        let prefix_ctx = self.get_or_prepare_prefix_context(question).await;

        let futures: Vec<_> = uncached_pairs
            .iter()
            .map(|&(i, j)| {
                let text1 = strings_list[i].clone();
                let text2 = strings_list[j].clone();
                let mut ctx_clone = prefix_ctx.fork();
                let eos_tokens = self.model.eos_tokens();
                let max_tokens = self.max_new_tokens;

                async move {
                    let suffix = format!("{}\nPossible Answer 2: {}\nResponse:", text1, text2);
                    ctx_clone.fill(&suffix);

                    let stop_cond = ends_with_any(eos_tokens.clone()).or(max_len(max_tokens));
                    let sampler = Sampler::top_p(entailment_temperature, 0.95);
                    let response = ctx_clone.generate(sampler, stop_cond).await;

                    let binary_response: String = response.chars().take(30).collect();
                    let result = if binary_response.contains("entailment") {
                        2u8
                    } else if binary_response.contains("neutral") {
                        1u8
                    } else if binary_response.contains("contradiction") {
                        0u8
                    } else {
                        1u8
                    };

                    // println!("Entailment: {} -> {}", text1, text2);
                    // println!("Result: {}", binary_response);

                    ((i, j), result, text1, text2)
                }
            })
            .collect();

        let batch_results = join_all(futures).await;

        for ((i, j), result, text1, text2) in batch_results {
            self.cache.insert((text1, text2), result);
            results.push(((i, j), result));
        }

        results
    }
}

/// Compute semantic IDs from responses using entailment.
pub async fn get_semantic_ids(
    strings_list: &[String],
    judge: &mut EntailmentJudge<'_>,
    question: &str,
    entailment_temperature: f32,
) -> Vec<usize> {
    let n = strings_list.len();
    if n == 0 {
        return Vec::new();
    }

    let mut pairs_to_check: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            pairs_to_check.push((i, j));
            pairs_to_check.push((j, i));
        }
    }

    let implication_results = judge
        .check_implications_batch(
            &pairs_to_check,
            strings_list,
            question,
            entailment_temperature,
        )
        .await;

    let mut implications: HashMap<(usize, usize), u8> = HashMap::new();
    for ((i, j), result) in implication_results {
        implications.insert((i, j), result);
    }

    let is_equivalent = |i: usize, j: usize| -> bool {
        let imp_ij = implications.get(&(i, j)).copied().unwrap_or(1);
        let imp_ji = implications.get(&(j, i)).copied().unwrap_or(1);

        if judge.strict {
            imp_ij == 2 && imp_ji == 2
        } else {
            imp_ij != 0 && imp_ji != 0 && !(imp_ij == 1 && imp_ji == 1)
        }
    };

    let mut semantic_set_ids = vec![-1i32; n];
    let mut next_id = 0i32;

    for i in 0..n {
        if semantic_set_ids[i] == -1 {
            semantic_set_ids[i] = next_id;
            for j in (i + 1)..n {
                if is_equivalent(i, j) {
                    semantic_set_ids[j] = next_id;
                }
            }
            next_id += 1;
        }
    }

    semantic_set_ids.into_iter().map(|x| x as usize).collect()
}

/// Compute log-sum-exp grouped by semantic ID.
pub fn logsumexp_by_id(semantic_ids: &[usize], log_likelihoods: &[f32]) -> Vec<f32> {
    assert_eq!(
        semantic_ids.len(),
        log_likelihoods.len(),
        "Semantic ids and likelihoods must match."
    );
    let total_prob: f32 = log_likelihoods.iter().map(|l| l.exp()).sum();
    if total_prob <= 0.0 {
        return vec![f32::NEG_INFINITY; semantic_ids.len()];
    }

    let mut per_id: HashMap<usize, f32> = HashMap::new();
    for (sid, loglik) in semantic_ids.iter().zip(log_likelihoods.iter()) {
        let entry = per_id.entry(*sid).or_insert(0.0);
        *entry += loglik.exp() / total_prob;
    }

    let mut ids: Vec<usize> = per_id.keys().cloned().collect();
    ids.sort_unstable();
    ids.iter()
        .map(|id| per_id.get(id).copied().unwrap_or(0.0).ln())
        .collect()
}

/// Compute predictive entropy using Rao's method.
pub fn predictive_entropy_rao(log_probs: &[f32]) -> f32 {
    -log_probs.iter().map(|lp| lp.exp() * lp).sum::<f32>()
}
