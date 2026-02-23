//! Utilities for semantic uncertainty text completion (NLI variant).
//!
//! Key difference from the original: `NliEntailmentJudge` uses the DeBERTa NLI model
//! via `queue.classify_batch()` instead of LLM-based entailment checking. This batches
//! all O(n²) entailment pairs into a single RPC call for efficient GPU inference.

use inferlet::Queue;
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

/// NLI-based entailment judge using a dedicated classification model.
///
/// Instead of generating text with the LLM and parsing labels, this judge
/// sends all (premise, hypothesis) pairs to the DeBERTa NLI model in a single
/// batched RPC call via `queue.classify_batch()`.
pub struct NliEntailmentJudge<'a> {
    pub queue: &'a Queue,
    pub cache: HashMap<(String, String), u8>,
    pub strict: bool,
}

impl<'a> NliEntailmentJudge<'a> {
    pub fn new(queue: &'a Queue) -> Self {
        Self {
            queue,
            cache: HashMap::new(),
            strict: false,
        }
    }

    pub async fn check_implications_batch(
        &mut self,
        pairs: &[(usize, usize)],
        strings_list: &[String],
        _question: &str,
    ) -> Vec<((usize, usize), u8)> {
        if pairs.is_empty() {
            return Vec::new();
        }

        let mut results: Vec<((usize, usize), u8)> = Vec::new();
        let mut uncached_pairs: Vec<(usize, usize)> = Vec::new();
        let mut uncached_nli_pairs: Vec<(String, String)> = Vec::new();

        for &(i, j) in pairs {
            let text1 = &strings_list[i];
            let text2 = &strings_list[j];
            if text1 == text2 {
                results.push(((i, j), 2u8)); // entailment
                continue;
            }
            if let Some(&val) = self.cache.get(&(text1.clone(), text2.clone())) {
                results.push(((i, j), val));
            } else {
                uncached_pairs.push((i, j));
                uncached_nli_pairs.push((text1.clone(), text2.clone()));
            }
        }

        if uncached_pairs.is_empty() {
            return results;
        }

        // Single batched RPC call for ALL uncached pairs
        let nli_results = self.queue.classify_batch(&uncached_nli_pairs).await;

        for (idx, &(i, j)) in uncached_pairs.iter().enumerate() {
            let (ref label, _score) = nli_results[idx];
            let result = match label.as_str() {
                "entailment" => 2u8,
                "neutral" => 1u8,
                "contradiction" => 0u8,
                _ => 1u8, // default to neutral
            };

            let text1 = strings_list[i].clone();
            let text2 = strings_list[j].clone();
            self.cache.insert((text1, text2), result);
            results.push(((i, j), result));
        }

        results
    }
}

/// Compute semantic IDs from responses using NLI entailment.
pub async fn get_semantic_ids(
    strings_list: &[String],
    judge: &mut NliEntailmentJudge<'_>,
    question: &str,
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
        .check_implications_batch(&pairs_to_check, strings_list, question)
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
