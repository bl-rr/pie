//! Semantic Uncertainty Text Completion example.
//!
//! Combines simple text completion with semantic entropy-based hallucination detection.
//! Generates multiple responses to assess semantic uncertainty and determines if the
//! output is likely hallucinated.

pub mod util;

use futures::future::join_all;
use inferlet::stop_condition::{StopCondition, ends_with_any, max_len};
use inferlet::{Args, Result, Sampler};
use rand::RngCore;
use std::fs;
use std::time::Instant;

use crate::util::*;

/// Default entropy threshold for hallucination detection.
/// Responses with entropy above this are considered likely hallucinated.
const DEFAULT_ENTROPY_THRESHOLD: f32 = 0.5;

#[inferlet::main]
async fn main(mut args: Args) -> Result<String> {
    // Parse command-line arguments
    let prompt: String = args.value_from_str(["-p", "--prompt"])?;
    let max_tokens: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(256);
    let system_message: String = args
        .value_from_str(["-s", "--system"])
        .unwrap_or_else(|_| "You are a helpful, respectful and honest assistant.".to_string());
    let temperature: f32 = args.value_from_str(["-t", "--temperature"]).unwrap_or(0.6);
    let top_p: f32 = args.value_from_str("--top-p").unwrap_or(0.95);
    let num_generations: usize = args.value_from_str("--num-generations").unwrap_or(5);
    let high_temperature: f32 = args.value_from_str("--high-temperature").unwrap_or(1.0);
    let entropy_threshold: f32 = args
        .value_from_str("--entropy-threshold")
        .unwrap_or(DEFAULT_ENTROPY_THRESHOLD);
    let entailment_temperature: f32 = args
        .value_from_str("--entailment-temperature")
        .unwrap_or(0.02);
    let seed: u64 = args.value_from_str("--seed").unwrap_or(42);
    let output_file: Option<String> = args.opt_value_from_str(["--output", "-o"])?;

    let start = Instant::now();

    let model = inferlet::get_auto_model();
    let mut rng = Xoshiro256StarStar::new(seed);

    // Create the shared prompt context
    let mut base_ctx = model.create_context();
    base_ctx.fill_system(&system_message);
    base_ctx.fill_user(&prompt);

    // First decode step to get initial distribution for all generations
    let seed_dist = base_ctx.decode_step_dist().await;
    let ttft = start.elapsed();

    let stop_cond = max_len(max_tokens).or(ends_with_any(model.eos_tokens()));

    // Pre-generate seeds for all generations
    let low_temp_seed = rng.next_u64();
    let gen_seeds: Vec<u64> = (0..num_generations).map(|_| rng.next_u64()).collect();

    // Create the low-temperature generation (main output)
    let low_temp_ctx = base_ctx.fork();
    let low_temp_future = {
        let mut ctx = low_temp_ctx;
        let sampler = Sampler::top_p(temperature, top_p);
        let stop = stop_cond.clone();
        let seed_ids = seed_dist.ids.clone();
        let seed_probs = seed_dist.probs.clone();

        async move {
            // Sample first token from seed distribution
            let mut local_rng = Xoshiro256StarStar::new(low_temp_seed);
            let first_idx = sample_idx(&seed_probs, temperature, &mut local_rng);
            let first_token = seed_ids[first_idx];
            ctx.fill_token(first_token);

            // Continue generation
            let text = ctx.generate(sampler, stop).await;
            text
        }
    };

    // Create high-temperature generations for semantic entropy
    let high_temp_futures: Vec<_> = gen_seeds
        .into_iter()
        .map(|seed| {
            let mut ctx = base_ctx.fork();
            let stop = stop_cond.clone();
            let seed_ids = seed_dist.ids.clone();
            let seed_probs = seed_dist.probs.clone();
            let eos_tokens = model.eos_tokens();

            async move {
                let mut local_rng = Xoshiro256StarStar::new(seed);

                // Sample first token from seed distribution
                let first_idx = sample_idx(&seed_probs, high_temperature, &mut local_rng);
                let first_token = seed_ids[first_idx];
                ctx.fill_token(first_token);
                let mut generated = vec![first_token];
                let mut logprobs = vec![seed_probs[first_idx].max(f32::MIN_POSITIVE).ln()];

                // Manual decode loop with seeded sampling
                while generated.len() < max_tokens && !stop.check(&generated) {
                    let dist = ctx.decode_step_dist().await;
                    let idx = sample_idx(&dist.probs, high_temperature, &mut local_rng);
                    let token = dist.ids[idx];
                    let lp = dist.probs[idx].max(f32::MIN_POSITIVE).ln();
                    ctx.fill_token(token);
                    generated.push(token);
                    logprobs.push(lp);
                }

                // Remove trailing stop tokens
                let mut final_tokens = generated;
                for stop_seq in &eos_tokens {
                    if !stop_seq.is_empty() && final_tokens.ends_with(stop_seq) {
                        final_tokens.truncate(final_tokens.len().saturating_sub(stop_seq.len()));
                        break;
                    }
                }

                let text = ctx.tokenizer.detokenize(&final_tokens).trim().to_string();
                let avg_ll = if logprobs.is_empty() {
                    f32::NEG_INFINITY
                } else {
                    logprobs.iter().sum::<f32>() / logprobs.len() as f32
                };
                (text, avg_ll)
            }
        })
        .collect();

    // Run all generations in parallel
    let gen_start = Instant::now();
    let (main_text, high_temp_results) =
        futures::join!(low_temp_future, join_all(high_temp_futures));
    let generation_time = gen_start.elapsed();

    // Extract responses and log-likelihoods
    let responses: Vec<String> = high_temp_results.iter().map(|(t, _)| t.clone()).collect();
    let avg_log_likelihoods: Vec<f32> = high_temp_results.iter().map(|(_, ll)| *ll).collect();

    // Compute semantic entropy
    let semantic_start = Instant::now();
    let mut judge = EntailmentJudge::new(&model, 32, false);
    let semantic_ids =
        get_semantic_ids(&responses, &mut judge, &prompt, entailment_temperature).await;
    let log_probs = logsumexp_by_id(&semantic_ids, &avg_log_likelihoods);
    let semantic_entropy = predictive_entropy_rao(&log_probs);
    let semantic_time = semantic_start.elapsed();

    // Determine hallucination likelihood
    let likely_hallucinated = semantic_entropy >= entropy_threshold;
    let num_unique_meanings = semantic_ids.iter().max().map(|m| m + 1).unwrap_or(0);

    let total_time = start.elapsed();

    // Build output JSON
    let output = serde_json::json!({
        "prompt": prompt,
        "text": main_text,
        "semantic_entropy": semantic_entropy,
        "likely_hallucinated": likely_hallucinated,
        "num_unique_meanings": num_unique_meanings,
        "num_generations": num_generations,
        "generated_texts": responses,
        "semantic_ids": semantic_ids,
        "timing": {
            "ttft_ms": ttft.as_millis(),
            "generation_ms": generation_time.as_millis(),
            "semantic_check_ms": semantic_time.as_millis(),
            "total_ms": total_time.as_millis(),
        },
        "config": {
            "temperature": temperature,
            "high_temperature": high_temperature,
            "entropy_threshold": entropy_threshold,
            "max_tokens": max_tokens,
        }
    });

    // Write to file if --output specified
    if let Some(ref path) = output_file {
        fs::write(path, serde_json::to_string_pretty(&output)?)?;
        eprintln!("Summary written to {}", path);
    }

    println!("likely hallucinated: {}", likely_hallucinated);
    Ok(main_text)
}

/// Sample an index from a probability distribution with temperature scaling.
fn sample_idx(probs: &[f32], temperature: f32, rng: &mut Xoshiro256StarStar) -> usize {
    use rand::distributions::{Distribution, WeightedIndex};

    if probs.is_empty() {
        return 0;
    }

    // Apply temperature
    let adjusted: Vec<f64> = if temperature <= 0.0 {
        // Greedy: pick max
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let mut result = vec![0.0; probs.len()];
        result[max_idx] = 1.0;
        result
    } else {
        let inv_t = 1.0f32 / temperature.max(1e-5);
        let mut adj: Vec<f64> = probs
            .iter()
            .map(|p| (*p as f64).powf(inv_t as f64))
            .collect();
        let sum: f64 = adj.iter().sum();
        if sum > 0.0 {
            for p in adj.iter_mut() {
                *p /= sum;
            }
        }
        adj
    };

    if let Ok(dist) = WeightedIndex::new(&adjusted) {
        dist.sample(rng)
    } else {
        adjusted
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}
