pub mod util;

use futures::future::join_all;
// use inferlet::profiler::{FutureTracer, Profiler, trace_init, trace_record, trace_write_json};
use inferlet::stop_condition::{StopCondition, ends_with_any, max_len};
use inferlet::{Args, Context, Result, Sampler};
use rand::distributions::{Distribution, WeightedIndex};
use rand::{RngCore, SeedableRng};
use std::io::Write;

use std::collections::HashMap;
use std::fs;

use crate::util::*;

use std::time::Instant;

#[inferlet::main]
async fn main(mut args: Args) -> Result<String> {
    let config_path: Option<String> = args.opt_value_from_str(["-c", "--config"])?;

    // load the config
    let cfg_str = fs::read_to_string(config_path.as_ref().unwrap())?;
    let cfg: RootConfig = toml::from_str(&cfg_str)?;

    // the more fine-grained config
    let gen_cfg = cfg.generation.unwrap();
    let ent_cfg = cfg.entailment.unwrap();
    let data_cfg = cfg.data.unwrap();
    let unc_cfg = cfg.uncertainty.unwrap();

    let dataset_path = data_cfg.dataset_path.unwrap();
    let dataset_name = data_cfg.dataset.unwrap();
    let brief_prompt = data_cfg.brief_prompt.unwrap();
    let use_context = data_cfg.use_context.unwrap();
    let brief_always = data_cfg.brief_always.unwrap();
    let enable_brief = data_cfg.enable_brief.unwrap();
    let metric_name = data_cfg.metric.unwrap();

    let num_few_shot = data_cfg.num_few_shot.unwrap();
    let num_samples = data_cfg.num_samples.unwrap();
    let random_seed = data_cfg.random_seed.unwrap();

    let entailment_max_new_tokens = ent_cfg.max_new_tokens.unwrap();
    let entailment_temperature = ent_cfg.temperature.unwrap();

    let num_generations = gen_cfg.num_generations.unwrap();
    let max_new_tokens = gen_cfg.max_new_tokens.unwrap();
    let low_temperature = gen_cfg.low_temperature.unwrap();
    let high_temperature = gen_cfg.temperature.unwrap();

    let strict_entailment = unc_cfg.strict_entailment.unwrap();

    let splits = load_dataset_splits(&dataset_path, &dataset_name)?;
    println!(
        "Loaded dataset '{}' from {} (train={}, validation={})",
        dataset_name,
        dataset_path,
        splits.train.len(),
        splits.validation.len()
    );
    let _ = std::io::stdout().flush();
    println!(
        "[DEBUG] Loaded dataset '{}' from {} (train={}, validation={})",
        dataset_name,
        dataset_path,
        splits.train.len(),
        splits.validation.len()
    );
    let _ = std::io::stdout().flush();

    let answerable_indices: Vec<usize> = splits
        .train
        .iter()
        .enumerate()
        .filter(|(_, ex)| !ex.answer_texts().is_empty())
        .map(|(idx, _)| idx)
        .collect();
    println!("Found {} answerable examples", answerable_indices.len());
    if answerable_indices.len() < num_few_shot {
        return Err(inferlet::Error::msg(
            "Not enough answerable examples for few-shot",
        ));
    }
    println!("Found {} answerable examples", answerable_indices.len());

    // use seed to generate Xoshiro256StarStar rng and pick num_few_shot indices from answerable_indices
    println!("[DEBUG] About to create RNG with seed {}", random_seed);
    let mut rng = Xoshiro256StarStar::new(random_seed);
    println!(
        "[DEBUG] About to pick {} indices from {} candidates",
        num_few_shot,
        answerable_indices.len()
    );

    let prompt_indices = pick_indices(&answerable_indices, num_few_shot, &mut rng);
    println!("[DEBUG] Picked indices: {:?}", prompt_indices);

    let brief_text = get_brief_text(&brief_prompt);
    let effective_brief_always = brief_always && enable_brief;
    let fewshot_prompt = construct_fewshot_prompt(
        &splits.train,
        &prompt_indices,
        brief_text,
        effective_brief_always,
        use_context,
    );

    let model = inferlet::get_auto_model();
    // Build stop tokens mirroring the Python STOP_SEQUENCES.
    let mut stop_tokens = model.eos_tokens();
    let stop_strings = ["\n\n\n\n", "\n\n\n", "\n\n", "\n", "Question:", "Context:"];
    let tokenizer = model.get_tokenizer();
    for s in stop_strings {
        stop_tokens.push(tokenizer.tokenize(s));
    }
    let mut judge = EntailmentJudge::new(&model, entailment_max_new_tokens, strict_entailment);

    println!(
        "Running semantic uncertainty pipeline on dataset '{}' ({} samples)...",
        dataset_name, num_samples
    );

    let mut semantic_entropies = Vec::new();
    let mut hallucination_labels = Vec::new();
    let mut end_to_end_latencies = Vec::new();
    let mut generation_times = Vec::new();
    let mut semantic_durations = Vec::new();
    let mut correct_count = 0usize;
    let mut incorrect_count = 0usize;
    let mut per_sample_f1 = Vec::new();
    let mut sample_outputs = Vec::new();

    for (idx, example) in splits.validation.iter().take(num_samples).enumerate() {
        if idx == 0 || (idx + 1) % 5 == 0 || (idx + 1) == num_samples {
            println!("Progress: {}/{} validation samples", idx + 1, num_samples);
        }

        let sample_start = Instant::now();

        let current_prompt = make_prompt(
            use_context,
            effective_brief_always,
            brief_text,
            example.context.as_deref(),
            &example.question,
            None,
        );
        let full_prompt = format!("{}{}", fewshot_prompt, current_prompt);

        let prompt_state = prepare_prompt_state(&model, &full_prompt).await;

        let stop_cond = ends_with_any(stop_tokens.clone()).or(max_len(max_new_tokens));

        // Pre-generate deterministic seeds for all parallel generations (low_temp + high_temp)
        // Each seed is unique to ensure different random sequences
        let low_temp_seed = rng.next_u64();
        let gen_seeds: Vec<u64> = (0..num_generations).map(|_| rng.next_u64()).collect();

        // Track fork overhead for low-temperature generation
        let low_temp_ctx = prompt_state.ctx.fork();

        // Create low-temperature generation future
        let low_temp_future = {
            let ctx = low_temp_ctx;
            let seed_ids = prompt_state.seed_ids.clone();
            let seed_probs = prompt_state.seed_probs.clone();
            let stop_tokens_clone = stop_tokens.clone();
            let stop_cond_clone = stop_cond.clone();

            async move {
                let mut local_rng = Xoshiro256StarStar::new(low_temp_seed);
                let mut ctx = ctx;

                let (first_token, first_lp) = sample_from_distribution(
                    &seed_ids,
                    &seed_probs,
                    low_temperature,
                    &mut local_rng,
                );
                ctx.fill_token(first_token);
                let mut generated = vec![first_token];
                let mut logprobs = vec![first_lp];

                while generated.len() < max_new_tokens && !stop_cond_clone.check(&generated) {
                    let dist = ctx.decode_step_dist().await;
                    let (token, lp_base) = sample_from_distribution(
                        &dist.ids,
                        &dist.probs,
                        low_temperature,
                        &mut local_rng,
                    );
                    ctx.fill_token(token);
                    generated.push(token);
                    logprobs.push(lp_base);
                }

                let mut final_tokens = generated;
                for stop in &stop_tokens_clone {
                    if !stop.is_empty() && final_tokens.ends_with(stop) {
                        final_tokens.truncate(final_tokens.len().saturating_sub(stop.len()));
                        break;
                    }
                }

                let text = ctx.tokenizer.detokenize(&final_tokens).trim().to_string();
                GenerationOutcome { text, logprobs }
                // tracer flushes all events to global tracer here on drop
            }
        };

        let mut high_temp_contexts = Vec::with_capacity(gen_seeds.len());
        for _ in 0..gen_seeds.len() {
            high_temp_contexts.push(prompt_state.ctx.fork());
        }

        let high_temp_futures: Vec<_> = gen_seeds
            .into_iter()
            .zip(high_temp_contexts)
            .map(|(seed, ctx)| {
                let seed_ids = prompt_state.seed_ids.clone();
                let seed_probs = prompt_state.seed_probs.clone();
                let stop_tokens_clone = stop_tokens.clone();
                let stop_cond_clone = stop_cond.clone();

                async move {
                    let mut local_rng = Xoshiro256StarStar::new(seed);
                    let mut ctx = ctx;

                    let (first_token, first_lp) = sample_from_distribution(
                        &seed_ids,
                        &seed_probs,
                        high_temperature,
                        &mut local_rng,
                    );
                    ctx.fill_token(first_token);
                    let mut generated = vec![first_token];
                    let mut logprobs = vec![first_lp];

                    while generated.len() < max_new_tokens && !stop_cond_clone.check(&generated) {
                        let dist = ctx.decode_step_dist().await;
                        let (token, lp_base) = sample_from_distribution(
                            &dist.ids,
                            &dist.probs,
                            high_temperature,
                            &mut local_rng,
                        );
                        ctx.fill_token(token);
                        generated.push(token);
                        logprobs.push(lp_base);
                    }

                    let mut final_tokens = generated;
                    for stop in &stop_tokens_clone {
                        if !stop.is_empty() && final_tokens.ends_with(stop) {
                            final_tokens.truncate(final_tokens.len().saturating_sub(stop.len()));
                            break;
                        }
                    }

                    let text = ctx.tokenizer.detokenize(&final_tokens).trim().to_string();
                    let avg_lp = average_logprob(&logprobs);
                    (text, avg_lp)
                    // tracer flushes all events to global tracer here on drop
                }
            })
            .collect();

        // Run ALL generations in parallel (low_temp + all high_temp)
        let gen_start = Instant::now();
        let (low_temp_outcome, high_temp_results) =
            futures::join!(low_temp_future, join_all(high_temp_futures));
        let generation_time = gen_start.elapsed();

        // Extract responses from high-temp results
        let mut responses = Vec::with_capacity(high_temp_results.len());
        let mut avg_log_likelihoods = Vec::with_capacity(high_temp_results.len());
        for (text, avg_lp) in high_temp_results {
            responses.push(text);
            avg_log_likelihoods.push(avg_lp);
        }

        let semantic_start = Instant::now();
        // Use base_tid 20+ for entailment threads (generation uses 0-11)
        let semantic_ids = get_semantic_ids(
            &responses,
            &mut judge,
            &example.question,
            &mut rng,
            entailment_temperature,
        )
        .await;
        let log_probs = logsumexp_by_id(&semantic_ids, &avg_log_likelihoods);
        let semantic_entropy = predictive_entropy_rao(&log_probs);
        let semantic_duration = semantic_start.elapsed();

        semantic_entropies.push(semantic_entropy);
        end_to_end_latencies.push(sample_start.elapsed().as_secs_f64());
        generation_times.push(generation_time.as_secs_f64());
        semantic_durations.push(semantic_duration.as_secs_f64());

        println!(
            "[semantic-check] idx={} duration_s={:.4} entropy={:.4}",
            idx,
            semantic_duration.as_secs_f64(),
            semantic_entropy
        );

        let (accuracy, best_f1) = compute_accuracy(
            &low_temp_outcome.text,
            example,
            metric_name.as_str(),
            &model,
            &stop_tokens,
            &mut rng,
        )
        .await;
        per_sample_f1.push((idx, best_f1, accuracy));
        if (accuracy - 1.0).abs() < f32::EPSILON {
            correct_count += 1;
        } else {
            incorrect_count += 1;
        }
        sample_outputs.push(serde_json::json!({
            "idx": idx,
            "acc": accuracy,
            "question": example.question,
            "prediction": low_temp_outcome.text,
            "references": example.answer_texts(),
        }));
        hallucination_labels.push(1u8.saturating_sub(accuracy as u8));
    }

    if semantic_entropies.is_empty() {
        println!("No validation samples found; skipping metric computation.");
        return Ok("{}".to_string());
    }

    let (threshold, f1) = pick_threshold(&semantic_entropies, &hallucination_labels);
    let predictions: Vec<u8> = semantic_entropies
        .iter()
        .map(|score| if *score >= threshold { 1 } else { 0 })
        .collect();
    let (tn, fp, fn_, tp) = confusion(&hallucination_labels, &predictions);

    let auroc = compute_auroc(&hallucination_labels, &semantic_entropies);
    // print hallucination_labes and semantic_entropies one by to to stderr
    for (label, entropy) in hallucination_labels.iter().zip(semantic_entropies.iter()) {
        eprintln!("{} {}", label, entropy);
    }

    let latency_summary = serde_json::json!({
        "avg_end_to_end_s": mean(&end_to_end_latencies),
        "avg_generation_time_s": mean(&generation_times),
        "avg_semantic_check_s": mean(&semantic_durations),

        "median_end_to_end_s": median(&mut end_to_end_latencies.clone()),
        "median_generation_time_s": median(&mut generation_times.clone()),
        "median_semantic_check_s": median(&mut semantic_durations.clone()),
    });

    let summary = serde_json::json!({
        "config": config_path.unwrap_or_else(|| "builtin-config".to_string()),
        "model_name": gen_cfg.model_name.unwrap_or_else(|| "llama-3.1-8b-instruct".to_string()),
        "dataset": dataset_name,
        "metric": metric_name,
        "num_samples": num_samples,
        "threshold": threshold,
        "f1": f1,
        "auroc": auroc,
        "accuracy": {
            "correct": correct_count,
            "incorrect": incorrect_count,
        },
        "counts": { "tp": tp, "tn": tn, "fp": fp, "fn": fn_ },
        "latency": latency_summary,
        "sample_outputs": sample_outputs,
    });

    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok("finished".to_string())
}

fn get_brief_text(name: &str) -> &'static str {
    match name {
        "chat" => CHAT_BRIEF,
        _ => DEFAULT_BRIEF,
    }
}

fn make_prompt(
    use_context: bool,
    brief_always: bool,
    brief: &str,
    context: Option<&str>,
    question: &str,
    answer: Option<&str>,
) -> String {
    let mut prompt = String::new();
    if brief_always {
        prompt.push_str(brief);
    }
    if use_context && let Some(ctx) = context {
        prompt.push_str(&format!("Context: {}\n", ctx));
    }
    prompt.push_str(&format!("Question: {}\n", question));
    if let Some(ans) = answer {
        prompt.push_str(&format!("Answer: {}\n\n", ans));
    } else {
        prompt.push_str("Answer:");
    }
    prompt
}

fn construct_fewshot_prompt(
    dataset: &[QAExample],
    example_indices: &[usize],
    brief: &str,
    brief_always: bool,
    use_context: bool,
) -> String {
    let mut prompt = if !brief_always {
        brief.to_string()
    } else {
        String::new()
    };

    for &idx in example_indices {
        if let Some(example) = dataset.get(idx) {
            let answer = example.answer_texts().first().cloned().unwrap_or_default();
            prompt.push_str(&make_prompt(
                use_context,
                brief_always,
                brief,
                example.context.as_deref(),
                &example.question,
                Some(&answer),
            ));
        }
    }
    prompt
}

async fn prepare_prompt_state(model: &inferlet::Model, prompt: &str) -> PromptState {
    let mut ctx = model.create_context();
    ctx.fill(prompt);
    let start = Instant::now();
    let seed_dist = ctx.decode_step_dist().await;
    let ttft = start.elapsed();
    PromptState {
        ctx,
        seed_ids: seed_dist.ids,
        seed_probs: seed_dist.probs,
        ttft,
    }
}

fn apply_temperature(probs: &[f32], temperature: f32) -> Vec<f64> {
    if probs.is_empty() {
        return vec![1.0];
    }

    if temperature <= 0.0 {
        let idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let mut adjusted = vec![0.0; probs.len()];
        adjusted[idx] = 1.0;
        return adjusted;
    }

    let inv_t = 1.0f32 / temperature.max(1e-5);
    let mut adjusted: Vec<f64> = probs
        .iter()
        .map(|p| (*p as f64).powf(inv_t as f64))
        .collect();
    let sum: f64 = adjusted.iter().sum();
    if sum > 0.0 {
        for p in adjusted.iter_mut() {
            *p /= sum;
        }
    }
    adjusted
}

fn sample_from_distribution(
    ids: &[u32],
    probs: &[f32],
    temperature: f32,
    rng: &mut Xoshiro256StarStar,
) -> (u32, f32) {
    let adjusted = apply_temperature(probs, temperature);
    let idx = if let Ok(dist) = WeightedIndex::new(&adjusted) {
        dist.sample(rng)
    } else {
        println!(
            "sampling failed, got result: {:?}\n\t, adjusted is {:?}\n\t, probs is {:?}",
            WeightedIndex::new(&adjusted),
            adjusted,
            probs
        );
        adjusted
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    };
    let token_id = ids.get(idx).copied().unwrap_or_default();
    let prob_base = probs
        .get(idx)
        .copied()
        .unwrap_or(1e-9)
        .max(f32::MIN_POSITIVE);
    (token_id, prob_base.ln())
}

async fn generate_with_state<C: inferlet::stop_condition::StopCondition>(
    prompt_state: &PromptState,
    temperature: f32,
    max_new_tokens: usize,
    stop_condition: &C,
    stop_tokens: &[Vec<u32>],
    rng: &mut Xoshiro256StarStar,
) -> GenerationOutcome {
    let mut ctx = prompt_state.ctx.fork();
    let mut generated = Vec::new();
    let mut logprobs = Vec::new();

    let (first_token, first_lp_base) = sample_from_distribution(
        &prompt_state.seed_ids,
        &prompt_state.seed_probs,
        temperature,
        rng,
    );
    ctx.fill_token(first_token);
    generated.push(first_token);
    logprobs.push(first_lp_base);

    while generated.len() < max_new_tokens && !stop_condition.check(&generated) {
        let dist = ctx.decode_step_dist().await;
        let (token, lp_base) = sample_from_distribution(&dist.ids, &dist.probs, temperature, rng);
        ctx.fill_token(token);
        generated.push(token);
        logprobs.push(lp_base);
    }

    // Remove any trailing stop-token sequences and trim whitespace like the Python version.
    let mut final_tokens = generated;
    for stop in stop_tokens {
        if stop.is_empty() {
            continue;
        }
        if final_tokens.ends_with(stop) {
            let new_len = final_tokens.len().saturating_sub(stop.len());
            final_tokens.truncate(new_len);
            break;
        }
    }

    let text = ctx.tokenizer.detokenize(&final_tokens).trim().to_string();
    GenerationOutcome { text, logprobs }
}

fn average_logprob(logprobs: &[f32]) -> f32 {
    if logprobs.is_empty() {
        return f32::NEG_INFINITY;
    }
    logprobs.iter().sum::<f32>() / (logprobs.len() as f32)
}

impl<'a> EntailmentJudge<'a> {
    fn new(model: &'a inferlet::Model, max_new_tokens: usize, strict: bool) -> Self {
        Self {
            model,
            max_new_tokens,
            strict,
            cache: HashMap::new(),
            global_prefix_ctx: None,
            question_prefix_cache: None,
        }
    }

    /// Initialize the global prefix context (invariant part shared across all questions).
    /// Call this once at startup to prefill the common prefix.
    async fn init_global_prefix(&mut self) {
        if self.global_prefix_ctx.is_some() {
            return;
        }
        // The invariant prefix shared across ALL entailment checks
        const GLOBAL_PREFIX: &str = "We are evaluating answers to the question ";
        let mut ctx = self.model.create_context();
        ctx.fill(GLOBAL_PREFIX);
        ctx.flush().await;
        self.global_prefix_ctx = Some(ctx);
    }

    /// Prepare (or retrieve from cache) the prefilled context for a question's entailment prefix.
    /// Uses two-level caching: global invariant prefix + per-question suffix.
    async fn get_or_prepare_prefix_context(&mut self, question: &str) -> Context {
        // Check if we already have a cached prefix for this question
        if let Some((ref cached_question, ref cached_ctx)) = self.question_prefix_cache
            && cached_question == question
        {
            return cached_ctx.fork();
        }

        // Ensure global prefix is initialized
        self.init_global_prefix().await;

        // Fork from global prefix and add question-specific suffix
        let mut ctx = self.global_prefix_ctx.as_ref().unwrap().fork();
        let suffix = format!(
            "\"{}\"\nHere are two possible answers, does Possible Answer 1 semantically entail Possible Answer 2? Respond only with entailment, contradiction, or neutral.\nPossible Answer 1: ",
            question
        );
        ctx.fill(&suffix);
        ctx.flush().await;

        // Cache this prefix context for reuse
        self.question_prefix_cache = Some((question.to_string(), ctx.fork()));

        ctx
    }

    /// Check a batch of implication pairs in parallel.
    /// Returns a vector of results (0=contradiction, 1=neutral, 2=entailment) in the same order.
    /// Trace events are recorded automatically via the global tracer.
    async fn check_implications_batch(
        &mut self,
        pairs: &[(usize, usize)], // indices into strings_list
        strings_list: &[String],
        question: &str,
        rng: &mut Xoshiro256StarStar,
        entailment_temperature: f32,
    ) -> Vec<((usize, usize), u8)> {
        if pairs.is_empty() {
            return Vec::new();
        }

        // First, check cache and filter out already-known pairs
        let mut results: Vec<((usize, usize), u8)> = Vec::new();
        let mut uncached_pairs: Vec<(usize, usize)> = Vec::new();

        for &(i, j) in pairs {
            let text1 = &strings_list[i];
            let text2 = &strings_list[j];
            // Short-circuit identical strings to avoid unnecessary entailment calls.
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

        // Get the prefix context (will be cached for this question)
        let prefix_ctx = self.get_or_prepare_prefix_context(question).await;

        // Pre-generate unique seeds for determinism (each pair gets unique seed)
        let seeds: Vec<u64> = uncached_pairs.iter().map(|_| rng.next_u64()).collect();

        // Create futures for all uncached pairs - each gets its own thread ID for tracing
        let futures: Vec<_> = uncached_pairs
            .iter()
            .zip(seeds.into_iter())
            .enumerate()
            .map(|(idx, (&(i, j), _seed))| {
                let text1 = strings_list[i].clone();
                let text2 = strings_list[j].clone();
                let mut ctx_clone = prefix_ctx.fork();
                let eos_tokens = self.model.eos_tokens();
                let max_tokens = self.max_new_tokens;

                async move {
                    let suffix = format!("{}\nPossible Answer 2: {}\nResponse:", text1, text2);
                    ctx_clone.fill(&suffix);
                    // ctx_clone.flush().await;

                    // Use ctx.generate() instead of manual decode loop - entailment doesn't need log probs
                    let stop_cond = ends_with_any(eos_tokens.clone()).or(max_len(max_tokens));
                    // println!("{idx}: entailment_temperature: {}", entailment_temperature);
                    let sampler = Sampler::top_p(entailment_temperature, 0.95);

                    // printout ctx_clone's info here
                    // println!("{idx}: ctx_clone: {:?}", ctx_clone);
                    let response = ctx_clone.generate(sampler, stop_cond).await;
                    // println!("{idx}: done");

                    // Match Python: check first 30 chars of lowercased response
                    let binary_response: String = response.chars().take(30).collect();
                    let result = if binary_response.contains("entailment") {
                        2u8
                    } else if binary_response.contains("neutral") {
                        1u8
                    } else if binary_response.contains("contradiction") {
                        0u8
                    } else {
                        // Python: logging.warning('MANUAL NEUTRAL!') and returns 1
                        1u8
                    };

                    ((i, j), result, text1, text2)
                    // tracer flushes all events to global tracer here on drop
                }
            })
            .collect();

        // Run all in parallel
        let batch_results = join_all(futures).await;

        // Update cache and collect results
        for ((i, j), result, text1, text2) in batch_results {
            self.cache.insert((text1, text2), result);
            results.push(((i, j), result));
        }

        results
    }
}

/// Compute semantic IDs with batched entailment checking for maximum parallelism.
/// Matches Python's get_semantic_ids exactly (no heuristic shortcuts).
/// Trace events are recorded via global tracer.
async fn get_semantic_ids(
    strings_list: &[String],
    judge: &mut EntailmentJudge<'_>,
    question: &str,
    rng: &mut Xoshiro256StarStar,
    entailment_temperature: f32,
) -> Vec<usize> {
    let n = strings_list.len();
    if n == 0 {
        return Vec::new();
    }

    // Collect ALL pairs that need checking (both directions for bidirectional entailment)
    // No heuristic shortcuts - matches Python implementation
    let mut pairs_to_check: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            // Need to check both directions: (i,j) and (j,i)
            pairs_to_check.push((i, j));
            pairs_to_check.push((j, i));
        }
    }

    // Batch check all pairs in parallel
    let implication_results = judge
        .check_implications_batch(
            &pairs_to_check,
            strings_list,
            question,
            rng,
            entailment_temperature,
        )
        .await;

    // Build lookup map for implications
    let mut implications: HashMap<(usize, usize), u8> = HashMap::new();
    for ((i, j), result) in implication_results {
        implications.insert((i, j), result);
    }

    // Determine equivalence based on implications (matches Python's are_equivalent)
    let is_equivalent = |i: usize, j: usize| -> bool {
        let imp_ij = implications.get(&(i, j)).copied().unwrap_or(1);
        let imp_ji = implications.get(&(j, i)).copied().unwrap_or(1);

        // Python asserts these are in [0,1,2]
        debug_assert!(imp_ij <= 2 && imp_ji <= 2);

        if judge.strict {
            // strict_entailment: both must be entailment (2)
            imp_ij == 2 && imp_ji == 2
        } else {
            // Check if none are contradiction (0) and not both neutral (1)
            imp_ij != 0 && imp_ji != 0 && !(imp_ij == 1 && imp_ji == 1)
        }
    };

    // Build semantic IDs (matches Python's algorithm exactly)
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

    // Python asserts -1 not in semantic_set_ids
    debug_assert!(!semantic_set_ids.contains(&-1));

    semantic_set_ids.into_iter().map(|x| x as usize).collect()
}

fn logsumexp_by_id(semantic_ids: &[usize], log_likelihoods: &[f32]) -> Vec<f32> {
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

fn predictive_entropy_rao(log_probs: &[f32]) -> f32 {
    -log_probs.iter().map(|lp| lp.exp() * lp).sum::<f32>()
}

fn pick_threshold(scores: &[f32], labels: &[u8]) -> (f32, f64) {
    let mut unique: Vec<f32> = scores.to_vec();
    unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique.dedup();
    if unique.len() > 200 {
        let min = *unique.first().unwrap_or(&0.0);
        let max = *unique.last().unwrap_or(&1.0);
        unique = (0..200)
            .map(|i| min + (max - min) * (i as f32) / 199.0)
            .collect();
    }

    let mut best_threshold = *unique.first().unwrap_or(&0.0);
    let mut best_f1 = -1.0;
    for t in unique {
        let preds: Vec<u8> = scores.iter().map(|s| if *s >= t { 1 } else { 0 }).collect();
        let f1 = binary_f1(labels, &preds);
        if f1 > best_f1 {
            best_f1 = f1;
            best_threshold = t;
        }
    }
    (best_threshold, best_f1)
}

fn binary_f1(labels: &[u8], preds: &[u8]) -> f64 {
    let (_tn, fp, fn_, tp) = confusion(labels, preds);
    let precision = tp as f64 / ((tp + fp) as f64 + f64::EPSILON);
    let recall = tp as f64 / ((tp + fn_) as f64 + f64::EPSILON);
    if (precision + recall) == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

fn compute_auroc(labels: &[u8], scores: &[f32]) -> f64 {
    if labels.len() != scores.len() {
        return f64::NAN;
    }

    let pos = labels.iter().filter(|&&l| l == 1).count();
    let neg = labels.len() - pos;

    if pos == 0 || neg == 0 {
        return f64::NAN;
    }

    let mut pairs: Vec<(f32, u8)> = scores.iter().cloned().zip(labels.iter().cloned()).collect();

    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut tp = 0f64;
    let mut fp = 0f64;
    let pos_f = pos as f64;
    let neg_f = neg as f64;

    let mut fpr_prev = 0f64;
    let mut tpr_prev = 0f64;
    let mut auc = 0f64;

    let mut i = 0usize;
    while i < pairs.len() {
        let score = pairs[i].0;

        while i < pairs.len() && pairs[i].0 == score {
            if pairs[i].1 == 1 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            i += 1;
        }

        let tpr = tp / pos_f;
        let fpr = fp / neg_f;

        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0;

        fpr_prev = fpr;
        tpr_prev = tpr;
    }

    auc
}

fn confusion(labels: &[u8], preds: &[u8]) -> (usize, usize, usize, usize) {
    let mut tp = 0usize;
    let mut tn = 0usize;
    let mut fp = 0usize;
    let mut fn_ = 0usize;
    for (label, pred) in labels.iter().zip(preds.iter()) {
        match (*label, *pred) {
            (1, 1) => tp += 1,
            (0, 0) => tn += 1,
            (0, 1) => fp += 1,
            (1, 0) => fn_ += 1,
            _ => {}
        }
    }
    (tn, fp, fn_, tp)
}

fn build_llm_metric_prompt(predicted_answer: &str, example: &QAExample) -> String {
    let answers = example.answer_texts();
    let mut prompt = format!(
        "We are assessing the quality of answers to the following question: {}\n",
        example.question
    );
    if answers.len() == 1 {
        prompt += &format!("The expected answer is: {}.\n", answers[0]);
    } else {
        prompt += &format!(
            "The following are expected answers to this question: {:?}.\n",
            answers
        );
    }
    prompt += &format!("The proposed answer is: {}\n", predicted_answer);
    prompt += "Within the context of the question, does the proposed answer mean the same as the expected answer(s)? Respond only with yes or no.\nResponse:";
    prompt
}

async fn generate_short_answer(
    model: &inferlet::Model,
    prompt: &str,
    temperature: f32,
    max_new_tokens: usize,
    stop_tokens: &[Vec<u32>],
    rng: &mut Xoshiro256StarStar,
) -> String {
    let prompt_state = prepare_prompt_state(model, prompt).await;
    let stop_cond = ends_with_any(stop_tokens.to_vec()).or(max_len(max_new_tokens));
    let out = generate_with_state(
        &prompt_state,
        temperature,
        max_new_tokens,
        &stop_cond,
        stop_tokens,
        rng,
    )
    .await;
    out.text
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

fn normalize_answer(text: &str) -> Vec<String> {
    // Mirror SQuAD-style normalization: lowercase, remove punctuation and articles,
    // collapse whitespace. Add light canonicalization for plurals and small numbers
    // to reduce near-miss mismatches.
    fn remove_punc(s: &str) -> String {
        s.chars()
            .map(|c| if c.is_ascii_punctuation() { ' ' } else { c })
            .collect()
    }
    fn remove_articles(s: &str) -> String {
        let mut out = Vec::new();
        for token in s.split_whitespace() {
            if token.eq_ignore_ascii_case("a")
                || token.eq_ignore_ascii_case("an")
                || token.eq_ignore_ascii_case("the")
            {
                continue;
            }
            out.push(token);
        }
        out.join(" ")
    }

    fn canonical_token(tok: &str) -> String {
        // Map number words/digits to a single canonical form.
        let num_map = [
            ("zero", "0"),
            ("one", "1"),
            ("two", "2"),
            ("three", "3"),
            ("four", "4"),
            ("five", "5"),
            ("six", "6"),
            ("seven", "7"),
            ("eight", "8"),
            ("nine", "9"),
            ("ten", "10"),
            ("eleven", "11"),
            ("twelve", "12"),
        ];
        for (word, digit) in num_map.iter() {
            if tok == *word || tok == *digit {
                return digit.to_string();
            }
        }

        // Simple plural/variant handling.
        let mut t = tok.to_string();
        if t.ends_with("ies") && t.len() > 3 {
            t.truncate(t.len() - 3);
            t.push('y');
            return t;
        }
        if t.ends_with("es") && t.len() > 3 {
            t.truncate(t.len() - 2);
            return t;
        }
        if t.ends_with('s') && t.len() > 3 {
            t.truncate(t.len() - 1);
            return t;
        }
        t
    }

    let lower = text.to_ascii_lowercase();
    let no_punc = remove_punc(&lower);
    let no_articles = remove_articles(&no_punc);
    no_articles
        .split_whitespace()
        .filter(|s| !s.is_empty())
        .map(canonical_token)
        .collect()
}

fn f1_against_answer(predicted: &str, answer: &str) -> f32 {
    use std::cmp::min;

    let pred_tokens = normalize_answer(predicted);
    let ans_tokens = normalize_answer(answer);
    if pred_tokens.is_empty() && ans_tokens.is_empty() {
        return 1.0;
    }
    if pred_tokens.is_empty() || ans_tokens.is_empty() {
        return 0.0;
    }

    // Bag-of-words overlap like official SQuAD F1.
    let mut overlap = 0usize;
    let mut pred_counts = HashMap::new();
    for t in pred_tokens.iter() {
        *pred_counts.entry(t).or_insert(0usize) += 1;
    }
    let mut ans_counts = HashMap::new();
    for t in ans_tokens.iter() {
        *ans_counts.entry(t).or_insert(0usize) += 1;
    }
    for (tok, pc) in pred_counts.iter() {
        if let Some(ac) = ans_counts.get(tok) {
            overlap += min(*pc, *ac);
        }
    }
    if overlap == 0 {
        return 0.0;
    }
    let precision = overlap as f32 / pred_tokens.len() as f32;
    let recall = overlap as f32 / ans_tokens.len() as f32;
    2.0 * precision * recall / (precision + recall)
}

async fn compute_accuracy(
    predicted_answer: &str,
    example: &QAExample,
    metric: &str,
    model: &inferlet::Model,
    stop_tokens: &[Vec<u32>],
    rng: &mut Xoshiro256StarStar,
) -> (f32, f32) {
    if metric == "llm" {
        let prompt = build_llm_metric_prompt(predicted_answer, example);
        let response = generate_short_answer(model, &prompt, 0.01, 32, stop_tokens, rng).await;
        let acc = if response.to_ascii_lowercase().contains("yes") {
            1.0
        } else {
            0.0
        };
        return (acc, 0.0);
    }

    let answers = example.answer_texts();
    if answers.is_empty() {
        return (0.0, 0.0);
    }
    let best = answers
        .iter()
        .map(|ans| f1_against_answer(predicted_answer, ans))
        .fold(0.0f32, f32::max);
    // Python code uses SQuAD metric threshold of 50 (percent).
    let acc = if best * 100.0 >= 50.0 { 1.0 } else { 0.0 };
    (acc, best * 100.0)
}
