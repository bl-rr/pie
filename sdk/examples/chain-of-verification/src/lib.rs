//! Chain-of-Verification (CoVe) inferlet for reducing LLM hallucinations.
//!
//! Implements the 4-phase CoVe pipeline with PIE's async parallelism:
//! 1. Route question to appropriate chain type (WIKI/MULTI/LONG)
//! 2. Generate baseline response
//! 3. Generate verification questions
//! 4. Execute verification (PARALLEL via join_all, with optional HTTP search)
//! 5. Generate final refined answer

pub mod util;

use futures::future::join_all;
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use inferlet::stop_condition::{ends_with_any, max_len, StopCondition};
use inferlet::wstd::http::Client;
use inferlet::{Args, Result, Sampler};
use std::time::Instant;

use crate::util::*;

const BATCH_PAYLOAD_CHUNK_BYTES: usize = 4096;

// ============================================================================
// Prompts (matching Python prompts.py exactly)
// ============================================================================

const ROUTER_PROMPT: &str = r#"Please classify the below question in one of the following categories. Output ONLY the category name, nothing else.

Categories:
WIKI_CHAIN: Good for answering questions which asks for a list or set of entities as its answer. 
MULTI_CHAIN: Good for answering questions which comprises of questions that have multiple independent answers (derived from a series of multiple discontiguous spans in the text) and multiple questions are asked in the original question.
LONG_CHAIN: Good for answering questions whose answer is long.

Examples:
WIKI_CHAIN: "Name some Endemic orchids of Vietnam." | "Who are the scientists who won nobel prize in the year 1970?" | "List some cricket players who are playing in indian cricket team."
MULTI_CHAIN: "Who is known for developing the theory of relativity, and in which year was it introduced?" | "Who is credited with inventing the telephone, and when did this invention take place?"
LONG_CHAIN: "Write few lines about Einstein." | "Tell me in short about first moon landing." | "Write a short biography of Carl Marx."

Question: {question}

Category:"#;

const BASELINE_PROMPT_WIKI: &str = r#"Answer the below question which is asking for a list of entities (names, places, locations etc). Output ONLY a numbered list with at most {max_items} items. Each line must contain only the concise entity name or short noun phrase. NO ADDITIONAL DETAILS, EXPLANATIONS, OR DUPLICATES.

Question: {question}

Answer:"#;

const BASELINE_PROMPT_MULTI: &str = r#"Answer the below question correctly and in a concise manner without much details. Only answer what the question is asked.

Question: {question}

Answer:"#;

const BASELINE_PROMPT_LONG: &str = r#"Answer the below question correctly.

Question: {question}

Answer:"#;

const VERIFICATION_QUESTION_TEMPLATE_PROMPT_WIKI: &str = r#"Your task is to create a verification question based on the below question provided. Output ONLY one short verification question.
Example Question: Who are some movie actors who were born in Boston?
Example Verification Question: Was [movie actor] born in [Boston]
Explanation: In the above example the verification question focused only on the ANSWER_ENTITY (name of the movie actor) and QUESTION_ENTITY (birth place).
Similarly you need to focus on the ANSWER_ENTITY and QUESTION_ENTITY from the actual question and generate verification question.

Actual Question: {question}

Final Verification Question:"#;

const VERIFICATION_QUESTION_PROMPT_WIKI: &str = r#"Your task is to create a series of verification questions based on the below question, the verification question template and baseline response. Output ONLY a numbered list with at most {max_items} short verification questions and no explanation.
Example Question: Who are some movie actors who were born in Boston?
Example Verification Question Template: Was [movie actor] born in Boston?
Example Baseline Response: 1. Matt Damon - Famous for his roles in films like "Good Will Hunting," "The Bourne Identity" series, and "The Martian," Damon is an Academy Award-winning actor, screenwriter, and producer.
2. Chris Evans - Famous for his portrayal of Captain America in the Marvel Cinematic Universe, Evans has also appeared in movies like "Snowpiercer" and "Knives Out."
Verification questions: 1. Was Matt Damon born in Boston?
2. Was Chris Evans born in Boston?
etc.
Example Verification Question: 1. Was Matt Damon born in Boston?
2. Was Chris Evans born in Boston?

Explanation: In the above example the verification questions focused only on the ANSWER_ENTITY (name of the movie actor) and QUESTION_ENTITY (birth place) based on the template and substitutes entity values from the baseline response.
Similarly you need to focus on the ANSWER_ENTITY and QUESTION_ENTITY from the actual question and substitute the entity values from the baseline response to generate verification questions.

Actual Question: {question}
Baseline Response: {baseline_response}
Verification Question Template: {verification_question_template}

Final Verification Questions:"#;

const VERIFICATION_QUESTION_PROMPT_MULTI: &str = r#"Your task is to create verification questions based on the below original question and the baseline response. The verification questions are meant for verifying the factual accuracy in the baseline response.
Example Question: Who invented the first printing press and in what year?
Example Baseline Response: Johannes Gutenberg, 1450.
Example Verification Questions: 1. Did Johannes Gutenberg invent first printing press?
2. Did Johannes Gutenberg invent first printing press in the year 1450?

Explanation: The verification questions are highly aligned with both the actual question and baseline response. The actual question comprises of multiple independent questions which in turn has multiple independent answers in the baseline response. Hence, the verification questions should also be independent for factual verification.

Actual Question: {question}
Baseline Response: {baseline_response}

Final Verification Questions:"#;

const VERIFICATION_QUESTION_PROMPT_LONG: &str = r#"Your task is to create verification questions based on the below original question and the baseline response. The verification questions are meant for verifying the factual accuracy in the baseline response. Output should be numbered list of verification questions.

Actual Question: {question}
Baseline Response: {baseline_response}

Final Verification Questions:"#;

const EXECUTE_PLAN_PROMPT_SEARCH: &str = r#"Answer the following question correctly based on the provided context. The question could be tricky as well, so think step by step and answer it correctly.

Context: {search_result}

Question: {verification_question}

Answer:"#;

const EXECUTE_PLAN_PROMPT_SELF: &str = r#"Answer the following question correctly.

Question: {verification_question}

Answer:"#;

const FINAL_REFINED_PROMPT: &str = r#"Given the below `Original Query` and `Baseline Answer`, analyze the `Verification Questions & Answers` to finally filter the refined answer.
Original Query: {question}
Baseline Answer: {baseline_response}

Verification Questions & Answer Pairs:
{verification_answers}

Final Refined Answer:"#;

const FINAL_REFINED_PROMPT_WIKI: &str = r#"Given the below `Original Query` and `Baseline Answer`, analyze the `Verification Questions & Answers` to finally filter the refined answer. Output ONLY a numbered list with at most {max_items} concise verified entities. Each line must contain only the entity name or short noun phrase. NO EXPLANATIONS, DUPLICATES, OR EXTRA PROSE.
Original Query: {question}
Baseline Answer: {baseline_response}

Verification Questions & Answer Pairs:
{verification_answers}

Final Refined Answer:"#;

const DEFAULT_STOP_SEQUENCES: [&str; 5] =
    ["\n\n", "\nQuestion:", "Question:", "\nAnswer:", "Answer:"];

// ============================================================================
// Chain Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum ChainType {
    Wiki,
    Multi,
    Long,
}

impl ChainType {
    fn from_str(s: &str) -> Self {
        let s_upper = s.to_uppercase();
        if s_upper.contains("WIKI") {
            ChainType::Wiki
        } else if s_upper.contains("LONG") {
            ChainType::Long
        } else {
            ChainType::Multi
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            ChainType::Wiki => "WIKI",
            ChainType::Multi => "MULTI",
            ChainType::Long => "LONG",
        }
    }
}

struct FactorReviseResult {
    verification_questions: Vec<String>,
    verification_answers: Vec<(String, String)>,
    cross_checks: Vec<(String, String)>,
    final_answer: String,
    verification_questions_ms: u128,
    execute_verification_ms: u128,
    cross_check_ms: u128,
    final_answer_ms: u128,
}

struct PassageFactorReviseResult {
    idx: usize,
    passage: String,
    verification_questions: Vec<String>,
    verification_answers: Vec<(String, String)>,
    cross_check: String,
    revised_passage: String,
    verification_questions_ms: u128,
    execute_verification_ms: u128,
    cross_check_ms: u128,
    final_answer_ms: u128,
}

struct TextStopTrace {
    raw_text: String,
    trimmed_text: String,
    matched_stop_suffix: Option<String>,
    raw_token_ids: Vec<u32>,
    token_timestamps_ms: Vec<f64>,
    total_elapsed_ms: f64,
}

struct CachedPrompt {
    prefix_ctx: inferlet::Context,
    suffix: String,
}

impl CachedPrompt {
    async fn new(model: &inferlet::Model, template: &str, placeholder: &str) -> Self {
        let (prefix, suffix) = template
            .split_once(placeholder)
            .unwrap_or_else(|| panic!("Prompt missing placeholder '{}'", placeholder));
        let mut prefix_ctx = model.create_context();
        prefix_ctx.fill(prefix);
        prefix_ctx.flush().await;
        Self {
            prefix_ctx,
            suffix: suffix.to_string(),
        }
    }

    fn fork_with_suffix(&self, dynamic_text: &str) -> inferlet::Context {
        self.fork_with_tail(&format!("{}{}", dynamic_text, self.suffix))
    }

    fn fork_with_tail(&self, tail: &str) -> inferlet::Context {
        let mut ctx = self.prefix_ctx.fork();
        ctx.fill(tail);
        ctx
    }

    fn suffix(&self) -> &str {
        &self.suffix
    }
}

struct PromptCache {
    router: CachedPrompt,
    baseline_wiki: CachedPrompt,
    baseline_multi: CachedPrompt,
    baseline_long: CachedPrompt,
    verification_template_wiki: CachedPrompt,
    verification_wiki: CachedPrompt,
    verification_multi: CachedPrompt,
    verification_long: CachedPrompt,
    execute_search: CachedPrompt,
    execute_self: CachedPrompt,
    final_refined_default: CachedPrompt,
    final_refined_wiki: CachedPrompt,
}

impl PromptCache {
    async fn new(model: &inferlet::Model, wiki_max_items: usize) -> Self {
        let wiki_max_items = wiki_max_items.max(1).to_string();
        let baseline_wiki = BASELINE_PROMPT_WIKI.replace("{max_items}", &wiki_max_items);
        let verification_wiki =
            VERIFICATION_QUESTION_PROMPT_WIKI.replace("{max_items}", &wiki_max_items);
        let final_refined_wiki = FINAL_REFINED_PROMPT_WIKI.replace("{max_items}", &wiki_max_items);
        Self {
            router: CachedPrompt::new(model, ROUTER_PROMPT, "{question}").await,
            baseline_wiki: CachedPrompt::new(model, &baseline_wiki, "{question}").await,
            baseline_multi: CachedPrompt::new(model, BASELINE_PROMPT_MULTI, "{question}").await,
            baseline_long: CachedPrompt::new(model, BASELINE_PROMPT_LONG, "{question}").await,
            verification_template_wiki: CachedPrompt::new(
                model,
                VERIFICATION_QUESTION_TEMPLATE_PROMPT_WIKI,
                "{question}",
            )
            .await,
            verification_wiki: CachedPrompt::new(model, &verification_wiki, "{question}").await,
            verification_multi: CachedPrompt::new(
                model,
                VERIFICATION_QUESTION_PROMPT_MULTI,
                "{question}",
            )
            .await,
            verification_long: CachedPrompt::new(
                model,
                VERIFICATION_QUESTION_PROMPT_LONG,
                "{question}",
            )
            .await,
            execute_search: CachedPrompt::new(model, EXECUTE_PLAN_PROMPT_SEARCH, "{search_result}")
                .await,
            execute_self: CachedPrompt::new(
                model,
                EXECUTE_PLAN_PROMPT_SELF,
                "{verification_question}",
            )
            .await,
            final_refined_default: CachedPrompt::new(model, FINAL_REFINED_PROMPT, "{question}")
                .await,
            final_refined_wiki: CachedPrompt::new(model, &final_refined_wiki, "{question}").await,
        }
    }

    fn baseline_prompt(&self, chain_type: ChainType) -> &CachedPrompt {
        match chain_type {
            ChainType::Wiki => &self.baseline_wiki,
            ChainType::Multi => &self.baseline_multi,
            ChainType::Long => &self.baseline_long,
        }
    }

    fn final_prompt(&self, chain_type: ChainType) -> &CachedPrompt {
        match chain_type {
            ChainType::Wiki => &self.final_refined_wiki,
            ChainType::Multi | ChainType::Long => &self.final_refined_default,
        }
    }
}

struct StopSequences {
    default_text: Vec<String>,
    default_tokens: Vec<Vec<u32>>,
    wiki_list_text: Vec<String>,
    wiki_list_tokens: Vec<Vec<u32>>,
}

impl StopSequences {
    fn new(model: &inferlet::Model, wiki_max_items: usize) -> Self {
        let tokenizer = model.get_tokenizer();
        let default_text = DEFAULT_STOP_SEQUENCES
            .iter()
            .map(|stop| (*stop).to_string())
            .collect::<Vec<_>>();
        let mut default_tokens = model.eos_tokens();
        for stop in &default_text {
            default_tokens.push(tokenizer.tokenize(stop));
        }

        let next_item = wiki_max_items.max(1) + 1;
        let mut wiki_list_text = default_text.clone();
        wiki_list_text.push(format!("\n{}.", next_item));
        wiki_list_text.push(format!("\n{})", next_item));
        let mut wiki_list_tokens = default_tokens.clone();
        for stop in wiki_list_text.iter().skip(default_text.len()) {
            wiki_list_tokens.push(tokenizer.tokenize(stop));
        }

        Self {
            default_text,
            default_tokens,
            wiki_list_text,
            wiki_list_tokens,
        }
    }

    fn default_tokens(&self) -> &[Vec<u32>] {
        &self.default_tokens
    }

    fn default_text(&self) -> &[String] {
        &self.default_text
    }

    fn wiki_list_tokens(&self) -> &[Vec<u32>] {
        &self.wiki_list_tokens
    }

    fn wiki_list_text(&self) -> &[String] {
        &self.wiki_list_text
    }
}

struct CompletedEvaluation {
    idx: usize,
    refs: Vec<String>,
    result: serde_json::Value,
    baseline_score: f64,
    cove_score: f64,
    baseline_pass: Option<bool>,
    cove_pass: Option<bool>,
    request_latency_s: f64,
}

// ============================================================================
// Main Entry Point
// ============================================================================

#[inferlet::main]
async fn main(mut args: Args) -> Result<String> {
    let config_path: Option<String> = args.opt_value_from_str(["-c", "--config"])?;
    let single_question: Option<String> = args.opt_value_from_str(["-q", "--question"])?;
    let debug_prompt: Option<String> = args.opt_value_from_str("--debug-prompt")?;
    let debug_stop_json: Option<String> = args.opt_value_from_str("--debug-stop-json")?;
    let debug_max_tokens: Option<usize> = args.opt_value_from_str("--debug-max-tokens")?;
    let show_intermediate: bool = args.contains(["-v", "--verbose"]);
    let single_question_debug = single_question.is_some();
    match std::env::current_dir() {
        Ok(path) => eprintln!("[cove-debug] current_dir={}", path.display()),
        Err(err) => eprintln!("[cove-debug] current_dir error: {}", err),
    }
    if let Some(path) = config_path.as_deref() {
        eprintln!("[cove-debug] config_arg={}", path);
    }

    // Load configuration. Single-question mode can use built-in defaults to avoid
    // requiring filesystem access inside the inferlet sandbox.
    let cfg = if let Some(config_path) = config_path.as_deref() {
        load_config(config_path)
    } else if single_question.is_some() || debug_prompt.is_some() {
        default_single_question_config()
    } else {
        panic!("--config is required for batch evaluation");
    };
    let metric_spec = resolve_metric(&cfg.data.metric, &cfg.data.dataset);

    let model = inferlet::get_auto_model();
    let stop_sequences = StopSequences::new(&model, cfg.cove.max_wiki_items);
    let prompt_cache = PromptCache::new(&model, cfg.cove.max_wiki_items).await;
    let search_client = Client::new();
    let search_resources =
        SearchResources::new(cfg.cove.search_mode, cfg.cove.search_cache_path.clone())?;

    if let Some(prompt) = debug_prompt {
        let stop_text: Vec<String> = debug_stop_json
            .as_deref()
            .map(|raw| serde_json::from_str(raw).expect("debug_stop_json must be a JSON string array"))
            .unwrap_or_default();
        let max_tokens = debug_max_tokens.unwrap_or(cfg.generation.max_new_tokens);
        let mut ctx = model.create_context();
        ctx.fill(&prompt);
        let trace = generate_from_context_text_stop_trace(
            ctx,
            &model,
            max_tokens,
            effective_phase_temperature(cfg.cove.temperature_verification, &cfg.cove),
            &stop_text,
        )
        .await;
        let result = serde_json::json!({
            "prompt": prompt,
            "stop_text": stop_text,
            "matched_stop_suffix": trace.matched_stop_suffix,
            "raw_response": trace.raw_text,
            "trimmed_response": trace.trimmed_text,
            "raw_token_ids": trace.raw_token_ids,
            "token_timestamps_ms": trace.token_timestamps_ms,
            "total_elapsed_ms": trace.total_elapsed_ms,
        });
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok("completed".to_string());
    }

    // Single question mode
    if let Some(question) = single_question {
        let result = run_cove_single(
            &model,
            &prompt_cache,
            &search_client,
            &search_resources,
            &question,
            None,
            &cfg.cove,
            &stop_sequences,
            show_intermediate,
            single_question_debug,
        )
        .await;

        search_resources.persist()?;
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok("completed".to_string());
    }

    // Batch evaluation mode
    eprintln!("[cove-debug] batch dataset_path={}", cfg.data.dataset_path);
    let dataset = load_dataset(&cfg.data.dataset_path);
    let validation_len = cfg.data.num_samples.min(dataset.validation.len());

    // Determine chain type from dataset
    let forced_chain_type = dataset.chain_type.as_ref().map(|s| ChainType::from_str(s));

    println!(
        "Chain-of-Verification Pipeline\n\
         Dataset: {} ({} validation samples)\n\
         Chain type: {}\n\
         Metric: {}\n\
         Search mode: {}\n",
        cfg.data.dataset,
        validation_len,
        forced_chain_type.map(|c| c.as_str()).unwrap_or("auto"),
        metric_spec.display_name,
        cfg.cove.search_mode.as_str(),
    );

    let mut results = Vec::new();
    let mut baseline_scores: Vec<f64> = Vec::new();
    let mut cove_scores: Vec<f64> = Vec::new();

    let total_start = Instant::now();
    let mut in_flight = FuturesUnordered::new();
    let mut pending_examples = dataset.validation.iter().take(validation_len).enumerate();
    let mut completed = Vec::with_capacity(validation_len);
    let question_concurrency = cfg.cove.question_concurrency.max(1);

    while completed.len() < validation_len {
        while in_flight.len() < question_concurrency {
            let Some((idx, example)) = pending_examples.next() else {
                break;
            };

            let question = example.question.clone();
            let refs = example.answer_texts();
            let model_ref = &model;
            let prompt_cache_ref = &prompt_cache;
            let search_client_ref = &search_client;
            let search_resources_ref = &search_resources;
            let cove_cfg_ref = &cfg.cove;
            let stop_sequences_ref = &stop_sequences;
            let metric_spec = metric_spec;

            in_flight.push(async move {
                let request_start = Instant::now();
                let result = run_cove_single(
                    model_ref,
                    prompt_cache_ref,
                    search_client_ref,
                    search_resources_ref,
                    &question,
                    forced_chain_type,
                    cove_cfg_ref,
                    stop_sequences_ref,
                    show_intermediate,
                    false,
                )
                .await;
                let request_latency_s = request_start.elapsed().as_secs_f64();

                let baseline_score = score_prediction(
                    metric_spec,
                    result["baseline_response"].as_str().unwrap_or(""),
                    &refs,
                );
                let cove_score = score_prediction(
                    metric_spec,
                    result["final_answer"].as_str().unwrap_or(""),
                    &refs,
                );
                let baseline_pass = metric_spec
                    .threshold
                    .map(|threshold| baseline_score >= threshold);
                let cove_pass = metric_spec
                    .threshold
                    .map(|threshold| cove_score >= threshold);

                CompletedEvaluation {
                    idx,
                    refs,
                    result,
                    baseline_score,
                    cove_score,
                    baseline_pass,
                    cove_pass,
                    request_latency_s,
                }
            });
        }

        let Some(entry) = in_flight.next().await else {
            break;
        };

        let completed_count = completed.len() + 1;
        if completed_count % 5 == 0 || completed_count == 1 || completed_count == validation_len {
            println!("Progress: {}/{}", completed_count, validation_len);
        }

        completed.push(entry);
    }

    completed.sort_by_key(|entry| entry.idx);

    let request_latencies = completed
        .iter()
        .map(|entry| entry.request_latency_s)
        .collect::<Vec<_>>();

    for entry in completed {
        baseline_scores.push(entry.baseline_score);
        cove_scores.push(entry.cove_score);

        results.push(serde_json::json!({
            "idx": entry.idx,
            "question": entry.result["question"],
            "references": entry.refs,
            "baseline_response": entry.result["baseline_response"],
            "final_answer": entry.result["final_answer"],
            "baseline_score": entry.baseline_score,
            "cove_score": entry.cove_score,
            "baseline_pass": entry.baseline_pass,
            "cove_pass": entry.cove_pass,
            "request_latency_s": entry.request_latency_s,
            "timings": entry.result["timings"],
        }));
    }

    let total_elapsed = total_start.elapsed();
    search_resources.persist()?;

    // Aggregate timing statistics from all results
    let mut sum_routing_ms: u64 = 0;
    let mut sum_baseline_ms: u64 = 0;
    let mut sum_vq_ms: u64 = 0;
    let mut sum_exec_ms: u64 = 0;
    let mut sum_final_ms: u64 = 0;

    for result in &results {
        if let Some(timings) = result.get("timings") {
            sum_routing_ms += timings["routing_ms"].as_u64().unwrap_or(0);
            sum_baseline_ms += timings["baseline_ms"].as_u64().unwrap_or(0);
            sum_vq_ms += timings["verification_questions_ms"].as_u64().unwrap_or(0);
            sum_exec_ms += timings["execute_verification_ms"].as_u64().unwrap_or(0);
            sum_final_ms += timings["final_answer_ms"].as_u64().unwrap_or(0);
        }
    }

    let n = validation_len as f64;
    let avg_routing_s = (sum_routing_ms as f64) / 1000.0 / n;
    let avg_baseline_s = (sum_baseline_ms as f64) / 1000.0 / n;
    let avg_verification_questions_s = (sum_vq_ms as f64) / 1000.0 / n;
    let avg_execute_verification_s = (sum_exec_ms as f64) / 1000.0 / n;
    let avg_final_answer_s = (sum_final_ms as f64) / 1000.0 / n;

    // Summary statistics
    let baseline_mean = baseline_scores.iter().sum::<f64>() / validation_len as f64;
    let cove_mean = cove_scores.iter().sum::<f64>() / validation_len as f64;
    let baseline_correct = results
        .iter()
        .filter(|result| result["baseline_pass"].as_bool() == Some(true))
        .count();
    let cove_correct = results
        .iter()
        .filter(|result| result["cove_pass"].as_bool() == Some(true))
        .count();
    let improvements = results
        .iter()
        .filter(|result| {
            result["cove_pass"].as_bool() == Some(true)
                && result["baseline_pass"].as_bool() == Some(false)
        })
        .count();
    let regressions = results
        .iter()
        .filter(|result| {
            result["cove_pass"].as_bool() == Some(false)
                && result["baseline_pass"].as_bool() == Some(true)
        })
        .count();

    let summary = serde_json::json!({
        "dataset": cfg.data.dataset,
        "num_samples": validation_len,
        "paper_faithful": cfg.cove.paper_faithful,
        "few_shot_examples": cfg.cove.few_shot_examples,
        "use_search": effective_use_search(&cfg.cove),
        "search_mode": cfg.cove.search_mode.as_str(),
        "search_cache_path": search_resources.cache_path(),
        "accuracy": {
            "metric_name": metric_spec.name,
            "metric_display_name": metric_spec.display_name,
            "score_threshold": metric_spec.threshold,
            "baseline_mean_score": baseline_mean,
            "cove_mean_score": cove_mean,
            "score_improvement": cove_mean - baseline_mean,
            "baseline_correct": metric_spec.threshold.map(|_| baseline_correct),
            "cove_correct": metric_spec.threshold.map(|_| cove_correct),
            "baseline_accuracy": metric_spec
                .threshold
                .map(|_| baseline_correct as f64 / validation_len as f64),
            "cove_accuracy": metric_spec
                .threshold
                .map(|_| cove_correct as f64 / validation_len as f64),
            "improvements": improvements,
            "regressions": regressions,
            "net_improvement": improvements as i64 - regressions as i64,
        },
        "latency": {
            "question_concurrency": question_concurrency,
            "batch_wall_time_s": total_elapsed.as_secs_f64(),
            "avg_batch_wall_time_per_question_s": total_elapsed.as_secs_f64() / validation_len as f64,
            "sum_request_time_s": request_latencies.iter().sum::<f64>(),
            "avg_time_per_question_s": request_latencies.iter().sum::<f64>() / validation_len as f64,
            "mean_request_latency_s": request_latencies.iter().sum::<f64>() / validation_len as f64,
            "p50_request_latency_s": percentile(&request_latencies, 50.0),
            "p95_request_latency_s": percentile(&request_latencies, 95.0),
            "throughput_qps": validation_len as f64 / total_elapsed.as_secs_f64(),
            "avg_routing_s": avg_routing_s,
            "avg_baseline_s": avg_baseline_s,
            "avg_verification_questions_s": avg_verification_questions_s,
            "avg_execute_verification_s": avg_execute_verification_s,
            "avg_final_answer_s": avg_final_answer_s,
        },
    });

    println!("\n{}", "=".repeat(60));
    println!("CHAIN-OF-VERIFICATION RESULTS");
    println!("{}", "=".repeat(60));
    println!("\n{}:", metric_spec.display_name);
    println!("  Baseline: {:.3}", baseline_mean);
    println!("  CoVe:     {:.3}", cove_mean);
    println!("  Improvement: {:+.3}", cove_mean - baseline_mean);
    if let Some(threshold) = metric_spec.threshold {
        println!("\nPass Rate (threshold >= {:.2}):", threshold);
        println!("  Baseline: {}/{}", baseline_correct, validation_len);
        println!("  CoVe:     {}/{}", cove_correct, validation_len);
        println!(
            "  Improvements: +{} | Regressions: -{} | Net: {:+}",
            improvements,
            regressions,
            improvements as i64 - regressions as i64
        );
    }
    println!("\nLATENCY:");
    println!(
        "  Mean request latency: {:.4}s",
        request_latencies.iter().sum::<f64>() / validation_len as f64
    );
    println!(
        "  P50 request latency:  {:.4}s",
        percentile(&request_latencies, 50.0)
    );
    println!(
        "  P95 request latency:  {:.4}s",
        percentile(&request_latencies, 95.0)
    );
    println!(
        "  Batch wall time:      {:.4}s",
        total_elapsed.as_secs_f64()
    );
    println!(
        "  Throughput:           {:.4} qps",
        validation_len as f64 / total_elapsed.as_secs_f64()
    );
    println!("  Concurrency:          {}", question_concurrency);
    println!("\nPHASE BREAKDOWN (avg per question):");
    println!("  Routing:                {:.4}s", avg_routing_s);
    println!("  Baseline:               {:.4}s", avg_baseline_s);
    println!(
        "  Verification Questions: {:.4}s",
        avg_verification_questions_s
    );
    println!(
        "  Execute Verification:   {:.4}s",
        avg_execute_verification_s
    );
    println!("  Final Answer:           {:.4}s", avg_final_answer_s);
    println!("\n{}", serde_json::to_string_pretty(&summary)?);
    let batch_payload = serde_json::json!({
        "summary": summary,
        "results": results,
    });
    let batch_payload_bytes = serde_json::to_vec(&batch_payload)?;
    let batch_payload_b64 = {
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD.encode(batch_payload_bytes)
    };
    let total_chunks = batch_payload_b64.len().div_ceil(BATCH_PAYLOAD_CHUNK_BYTES);
    println!("COVE_BATCH_BASE64_BEGIN={total_chunks}");
    for (idx, chunk) in batch_payload_b64
        .as_bytes()
        .chunks(BATCH_PAYLOAD_CHUNK_BYTES)
        .enumerate()
    {
        println!(
            "COVE_BATCH_BASE64_CHUNK {} {}",
            idx,
            std::str::from_utf8(chunk)?
        );
    }
    println!("COVE_BATCH_BASE64_END");

    Ok("completed".to_string())
}

// ============================================================================
// Core CoVe Pipeline
// ============================================================================

async fn run_cove_single(
    model: &inferlet::Model,
    prompt_cache: &PromptCache,
    search_client: &Client,
    search_resources: &SearchResources,
    question: &str,
    forced_chain_type: Option<ChainType>,
    cove_cfg: &CoVeConfig,
    stop_sequences: &StopSequences,
    verbose: bool,
    debug_trace: bool,
) -> serde_json::Value {
    let total_start = Instant::now();
    let mut debug_two_step_execute: Option<serde_json::Value> = None;

    // Phase 0: Route
    let routing_start = Instant::now();
    let chain_type = if let Some(ct) = forced_chain_type {
        ct
    } else {
        route_question(
            prompt_cache,
            question,
            cove_cfg.max_tokens_routing,
            effective_phase_temperature(cove_cfg.temperature_routing, cove_cfg),
            stop_sequences.default_tokens(),
            stop_sequences.default_text(),
        )
        .await
    };
    let routing_ms = routing_start.elapsed().as_millis();

    if verbose {
        println!("[CoVe] Chain: {:?} ({}ms)", chain_type, routing_ms);
    }

    let paper_variant = resolve_paper_variant(cove_cfg, None, chain_type.as_str());

    // Phase 1: Baseline
    let baseline_start = Instant::now();
    let baseline = generate_baseline(
        model,
        prompt_cache,
        question,
        chain_type,
        effective_baseline_tokens(cove_cfg, chain_type),
        effective_phase_temperature(cove_cfg.temperature_baseline, cove_cfg),
        cove_cfg,
        stop_sequences,
    )
    .await;
    let baseline_ms = baseline_start.elapsed().as_millis();

    if verbose {
        println!(
            "[CoVe] Baseline ({}ms): {}...",
            baseline_ms,
            &baseline[..baseline.len().min(80)]
        );
    }

    let mut cross_checks: Vec<(String, String)> = Vec::new();
    let (verif_questions, verif_answers, vq_ms, exec_ms, cross_check_ms, final_answer, final_ms) =
        if cove_cfg.paper_faithful && paper_variant == CoVeVariant::Joint {
            let vq_start = Instant::now();
            let verif_answers =
                generate_joint_verification_pairs(model, question, &baseline, chain_type, cove_cfg)
                    .await;
            let vq_ms = vq_start.elapsed().as_millis();
            let verif_questions = verif_answers
                .iter()
                .map(|(question, _)| question.clone())
                .collect::<Vec<_>>();
            let final_start = Instant::now();
            let final_answer = generate_final_answer(
                model,
                prompt_cache,
                question,
                &baseline,
                &verif_answers,
                &cross_checks,
                chain_type,
                effective_final_tokens(cove_cfg, chain_type),
                effective_phase_temperature(cove_cfg.temperature_final, cove_cfg),
                cove_cfg,
                paper_variant,
                stop_sequences,
            )
            .await;
            (
                verif_questions,
                verif_answers,
                vq_ms,
                0,
                0,
                final_answer,
                final_start.elapsed().as_millis(),
            )
        } else if cove_cfg.paper_faithful
            && paper_variant == CoVeVariant::FactorRevise
            && chain_type == ChainType::Long
        {
            let result = run_factor_revise(
                model,
                prompt_cache,
                search_client,
                search_resources,
                question,
                &baseline,
                chain_type,
                cove_cfg,
                stop_sequences,
            )
            .await;
            cross_checks = result.cross_checks;
            (
                result.verification_questions,
                result.verification_answers,
                result.verification_questions_ms,
                result.execute_verification_ms,
                result.cross_check_ms,
                result.final_answer,
                result.final_answer_ms,
            )
        } else {
            let vq_start = Instant::now();
            let verif_questions = generate_verification_questions(
                model,
                prompt_cache,
                question,
                &baseline,
                chain_type,
                effective_max_verification_questions(cove_cfg, chain_type),
                effective_verification_tokens(cove_cfg, chain_type),
                effective_phase_temperature(cove_cfg.temperature_verification, cove_cfg),
                cove_cfg,
                stop_sequences,
            )
            .await;
            let vq_ms = vq_start.elapsed().as_millis();
            let exec_start = Instant::now();
            let verif_answers = if debug_trace
                && cove_cfg.paper_faithful
                && paper_variant == CoVeVariant::TwoStep
            {
                let prompt = build_paper_two_step_execute_prompt(
                    &verif_questions,
                    chain_type.as_str(),
                    cove_cfg.few_shot_examples,
                );
                let mut ctx = model.create_context();
                ctx.fill(&prompt);
                let stop_text = paper_stop_sequences(
                    "two_step_execute",
                    chain_type.as_str(),
                    cove_cfg.max_wiki_items,
                );
                let trace = generate_from_context_text_stop_trace(
                        ctx,
                        model,
                        paper_batch_tokens(cove_cfg, chain_type),
                        effective_phase_temperature(cove_cfg.temperature_verification, cove_cfg),
                        &stop_text,
                    )
                    .await;
                let answers = parse_numbered_answers(&trace.trimmed_text, verif_questions.len());
                debug_two_step_execute = Some(serde_json::json!({
                    "prompt": prompt,
                    "stop_text": stop_text,
                    "matched_stop_suffix": trace.matched_stop_suffix,
                    "raw_response": trace.raw_text,
                    "trimmed_response": trace.trimmed_text,
                    "raw_token_ids": trace.raw_token_ids,
                    "token_timestamps_ms": trace.token_timestamps_ms,
                    "total_elapsed_ms": trace.total_elapsed_ms,
                }));
                verif_questions
                    .iter()
                    .cloned()
                    .zip(answers.into_iter())
                    .collect::<Vec<_>>()
            } else {
                execute_verification_parallel(
                    model,
                    prompt_cache,
                    search_client,
                    search_resources,
                    &verif_questions,
                    chain_type,
                    cove_cfg,
                    paper_variant,
                    stop_sequences,
                )
                .await
            };
            let exec_ms = exec_start.elapsed().as_millis();
            let final_start = Instant::now();
            let final_answer = generate_final_answer(
                model,
                prompt_cache,
                question,
                &baseline,
                &verif_answers,
                &cross_checks,
                chain_type,
                effective_final_tokens(cove_cfg, chain_type),
                effective_phase_temperature(cove_cfg.temperature_final, cove_cfg),
                cove_cfg,
                paper_variant,
                stop_sequences,
            )
            .await;
            (
                verif_questions,
                verif_answers,
                vq_ms,
                exec_ms,
                0,
                final_answer,
                final_start.elapsed().as_millis(),
            )
        };

    if verbose {
        println!("[CoVe] Variant: {}", paper_variant.as_str());
        println!(
            "[CoVe] {} verification Qs ({}ms)",
            verif_questions.len(),
            vq_ms
        );
        println!(
            "[CoVe] Executed {} verifications ({}ms)",
            verif_answers.len(),
            exec_ms
        );
        if cross_check_ms > 0 {
            println!("[CoVe] Cross-check ({}ms)", cross_check_ms);
        }
    }

    let total_ms = total_start.elapsed().as_millis();

    let mut result = serde_json::json!({
        "question": question,
        "chain_type": chain_type.as_str(),
        "cove_variant": paper_variant.as_str(),
        "baseline_response": baseline,
        "verification_questions": verif_questions,
        "verification_answers": verif_answers.iter()
            .map(|(q, a)| serde_json::json!({"q": q, "a": a}))
            .collect::<Vec<_>>(),
        "cross_check_results": cross_checks.iter()
            .map(|(fact, response)| serde_json::json!({"fact": fact, "response": response}))
            .collect::<Vec<_>>(),
        "final_answer": final_answer,
        "timings": {
            "routing_ms": routing_ms,
            "baseline_ms": baseline_ms,
            "verification_questions_ms": vq_ms,
            "execute_verification_ms": exec_ms,
            "cross_check_ms": cross_check_ms,
            "final_answer_ms": final_ms,
            "total_ms": total_ms,
        }
    });
    if let Some(debug) = debug_two_step_execute {
        result["debug_two_step_execute"] = debug;
    }
    result
}

// ============================================================================
// Helper Functions
// ============================================================================

async fn route_question(
    prompt_cache: &PromptCache,
    question: &str,
    max_tokens: usize,
    temperature: f32,
    stop_tokens: &[Vec<u32>],
    stop_text: &[String],
) -> ChainType {
    let ctx = prompt_cache.router.fork_with_suffix(question);
    let response =
        generate_from_context(ctx, max_tokens, temperature, stop_tokens, stop_text).await;
    ChainType::from_str(&response)
}

async fn generate_baseline(
    model: &inferlet::Model,
    prompt_cache: &PromptCache,
    question: &str,
    chain_type: ChainType,
    max_tokens: usize,
    temperature: f32,
    cove_cfg: &CoVeConfig,
    stop_sequences: &StopSequences,
) -> String {
    if cove_cfg.paper_faithful {
        let prompt =
            build_paper_baseline_prompt(question, chain_type.as_str(), cove_cfg.few_shot_examples);
        let mut ctx = model.create_context();
        ctx.fill(&prompt);
        let stop_text =
            paper_stop_sequences("baseline", chain_type.as_str(), cove_cfg.max_wiki_items);
        let stop_tokens = stop_tokens_for_text(model, &stop_text);
        return generate_from_context(ctx, max_tokens, temperature, &stop_tokens, &stop_text).await;
    }

    let _ = model;
    let ctx = prompt_cache
        .baseline_prompt(chain_type)
        .fork_with_suffix(question);
    let (stop_tokens, stop_text) = wiki_list_stop_set(chain_type, stop_sequences);
    generate_from_context(ctx, max_tokens, temperature, stop_tokens, stop_text).await
}

async fn generate_verification_questions(
    model: &inferlet::Model,
    prompt_cache: &PromptCache,
    question: &str,
    baseline: &str,
    chain_type: ChainType,
    max_questions: usize,
    max_tokens: usize,
    temperature: f32,
    cove_cfg: &CoVeConfig,
    stop_sequences: &StopSequences,
) -> Vec<String> {
    if cove_cfg.paper_faithful {
        let prompt = build_paper_verification_prompt(
            question,
            baseline,
            chain_type.as_str(),
            cove_cfg.few_shot_examples,
        );
        let mut ctx = model.create_context();
        ctx.fill(&prompt);
        let stop_text =
            paper_stop_sequences("verification", chain_type.as_str(), cove_cfg.max_wiki_items);
        let stop_tokens = stop_tokens_for_text(model, &stop_text);
        let response =
            generate_from_context(ctx, max_tokens, temperature, &stop_tokens, &stop_text).await;
        return parse_verification_questions(&response, max_questions);
    }

    let ctx = match chain_type {
        ChainType::Wiki => {
            let template_ctx = prompt_cache
                .verification_template_wiki
                .fork_with_suffix(question);
            let verification_question_template = generate_from_context(
                template_ctx,
                max_tokens,
                0.2,
                stop_sequences.default_tokens(),
                stop_sequences.default_text(),
            )
            .await;

            let tail = format!(
                "{}{}",
                question,
                prompt_cache
                    .verification_wiki
                    .suffix()
                    .replace("{baseline_response}", baseline)
                    .replace(
                        "{verification_question_template}",
                        &verification_question_template
                    )
            );
            prompt_cache.verification_wiki.fork_with_tail(&tail)
        }
        ChainType::Multi => {
            let tail = format!(
                "{}{}",
                question,
                prompt_cache
                    .verification_multi
                    .suffix()
                    .replace("{baseline_response}", baseline)
            );
            prompt_cache.verification_multi.fork_with_tail(&tail)
        }
        ChainType::Long => {
            let tail = format!(
                "{}{}",
                question,
                prompt_cache
                    .verification_long
                    .suffix()
                    .replace("{baseline_response}", baseline)
            );
            prompt_cache.verification_long.fork_with_tail(&tail)
        }
    };
    let _ = model;
    let (stop_tokens, stop_text) = wiki_list_stop_set(chain_type, stop_sequences);
    let response =
        generate_from_context(ctx, max_tokens, temperature, stop_tokens, stop_text).await;
    parse_verification_questions(&response, max_questions)
}

async fn generate_joint_verification_pairs(
    model: &inferlet::Model,
    question: &str,
    baseline: &str,
    chain_type: ChainType,
    cove_cfg: &CoVeConfig,
) -> Vec<(String, String)> {
    let prompt = build_paper_joint_prompt(
        question,
        baseline,
        chain_type.as_str(),
        cove_cfg.few_shot_examples,
    );
    let mut ctx = model.create_context();
    ctx.fill(&prompt);
    let stop_text = paper_stop_sequences("joint", chain_type.as_str(), cove_cfg.max_wiki_items);
    let stop_tokens = stop_tokens_for_text(model, &stop_text);
    let response = generate_from_context(
        ctx,
        paper_batch_tokens(cove_cfg, chain_type),
        effective_phase_temperature(cove_cfg.temperature_verification, cove_cfg),
        &stop_tokens,
        &stop_text,
    )
    .await;
    parse_verification_pairs(
        &response,
        effective_max_verification_questions(cove_cfg, chain_type),
    )
}

async fn execute_verification_parallel(
    model: &inferlet::Model,
    prompt_cache: &PromptCache,
    search_client: &Client,
    search_resources: &SearchResources,
    questions: &[String],
    chain_type: ChainType,
    cove_cfg: &CoVeConfig,
    paper_variant: CoVeVariant,
    stop_sequences: &StopSequences,
) -> Vec<(String, String)> {
    if questions.is_empty() {
        return Vec::new();
    }

    let use_search = effective_use_search(cove_cfg);
    let max_tokens = effective_verification_tokens(cove_cfg, chain_type);
    let temperature = effective_phase_temperature(cove_cfg.temperature_verification, cove_cfg);

    if cove_cfg.paper_faithful && paper_variant == CoVeVariant::TwoStep {
        let prompt = build_paper_two_step_execute_prompt(
            questions,
            chain_type.as_str(),
            cove_cfg.few_shot_examples,
        );
        let mut ctx = model.create_context();
        ctx.fill(&prompt);
        let stop_text = paper_stop_sequences(
            "two_step_execute",
            chain_type.as_str(),
            cove_cfg.max_wiki_items,
        );
        let (_, response, _) = generate_from_context_text_stop(
            ctx,
            model,
            paper_batch_tokens(cove_cfg, chain_type),
            temperature,
            &stop_text,
        )
        .await;
        let answers = parse_numbered_answers(&response, questions.len());
        return questions
            .iter()
            .cloned()
            .zip(answers.into_iter())
            .collect::<Vec<_>>();
    }

    // Run all verifications independently so the effective prompt text matches
    // the standalone vLLM implementation exactly.
    let futures: Vec<_> = questions
        .iter()
        .map(|q| {
            let question = q.clone();
            let model = model.clone();
            let prompt_cache = prompt_cache;
            let search_client = search_client;
            let search_resources = search_resources;
            let paper_faithful = cove_cfg.paper_faithful;
            let few_shot_examples = cove_cfg.few_shot_examples;
            let max_wiki_items = cove_cfg.max_wiki_items;
            let max_search_results = cove_cfg.max_search_results;
            let search_timeout = cove_cfg.search_timeout;
            let chain_type_name = chain_type.as_str().to_string();
            let paper_stop_text = if paper_faithful {
                paper_stop_sequences("execute", &chain_type_name, max_wiki_items)
            } else {
                Vec::new()
            };
            let paper_stop_tokens = if paper_faithful {
                stop_tokens_for_text(&model, &paper_stop_text)
            } else {
                Vec::new()
            };
            let default_stop_tokens = stop_sequences.default_tokens().to_vec();
            let default_stop_text = stop_sequences.default_text().to_vec();

            async move {
                if paper_faithful {
                    let prompt =
                        build_paper_execute_prompt(&question, &chain_type_name, few_shot_examples);
                    let mut ctx = model.create_context();
                    ctx.fill(&prompt);
                    let answer = generate_from_context(
                        ctx,
                        max_tokens,
                        temperature,
                        &paper_stop_tokens,
                        &paper_stop_text,
                    )
                    .await;
                    return (question, answer);
                }

                // Get search context if enabled
                let search_context = if use_search {
                    match search_resources
                        .search_ddg(search_client, &question, max_search_results, search_timeout)
                        .await
                    {
                        Ok(result) => result,
                        Err(error) => {
                            if search_resources.mode() == SearchMode::Replay {
                                panic!("Frozen search failed for '{}': {}", question, error);
                            }
                            String::new()
                        }
                    }
                } else {
                    String::new()
                };

                let ctx = if use_search && !search_context.is_empty() {
                    let tail = format!(
                        "{}{}",
                        search_context,
                        prompt_cache
                            .execute_search
                            .suffix()
                            .replace("{verification_question}", &question)
                    );
                    prompt_cache.execute_search.fork_with_tail(&tail)
                } else {
                    prompt_cache.execute_self.fork_with_suffix(&question)
                };

                let answer = generate_from_context(
                    ctx,
                    max_tokens,
                    temperature,
                    &default_stop_tokens,
                    &default_stop_text,
                )
                .await;
                (question, answer)
            }
        })
        .collect();

    join_all(futures).await
}

async fn generate_final_answer(
    model: &inferlet::Model,
    prompt_cache: &PromptCache,
    question: &str,
    baseline: &str,
    verif_answers: &[(String, String)],
    cross_check_results: &[(String, String)],
    chain_type: ChainType,
    max_tokens: usize,
    temperature: f32,
    cove_cfg: &CoVeConfig,
    paper_variant: CoVeVariant,
    stop_sequences: &StopSequences,
) -> String {
    if cove_cfg.paper_faithful {
        let _ = paper_variant;
        let _ = cross_check_results;
        let prompt = build_paper_final_prompt(
            question,
            baseline,
            verif_answers,
            chain_type.as_str(),
            cove_cfg.few_shot_examples,
        );
        let mut ctx = model.create_context();
        ctx.fill(&prompt);
        let stop_text = paper_stop_sequences("final", chain_type.as_str(), cove_cfg.max_wiki_items);
        let stop_tokens = stop_tokens_for_text(model, &stop_text);
        return generate_from_context(ctx, max_tokens, temperature, &stop_tokens, &stop_text).await;
    }

    let verif_text: String = verif_answers
        .iter()
        .map(|(q, a)| format!("Question: {}\nAnswer: {}", q, a))
        .collect::<Vec<_>>()
        .join("\n");

    let tail = format!(
        "{}{}",
        question,
        prompt_cache
            .final_prompt(chain_type)
            .suffix()
            .replace("{baseline_response}", baseline)
            .replace("{verification_answers}", &verif_text)
    );
    let _ = model;
    let ctx = prompt_cache.final_prompt(chain_type).fork_with_tail(&tail);
    let (stop_tokens, stop_text) = wiki_list_stop_set(chain_type, stop_sequences);
    generate_from_context(ctx, max_tokens, temperature, stop_tokens, stop_text).await
}

async fn cross_check_passage(
    model: &inferlet::Model,
    original_fact: &str,
    verification_answers: &[(String, String)],
    cove_cfg: &CoVeConfig,
) -> String {
    if verification_answers.is_empty() {
        return "INCONSISTENT.".to_string();
    }

    let prompt = build_paper_crosscheck_prompt(
        original_fact,
        verification_answers,
        cove_cfg.few_shot_examples,
    );
    let mut ctx = model.create_context();
    ctx.fill(&prompt);
    let stop_text = paper_stop_sequences("cross_check", "LONG_CHAIN", cove_cfg.max_wiki_items);
    let stop_tokens = stop_tokens_for_text(model, &stop_text);
    let response = generate_from_context(
        ctx,
        effective_final_tokens(cove_cfg, ChainType::Long),
        effective_phase_temperature(cove_cfg.temperature_final, cove_cfg),
        &stop_tokens,
        &stop_text,
    )
    .await;
    normalize_crosscheck_response(&response)
}

async fn revise_passage(cross_check_response: &str) -> String {
    let Some(consistent_body) = extract_consistent_crosscheck_body(cross_check_response) else {
        return String::new();
    };
    splice_consistent_body(&consistent_body)
}

async fn run_factor_revise_passage(
    idx: usize,
    model: &inferlet::Model,
    prompt_cache: &PromptCache,
    search_client: &Client,
    search_resources: &SearchResources,
    question: &str,
    passage: String,
    chain_type: ChainType,
    cove_cfg: &CoVeConfig,
    stop_sequences: &StopSequences,
) -> PassageFactorReviseResult {
    let vq_start = Instant::now();
    let questions = generate_verification_questions(
        model,
        prompt_cache,
        question,
        &passage,
        chain_type,
        effective_max_verification_questions(cove_cfg, chain_type),
        effective_verification_tokens(cove_cfg, chain_type),
        effective_phase_temperature(cove_cfg.temperature_verification, cove_cfg),
        cove_cfg,
        stop_sequences,
    )
    .await;
    let verification_questions_ms = vq_start.elapsed().as_millis();

    let exec_start = Instant::now();
    let answers = execute_verification_parallel(
        model,
        prompt_cache,
        search_client,
        search_resources,
        &questions,
        chain_type,
        cove_cfg,
        CoVeVariant::Factored,
        stop_sequences,
    )
    .await;
    let execute_verification_ms = exec_start.elapsed().as_millis();

    let cross_start = Instant::now();
    let cross_check = cross_check_passage(model, &passage, &answers, cove_cfg).await;
    let cross_check_ms = cross_start.elapsed().as_millis();

    let revise_start = Instant::now();
    let revised_passage = revise_passage(&cross_check).await;
    let final_answer_ms = revise_start.elapsed().as_millis();

    PassageFactorReviseResult {
        idx,
        passage,
        verification_questions: questions,
        verification_answers: answers,
        cross_check,
        revised_passage,
        verification_questions_ms,
        execute_verification_ms,
        cross_check_ms,
        final_answer_ms,
    }
}

async fn run_factor_revise(
    model: &inferlet::Model,
    prompt_cache: &PromptCache,
    search_client: &Client,
    search_resources: &SearchResources,
    question: &str,
    baseline: &str,
    chain_type: ChainType,
    cove_cfg: &CoVeConfig,
    stop_sequences: &StopSequences,
) -> FactorReviseResult {
    let passages = split_passages(baseline);
    let total_passages = passages.len();
    let mut verification_questions = Vec::new();
    let mut verification_answers = Vec::new();
    let mut cross_checks = Vec::new();
    let mut revised_passages = Vec::new();
    let mut verification_questions_ms = 0u128;
    let mut execute_verification_ms = 0u128;
    let mut cross_check_ms = 0u128;
    let mut final_answer_ms = 0u128;

    let max_in_flight_passages =
        resolve_factor_revise_passage_concurrency(cove_cfg, total_passages, false);
    let mut pending_passages = passages.into_iter().enumerate();
    let mut in_flight = FuturesUnordered::new();
    let mut completed = Vec::with_capacity(total_passages);

    while completed.len() < total_passages {
        while in_flight.len() < max_in_flight_passages {
            let Some((idx, passage)) = pending_passages.next() else {
                break;
            };
            in_flight.push(run_factor_revise_passage(
                idx,
                model,
                prompt_cache,
                search_client,
                search_resources,
                question,
                passage,
                chain_type,
                cove_cfg,
                stop_sequences,
            ));
        }

        let Some(result) = in_flight.next().await else {
            break;
        };
        completed.push(result);
    }

    completed.sort_by_key(|result| result.idx);

    for result in completed {
        verification_questions_ms += result.verification_questions_ms;
        execute_verification_ms += result.execute_verification_ms;
        cross_check_ms += result.cross_check_ms;
        final_answer_ms += result.final_answer_ms;
        verification_questions.extend(result.verification_questions);
        verification_answers.extend(result.verification_answers);
        cross_checks.push((result.passage, result.cross_check));
        if !result.revised_passage.is_empty() {
            revised_passages.push(result.revised_passage);
        }
    }

    let final_answer = if revised_passages.is_empty() {
        baseline.trim().to_string()
    } else {
        splice_consistent_body(&revised_passages.join(" "))
    };

    FactorReviseResult {
        verification_questions,
        verification_answers,
        cross_checks,
        final_answer,
        verification_questions_ms,
        execute_verification_ms,
        cross_check_ms,
        final_answer_ms,
    }
}

fn effective_baseline_tokens(cove_cfg: &CoVeConfig, chain_type: ChainType) -> usize {
    if cove_cfg.paper_faithful {
        return cove_cfg.max_tokens_baseline;
    }
    if chain_type == ChainType::Wiki {
        cove_cfg
            .max_tokens_baseline
            .min(cove_cfg.max_tokens_wiki_baseline)
    } else {
        cove_cfg.max_tokens_baseline
    }
}

fn effective_verification_tokens(cove_cfg: &CoVeConfig, chain_type: ChainType) -> usize {
    if cove_cfg.paper_faithful {
        return cove_cfg.max_tokens_verification;
    }
    if chain_type == ChainType::Wiki {
        cove_cfg
            .max_tokens_verification
            .min(cove_cfg.max_tokens_wiki_verification)
    } else {
        cove_cfg.max_tokens_verification
    }
}

fn effective_final_tokens(cove_cfg: &CoVeConfig, chain_type: ChainType) -> usize {
    if cove_cfg.paper_faithful {
        return cove_cfg.max_tokens_final;
    }
    if chain_type == ChainType::Wiki {
        cove_cfg
            .max_tokens_final
            .min(cove_cfg.max_tokens_wiki_final)
    } else {
        cove_cfg.max_tokens_final
    }
}

fn paper_batch_tokens(cove_cfg: &CoVeConfig, chain_type: ChainType) -> usize {
    effective_final_tokens(cove_cfg, chain_type).max(
        effective_verification_tokens(cove_cfg, chain_type)
            * effective_max_verification_questions(cove_cfg, chain_type).max(1),
    )
}

fn effective_max_verification_questions(cove_cfg: &CoVeConfig, chain_type: ChainType) -> usize {
    if cove_cfg.paper_faithful {
        return cove_cfg.max_verification_questions;
    }
    if chain_type == ChainType::Wiki {
        cove_cfg
            .max_verification_questions
            .min(cove_cfg.max_wiki_items.max(1))
    } else {
        cove_cfg.max_verification_questions
    }
}

fn wiki_list_stop_set<'a>(
    chain_type: ChainType,
    stop_sequences: &'a StopSequences,
) -> (&'a [Vec<u32>], &'a [String]) {
    if chain_type == ChainType::Wiki {
        (
            stop_sequences.wiki_list_tokens(),
            stop_sequences.wiki_list_text(),
        )
    } else {
        (
            stop_sequences.default_tokens(),
            stop_sequences.default_text(),
        )
    }
}

fn trim_stop_suffixes(mut text: String, stop_texts: &[String]) -> String {
    loop {
        let trimmed = text.trim_end().to_string();
        if trimmed.len() != text.len() {
            text = trimmed;
        }
        let mut removed = false;
        for stop in stop_texts {
            let suffix = stop.trim_end();
            if !suffix.is_empty() && text.ends_with(suffix) {
                text.truncate(text.len().saturating_sub(suffix.len()));
                text = text.trim_end().to_string();
                removed = true;
            }
        }
        if !removed {
            break;
        }
    }
    text.trim().to_string()
}

fn matched_stop_suffix(text: &str, stop_texts: &[String]) -> Option<String> {
    let trimmed = text.trim_end();
    for stop in stop_texts {
        let suffix = stop.trim_end();
        if !suffix.is_empty() && trimmed.ends_with(suffix) {
            return Some(suffix.to_string());
        }
    }
    None
}

async fn generate_from_context(
    mut ctx: inferlet::Context,
    max_tokens: usize,
    temperature: f32,
    stop_tokens: &[Vec<u32>],
    stop_text: &[String],
) -> String {
    let stop = max_len(max_tokens.max(1)).or(ends_with_any(stop_tokens.to_vec()));
    let text = ctx.generate(make_sampler(temperature), stop).await;
    trim_stop_suffixes(text, stop_text)
}

async fn generate_from_context_text_stop(
    mut ctx: inferlet::Context,
    model: &inferlet::Model,
    max_tokens: usize,
    temperature: f32,
    stop_text: &[String],
) -> (String, String, Option<String>) {
    let tokenizer = model.get_tokenizer();
    let sampler = make_sampler(temperature);
    let mut generated_token_ids = Vec::new();

    for _ in 0..max_tokens.max(1) {
        let next_token_id = ctx.decode_step(&sampler).await;
        ctx.fill_token(next_token_id);
        generated_token_ids.push(next_token_id);

        let raw_text = tokenizer.detokenize(&generated_token_ids);
        if let Some(matched) = matched_stop_suffix(&raw_text, stop_text) {
            let trimmed = trim_stop_suffixes(raw_text.clone(), stop_text);
            return (raw_text, trimmed, Some(matched));
        }
    }

    let raw_text = tokenizer.detokenize(&generated_token_ids);
    let trimmed = trim_stop_suffixes(raw_text.clone(), stop_text);
    (raw_text, trimmed, None)
}

async fn generate_from_context_text_stop_trace(
    mut ctx: inferlet::Context,
    model: &inferlet::Model,
    max_tokens: usize,
    temperature: f32,
    stop_text: &[String],
) -> TextStopTrace {
    let tokenizer = model.get_tokenizer();
    let sampler = make_sampler(temperature);
    let mut generated_token_ids = Vec::new();
    let mut token_timestamps_ms = Vec::new();
    let start = Instant::now();

    for _ in 0..max_tokens.max(1) {
        let next_token_id = ctx.decode_step(&sampler).await;
        ctx.fill_token(next_token_id);
        generated_token_ids.push(next_token_id);
        token_timestamps_ms.push(start.elapsed().as_secs_f64() * 1000.0);

        let raw_text = tokenizer.detokenize(&generated_token_ids);
        if let Some(matched) = matched_stop_suffix(&raw_text, stop_text) {
            let trimmed = trim_stop_suffixes(raw_text.clone(), stop_text);
            return TextStopTrace {
                raw_text,
                trimmed_text: trimmed,
                matched_stop_suffix: Some(matched),
                raw_token_ids: generated_token_ids,
                token_timestamps_ms,
                total_elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            };
        }
    }

    let raw_text = tokenizer.detokenize(&generated_token_ids);
    let trimmed = trim_stop_suffixes(raw_text.clone(), stop_text);
    TextStopTrace {
        raw_text,
        trimmed_text: trimmed,
        matched_stop_suffix: None,
        raw_token_ids: generated_token_ids,
        token_timestamps_ms,
        total_elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
    }
}

fn make_sampler(temperature: f32) -> Sampler {
    if temperature <= 0.0 {
        Sampler::greedy()
    } else {
        Sampler::top_p(temperature, 1.0)
    }
}

fn effective_phase_temperature(temperature: f32, cove_cfg: &CoVeConfig) -> f32 {
    if cove_cfg.paper_faithful {
        0.0
    } else {
        temperature
    }
}
