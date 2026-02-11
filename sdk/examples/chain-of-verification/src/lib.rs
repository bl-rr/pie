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
use inferlet::stop_condition::{ends_with_any, max_len, StopCondition};
use inferlet::{Args, Context, Result, Sampler};
use std::time::Instant;

use crate::util::*;

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

const BASELINE_PROMPT_WIKI: &str = r#"Answer the below question which is asking for a list of entities (names, places, locations etc). Output should be a numbered list and only contains the relevant & concise entities as answer. NO ADDITIONAL DETAILS.

Question: {question}

Answer:"#;

const BASELINE_PROMPT_MULTI: &str = r#"Answer the below question correctly and in a concise manner without much details. Only answer what the question is asked.

Question: {question}

Answer:"#;

const BASELINE_PROMPT_LONG: &str = r#"Answer the below question correctly.

Question: {question}

Answer:"#;

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

// ============================================================================
// Main Entry Point
// ============================================================================

#[inferlet::main]
async fn main(mut args: Args) -> Result<String> {
    let config_path: String = args.value_from_str(["-c", "--config"])?;
    let single_question: Option<String> = args.opt_value_from_str(["-q", "--question"])?;
    let show_intermediate: bool = args.contains(["-v", "--verbose"]);

    // Load configuration (panics if any required field is missing)
    let cfg = load_config(&config_path);

    let model = inferlet::get_auto_model();
    let eos_tokens = model.eos_tokens();

    // Single question mode
    if let Some(question) = single_question {
        let result = run_cove_single(
            &model,
            &question,
            None,
            &cfg.cove,
            &eos_tokens,
            show_intermediate,
        )
        .await;

        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok("completed".to_string());
    }

    // Batch evaluation mode
    let dataset = load_dataset(&cfg.data.dataset_path);
    let validation_len = cfg.data.num_samples.min(dataset.validation.len());

    // Determine chain type from dataset
    let forced_chain_type = dataset.chain_type.as_ref().map(|s| ChainType::from_str(s));

    println!(
        "Chain-of-Verification Pipeline\n\
         Dataset: {} ({} validation samples)\n\
         Chain type: {}\n",
        cfg.data.dataset,
        validation_len,
        forced_chain_type.map(|c| c.as_str()).unwrap_or("auto"),
    );

    let mut results = Vec::new();
    let mut baseline_scores: Vec<f64> = Vec::new();
    let mut cove_scores: Vec<f64> = Vec::new();

    let total_start = Instant::now();

    for (idx, example) in dataset.validation.iter().take(validation_len).enumerate() {
        if (idx + 1) % 5 == 0 || idx == 0 {
            println!("Progress: {}/{}", idx + 1, validation_len);
        }

        let result = run_cove_single(
            &model,
            &example.question,
            forced_chain_type,
            &cfg.cove,
            &eos_tokens,
            show_intermediate,
        )
        .await;

        let refs = example.answer_texts();
        let baseline_score =
            calculate_factscore(result["baseline_response"].as_str().unwrap_or(""), &refs);
        let cove_score = calculate_factscore(result["final_answer"].as_str().unwrap_or(""), &refs);

        baseline_scores.push(baseline_score);
        cove_scores.push(cove_score);

        results.push(serde_json::json!({
            "idx": idx,
            "question": example.question,
            "references": refs,
            "baseline_response": result["baseline_response"],
            "final_answer": result["final_answer"],
            "baseline_score": baseline_score,
            "cove_score": cove_score,
            "timings": result["timings"],
        }));
    }

    let total_elapsed = total_start.elapsed();

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
    let baseline_correct = baseline_scores.iter().filter(|&&s| s >= 0.5).count();
    let cove_correct = cove_scores.iter().filter(|&&s| s >= 0.5).count();

    let summary = serde_json::json!({
        "dataset": cfg.data.dataset,
        "num_samples": validation_len,
        "accuracy": {
            "baseline_mean_factscore": baseline_mean,
            "cove_mean_factscore": cove_mean,
            "factscore_improvement": cove_mean - baseline_mean,
            "baseline_correct": baseline_correct,
            "cove_correct": cove_correct,
        },
        "latency": {
            "total_time_s": total_elapsed.as_secs_f64(),
            "avg_time_per_question_s": total_elapsed.as_secs_f64() / validation_len as f64,
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
    println!("\nFACTSCORE:");
    println!("  Baseline: {:.3}", baseline_mean);
    println!("  CoVe:     {:.3}", cove_mean);
    println!("  Improvement: {:+.3}", cove_mean - baseline_mean);
    println!("\nAccuracy (>= 0.5):");
    println!("  Baseline: {}/{}", baseline_correct, validation_len);
    println!("  CoVe:     {}/{}", cove_correct, validation_len);
    println!("\nLATENCY BREAKDOWN (avg per question):");
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

    Ok("completed".to_string())
}

// ============================================================================
// Core CoVe Pipeline
// ============================================================================

async fn run_cove_single(
    model: &inferlet::Model,
    question: &str,
    forced_chain_type: Option<ChainType>,
    cove_cfg: &CoVeConfig,
    eos_tokens: &[Vec<u32>],
    verbose: bool,
) -> serde_json::Value {
    let total_start = Instant::now();

    // Phase 0: Route
    let routing_start = Instant::now();
    let chain_type = if let Some(ct) = forced_chain_type {
        ct
    } else {
        route_question(model, question, cove_cfg.temperature_routing, eos_tokens).await
    };
    let routing_ms = routing_start.elapsed().as_millis();

    if verbose {
        println!("[CoVe] Chain: {:?} ({}ms)", chain_type, routing_ms);
    }

    // Phase 1: Baseline
    let baseline_start = Instant::now();
    let baseline = generate_baseline(
        model,
        question,
        chain_type,
        cove_cfg.max_tokens_baseline,
        cove_cfg.temperature_baseline,
        eos_tokens,
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

    // Phase 2: Verification Questions
    let vq_start = Instant::now();
    let verif_questions = generate_verification_questions(
        model,
        question,
        &baseline,
        chain_type,
        cove_cfg.max_verification_questions,
        cove_cfg.max_tokens_verification * cove_cfg.max_verification_questions,
        cove_cfg.temperature_verification,
        eos_tokens,
    )
    .await;
    let vq_ms = vq_start.elapsed().as_millis();

    if verbose {
        println!(
            "[CoVe] {} verification Qs ({}ms)",
            verif_questions.len(),
            vq_ms
        );
    }

    // Phase 3: Execute Verification (PARALLEL)
    let exec_start = Instant::now();
    let verif_answers = execute_verification_parallel(
        model,
        &verif_questions,
        cove_cfg.use_search,
        cove_cfg.max_search_results,
        cove_cfg.max_tokens_verification,
        cove_cfg.temperature_verification,
        eos_tokens,
    )
    .await;
    let exec_ms = exec_start.elapsed().as_millis();

    if verbose {
        println!(
            "[CoVe] Executed {} verifications ({}ms)",
            verif_answers.len(),
            exec_ms
        );
    }

    // Phase 4: Final Answer
    let final_start = Instant::now();
    let final_answer = generate_final_answer(
        model,
        question,
        &baseline,
        &verif_answers,
        cove_cfg.max_tokens_final,
        cove_cfg.temperature_final,
        eos_tokens,
    )
    .await;
    let final_ms = final_start.elapsed().as_millis();

    let total_ms = total_start.elapsed().as_millis();

    serde_json::json!({
        "question": question,
        "chain_type": chain_type.as_str(),
        "baseline_response": baseline,
        "verification_questions": verif_questions,
        "verification_answers": verif_answers.iter()
            .map(|(q, a)| serde_json::json!({"q": q, "a": a}))
            .collect::<Vec<_>>(),
        "final_answer": final_answer,
        "timings": {
            "routing_ms": routing_ms,
            "baseline_ms": baseline_ms,
            "verification_questions_ms": vq_ms,
            "execute_verification_ms": exec_ms,
            "final_answer_ms": final_ms,
            "total_ms": total_ms,
        }
    })
}

// ============================================================================
// Helper Functions
// ============================================================================

async fn route_question(
    model: &inferlet::Model,
    question: &str,
    temperature: f32,
    eos_tokens: &[Vec<u32>],
) -> ChainType {
    let prompt = ROUTER_PROMPT.replace("{question}", question);
    let mut ctx = model.create_context();
    ctx.fill(&prompt);

    let stop = max_len(16).or(ends_with_any(eos_tokens.to_vec()));
    let sampler = if temperature <= 0.1 {
        Sampler::greedy()
    } else {
        Sampler::top_p(temperature, 0.95)
    };
    let response = ctx.generate(sampler, stop).await;
    ChainType::from_str(&response)
}

async fn generate_baseline(
    model: &inferlet::Model,
    question: &str,
    chain_type: ChainType,
    max_tokens: usize,
    temperature: f32,
    eos_tokens: &[Vec<u32>],
) -> String {
    let template = match chain_type {
        ChainType::Wiki => BASELINE_PROMPT_WIKI,
        ChainType::Multi => BASELINE_PROMPT_MULTI,
        ChainType::Long => BASELINE_PROMPT_LONG,
    };
    let prompt = template.replace("{question}", question);
    let mut ctx = model.create_context();
    ctx.fill(&prompt);

    let stop = max_len(max_tokens).or(ends_with_any(eos_tokens.to_vec()));
    ctx.generate(Sampler::top_p(temperature, 0.95), stop).await
}

async fn generate_verification_questions(
    model: &inferlet::Model,
    question: &str,
    baseline: &str,
    chain_type: ChainType,
    max_questions: usize,
    max_tokens: usize,
    temperature: f32,
    eos_tokens: &[Vec<u32>],
) -> Vec<String> {
    let template = match chain_type {
        ChainType::Wiki | ChainType::Multi => VERIFICATION_QUESTION_PROMPT_MULTI,
        ChainType::Long => VERIFICATION_QUESTION_PROMPT_LONG,
    };
    let prompt = template
        .replace("{question}", question)
        .replace("{baseline_response}", baseline);
    let mut ctx = model.create_context();
    ctx.fill(&prompt);

    let stop = max_len(max_tokens).or(ends_with_any(eos_tokens.to_vec()));
    let response = ctx.generate(Sampler::top_p(temperature, 0.95), stop).await;
    parse_questions(&response, max_questions)
}

fn parse_questions(text: &str, max: usize) -> Vec<String> {
    let mut questions = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let cleaned = line
            .trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ')' || c == ' ');
        if cleaned.len() > 10 && cleaned.contains('?') {
            questions.push(cleaned.to_string());
            if questions.len() >= max {
                break;
            }
        }
    }
    questions
}

async fn execute_verification_parallel(
    model: &inferlet::Model,
    questions: &[String],
    use_search: bool,
    max_search_results: usize,
    max_tokens: usize,
    temperature: f32,
    eos_tokens: &[Vec<u32>],
) -> Vec<(String, String)> {
    if questions.is_empty() {
        return Vec::new();
    }

    // Prefill shared base context
    let mut base = model.create_context();
    base.fill("You are a helpful assistant. Answer questions accurately.\n\n");
    base.flush().await;

    // Fork and run ALL verifications in parallel
    let futures: Vec<_> = questions
        .iter()
        .map(|q| {
            let mut ctx = base.fork();
            let question = q.clone();
            let eos = eos_tokens.to_vec();

            async move {
                // Get search context if enabled
                let search_context = if use_search {
                    search_ddg(&question, max_search_results)
                        .await
                        .unwrap_or_default()
                } else {
                    String::new()
                };

                let prompt = if use_search && !search_context.is_empty() {
                    EXECUTE_PLAN_PROMPT_SEARCH
                        .replace("{search_result}", &search_context)
                        .replace("{verification_question}", &question)
                } else {
                    EXECUTE_PLAN_PROMPT_SELF.replace("{verification_question}", &question)
                };

                ctx.fill(&prompt);
                let stop = max_len(max_tokens).or(ends_with_any(eos));
                let sampler = if temperature <= 0.1 {
                    Sampler::greedy()
                } else {
                    Sampler::top_p(temperature, 0.95)
                };
                let answer = ctx.generate(sampler, stop).await;
                (question, answer)
            }
        })
        .collect();

    join_all(futures).await
}

async fn generate_final_answer(
    model: &inferlet::Model,
    question: &str,
    baseline: &str,
    verif_answers: &[(String, String)],
    max_tokens: usize,
    temperature: f32,
    eos_tokens: &[Vec<u32>],
) -> String {
    let verif_text: String = verif_answers
        .iter()
        .map(|(q, a)| format!("Question: {}\nAnswer: {}", q, a))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = FINAL_REFINED_PROMPT
        .replace("{question}", question)
        .replace("{baseline_response}", baseline)
        .replace("{verification_answers}", &verif_text);

    let mut ctx = model.create_context();
    ctx.fill(&prompt);

    let stop = max_len(max_tokens).or(ends_with_any(eos_tokens.to_vec()));
    ctx.generate(Sampler::top_p(temperature, 0.95), stop).await
}
