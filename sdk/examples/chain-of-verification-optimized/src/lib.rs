//! Optimized Chain-of-Verification (CoVe) inferlet.
//!
//! This variant keeps the same external CLI/config/result schema as the baseline
//! CoVe example, but uses Pie-native control features more aggressively:
//! - exported KV prompt modules reused across inferlet instances
//! - `fork()`-based branch sharing within a request
//! - structured WIKI verification templates and YES/NO verdicts
//! - deterministic WIKI final refinement when verification is clear

#[path = "../../chain-of-verification/src/util.rs"]
pub mod util;

use futures::future::join_all;
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use inferlet::forward::Forward;
use inferlet::stop_condition::{ends_with_any, max_len, StopCondition};
use inferlet::wstd::http::Client;
use inferlet::{store_get, store_set, Args, Context, Model, Result, Sampler};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeSet;
use std::hash::{Hash, Hasher};
use std::time::Instant;

use crate::util::*;

const BATCH_PAYLOAD_CHUNK_BYTES: usize = 4096;

// ============================================================================
// Prompts
// ============================================================================

const MODULE_CACHE_VERSION: &str = "cove-opt-v2";
const PAPER_DYNAMIC_MARKER: &str = "__COVE_PAPER_DYNAMIC__";

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

const BASELINE_PROMPT_WIKI: &str = r#"Answer the below question with ONLY a numbered list of likely entity candidates. Output at most {max_items} items. Each line must contain only the concise entity name or short noun phrase. NO EXPLANATIONS, DESCRIPTIONS, DUPLICATES, OR EXTRA PROSE.

Question: {question}

Answer:"#;

const BASELINE_PROMPT_MULTI: &str = r#"Answer the below question correctly and in a concise manner without much details. Only answer what the question is asked.

Question: {question}

Answer:"#;

const BASELINE_PROMPT_LONG: &str = r#"Answer the below question correctly.

Question: {question}

Answer:"#;

const VERIFICATION_QUESTION_TEMPLATE_PROMPT_WIKI: &str = r#"Create EXACTLY ONE short verification question for the actual query below. The question MUST contain the literal placeholder [ANSWER_ENTITY] exactly once, where the candidate entity should be inserted. Output ONLY the verification question and nothing else.

Example Query: Who are some movie actors who were born in Boston?
Example Output: Was [ANSWER_ENTITY] born in Boston?

Actual Query: {question}

Verification Question:"#;

const VERIFICATION_QUESTION_PROMPT_WIKI_FALLBACK: &str = r#"Your task is to create a series of verification questions based on the below question, the verification question template and baseline response. Output ONLY a numbered list with at most {max_items} short verification questions and no explanation.
Example Question: Who are some movie actors who were born in Boston?
Example Verification Question Template: Was [ANSWER_ENTITY] born in Boston?
Example Baseline Response: 1. Matt Damon
2. Chris Evans
Example Verification Questions: 1. Was Matt Damon born in Boston?
2. Was Chris Evans born in Boston?

Actual Question: {question}
Baseline Response: {baseline_response}
Verification Question Template: {verification_question_template}

Final Verification Questions:"#;

const VERIFICATION_QUESTION_PROMPT_MULTI: &str = r#"Your task is to create verification questions based on the below original question and the baseline response. The verification questions are meant for verifying the factual accuracy in the baseline response.
Example Question: Who invented the first printing press and in what year?
Example Baseline Response: Johannes Gutenberg, 1450.
Example Verification Questions: 1. Did Johannes Gutenberg invent first printing press?
2. Did Johannes Gutenberg invent first printing press in the year 1450?

Actual Question: {question}
Baseline Response: {baseline_response}

Final Verification Questions:"#;

const VERIFICATION_QUESTION_PROMPT_LONG: &str = r#"Your task is to create verification questions based on the below original question and the baseline response. The verification questions are meant for verifying the factual accuracy in the baseline response. Output should be numbered list of verification questions.

Actual Question: {question}
Baseline Response: {baseline_response}

Final Verification Questions:"#;

const EXECUTE_PLAN_PROMPT_SEARCH_WIKI: &str = r#"Answer the following question using ONLY YES or NO based on the provided context. Reply with only YES or NO.

Context: {search_result}

Question: {verification_question}

Answer:"#;

const EXECUTE_PLAN_PROMPT_SELF_WIKI: &str = r#"Answer the following question using ONLY YES or NO. Reply with only YES or NO.

Question: {verification_question}

Answer:"#;

const EXECUTE_PLAN_PROMPT_SEARCH_DEFAULT: &str = r#"Answer the following question correctly based on the provided context. The question could be tricky as well, so think step by step and answer it correctly.

Context: {search_result}

Question: {verification_question}

Answer:"#;

const EXECUTE_PLAN_PROMPT_SELF_DEFAULT: &str = r#"Answer the following question correctly.

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
const YES_NO_STOP_SEQUENCES: [&str; 3] = ["\n", "\n\n", "\r\n"];

// ============================================================================
// Core Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VerificationVerdict {
    Yes,
    No,
    Ambiguous,
}

#[derive(Debug, Clone)]
struct VerificationTask {
    candidate: Option<String>,
    question: String,
}

#[derive(Debug, Clone)]
struct VerificationRecord {
    candidate: Option<String>,
    question: String,
    answer: String,
    verdict: VerificationVerdict,
}

#[derive(Debug, Clone)]
struct WikiBaseline {
    text: String,
    candidates: Vec<String>,
}

#[derive(Debug, Clone)]
struct WikiVerificationPlan {
    tasks: Vec<VerificationTask>,
    used_template: bool,
}

#[derive(Debug)]
struct GeneratedOutput {
    text: String,
    ctx: Context,
    prompt_token_len: usize,
}

struct TextStopTrace {
    raw_text: String,
    trimmed_text: String,
    matched_stop_suffix: Option<String>,
    raw_token_ids: Vec<u32>,
    token_timestamps_ms: Vec<f64>,
    total_elapsed_ms: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
struct KvOptimizationStats {
    tail_compactions: u64,
    tail_tokens_masked: u64,
    tail_pages_dropped: u64,
    window_evictions: u64,
    window_tokens_masked: u64,
    window_pages_dropped: u64,
    question_mask_attempts: u64,
    question_mask_tokens_masked: u64,
    question_mask_pages_dropped: u64,
}

impl KvOptimizationStats {
    fn merge(&mut self, other: &Self) {
        self.tail_compactions += other.tail_compactions;
        self.tail_tokens_masked += other.tail_tokens_masked;
        self.tail_pages_dropped += other.tail_pages_dropped;
        self.window_evictions += other.window_evictions;
        self.window_tokens_masked += other.window_tokens_masked;
        self.window_pages_dropped += other.window_pages_dropped;
        self.question_mask_attempts += other.question_mask_attempts;
        self.question_mask_tokens_masked += other.question_mask_tokens_masked;
        self.question_mask_pages_dropped += other.question_mask_pages_dropped;
    }
}

struct FactorReviseResult {
    verification_tasks: Vec<VerificationTask>,
    verification_records: Vec<VerificationRecord>,
    cross_checks: Vec<(String, String)>,
    final_answer: String,
    verification_questions_ms: u128,
    execute_verification_ms: u128,
    cross_check_ms: u128,
    final_answer_ms: u128,
    kv_optimization: KvOptimizationStats,
}

struct PassageFactorReviseResult {
    idx: usize,
    passage: String,
    verification_tasks: Vec<VerificationTask>,
    verification_records: Vec<VerificationRecord>,
    cross_check: String,
    revised_passage: String,
    verification_questions_ms: u128,
    execute_verification_ms: u128,
    cross_check_ms: u128,
    final_answer_ms: u128,
    kv_optimization: KvOptimizationStats,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct CachedModuleState {
    cache_key: String,
    export_name: String,
    token_ids: Vec<u32>,
    kv_page_last_len: usize,
}

struct ReusablePromptModule {
    base_ctx: Context,
    suffix: String,
}

impl ReusablePromptModule {
    async fn load_or_build_prefix(model: &Model, logical_name: &str, prefix: &str) -> Self {
        Self::load_or_build_parts(model, logical_name, prefix, "").await
    }

    async fn load_or_build(
        model: &Model,
        logical_name: &str,
        template: &str,
        placeholder: &str,
    ) -> Self {
        let (prefix, suffix) = template
            .split_once(placeholder)
            .unwrap_or_else(|| panic!("Prompt missing placeholder '{}'", placeholder));

        Self::load_or_build_parts(model, logical_name, prefix, suffix).await
    }

    async fn load_or_build_parts(
        model: &Model,
        logical_name: &str,
        prefix: &str,
        suffix: &str,
    ) -> Self {
        let cache_key = make_module_cache_key(&model.get_name(), logical_name, prefix, suffix);
        let export_name = make_module_export_name(logical_name, &cache_key);
        let store_key = format!("{}.state.{}", MODULE_CACHE_VERSION, logical_name);
        let queue = model.create_queue();

        if let Some(raw_state) = store_get(&store_key) {
            if let Ok(state) = serde_json::from_str::<CachedModuleState>(&raw_state) {
                if state.cache_key == cache_key {
                    let imported_pages = queue.import_kv_pages(&state.export_name);
                    if !imported_pages.is_empty() {
                        let base_ctx = Context::from_imported_state(
                            model,
                            imported_pages,
                            state.token_ids,
                            state.kv_page_last_len,
                        );
                        return Self {
                            base_ctx,
                            suffix: suffix.to_string(),
                        };
                    }
                }

                if !state.export_name.is_empty() {
                    queue.release_exported_kv_pages(&state.export_name);
                }
            }
        }

        let mut base_ctx = model.create_context();
        base_ctx.fill(prefix);
        base_ctx.flush().await;

        let state = CachedModuleState {
            cache_key,
            export_name,
            token_ids: base_ctx.get_token_ids().to_vec(),
            kv_page_last_len: base_ctx.get_kv_page_last_len(),
        };

        base_ctx
            .queue()
            .export_kv_pages(&base_ctx.kv_pages, &state.export_name);
        store_set(
            &store_key,
            &serde_json::to_string(&state).expect("serializing cached module state"),
        );

        Self {
            base_ctx,
            suffix: suffix.to_string(),
        }
    }

    fn fork_with_suffix(&self, dynamic_text: &str) -> Context {
        self.fork_with_tail(&format!("{}{}", dynamic_text, self.suffix))
    }

    fn fork_with_tail(&self, tail: &str) -> Context {
        let mut ctx = self.base_ctx.fork();
        ctx.fill(tail);
        ctx
    }

    fn suffix(&self) -> &str {
        &self.suffix
    }
}

struct PaperModuleCache {
    baseline_wiki: ReusablePromptModule,
    baseline_multi: ReusablePromptModule,
    baseline_long: ReusablePromptModule,
    verification_wiki: ReusablePromptModule,
    verification_multi: ReusablePromptModule,
    verification_long: ReusablePromptModule,
    execute_wiki: ReusablePromptModule,
    execute_multi: ReusablePromptModule,
    execute_long: ReusablePromptModule,
    two_step_execute_wiki: ReusablePromptModule,
    two_step_execute_multi: ReusablePromptModule,
    two_step_execute_long: ReusablePromptModule,
    joint_wiki: ReusablePromptModule,
    joint_multi: ReusablePromptModule,
    joint_long: ReusablePromptModule,
    final_wiki: ReusablePromptModule,
    final_multi: ReusablePromptModule,
    final_long: ReusablePromptModule,
    cross_check_long: ReusablePromptModule,
}

impl PaperModuleCache {
    async fn new(model: &Model, few_shot_examples: usize) -> Self {
        Self {
            baseline_wiki: ReusablePromptModule::load_or_build_prefix(
                model,
                "paper_baseline_wiki",
                &build_paper_baseline_prefix("WIKI_CHAIN", few_shot_examples),
            )
            .await,
            baseline_multi: ReusablePromptModule::load_or_build_prefix(
                model,
                "paper_baseline_multi",
                &build_paper_baseline_prefix("MULTI_CHAIN", few_shot_examples),
            )
            .await,
            baseline_long: ReusablePromptModule::load_or_build_prefix(
                model,
                "paper_baseline_long",
                &build_paper_baseline_prefix("LONG_CHAIN", few_shot_examples),
            )
            .await,
            verification_wiki: ReusablePromptModule::load_or_build_prefix(
                model,
                "paper_verification_wiki",
                &build_paper_verification_prefix("WIKI_CHAIN", few_shot_examples),
            )
            .await,
            verification_multi: ReusablePromptModule::load_or_build_prefix(
                model,
                "paper_verification_multi",
                &build_paper_verification_prefix("MULTI_CHAIN", few_shot_examples),
            )
            .await,
            verification_long: ReusablePromptModule::load_or_build_prefix(
                model,
                "paper_verification_long",
                &build_paper_verification_prefix("LONG_CHAIN", few_shot_examples),
            )
            .await,
            execute_wiki: ReusablePromptModule::load_or_build_prefix(
                model,
                "paper_execute_wiki",
                &build_paper_execute_prefix("WIKI_CHAIN", few_shot_examples),
            )
            .await,
            execute_multi: ReusablePromptModule::load_or_build_prefix(
                model,
                "paper_execute_multi",
                &build_paper_execute_prefix("MULTI_CHAIN", few_shot_examples),
            )
            .await,
            execute_long: ReusablePromptModule::load_or_build_prefix(
                model,
                "paper_execute_long",
                &build_paper_execute_prefix("LONG_CHAIN", few_shot_examples),
            )
            .await,
            two_step_execute_wiki: ReusablePromptModule::load_or_build_prefix(
                model,
                "paper_two_step_execute_wiki",
                &build_paper_two_step_execute_prefix("WIKI_CHAIN", few_shot_examples),
            )
            .await,
            two_step_execute_multi: ReusablePromptModule::load_or_build_prefix(
                model,
                "paper_two_step_execute_multi",
                &build_paper_two_step_execute_prefix("MULTI_CHAIN", few_shot_examples),
            )
            .await,
            two_step_execute_long: ReusablePromptModule::load_or_build_prefix(
                model,
                "paper_two_step_execute_long",
                &build_paper_two_step_execute_prefix("LONG_CHAIN", few_shot_examples),
            )
            .await,
            joint_wiki: ReusablePromptModule::load_or_build_prefix(
                model,
                "paper_joint_wiki",
                &build_paper_joint_prefix("WIKI_CHAIN", few_shot_examples),
            )
            .await,
            joint_multi: ReusablePromptModule::load_or_build_prefix(
                model,
                "paper_joint_multi",
                &build_paper_joint_prefix("MULTI_CHAIN", few_shot_examples),
            )
            .await,
            joint_long: ReusablePromptModule::load_or_build_prefix(
                model,
                "paper_joint_long",
                &build_paper_joint_prefix("LONG_CHAIN", few_shot_examples),
            )
            .await,
            final_wiki: ReusablePromptModule::load_or_build_prefix(
                model,
                "paper_final_wiki",
                &build_paper_final_prefix("WIKI_CHAIN", few_shot_examples),
            )
            .await,
            final_multi: ReusablePromptModule::load_or_build_prefix(
                model,
                "paper_final_multi",
                &build_paper_final_prefix("MULTI_CHAIN", few_shot_examples),
            )
            .await,
            final_long: ReusablePromptModule::load_or_build_prefix(
                model,
                "paper_final_long",
                &build_paper_final_prefix("LONG_CHAIN", few_shot_examples),
            )
            .await,
            cross_check_long: ReusablePromptModule::load_or_build_prefix(
                model,
                "paper_cross_check_long",
                &build_paper_cross_check_prefix(few_shot_examples),
            )
            .await,
        }
    }

    fn baseline_prompt(&self, chain_type: ChainType) -> &ReusablePromptModule {
        match chain_type {
            ChainType::Wiki => &self.baseline_wiki,
            ChainType::Multi => &self.baseline_multi,
            ChainType::Long => &self.baseline_long,
        }
    }

    fn verification_prompt(&self, chain_type: ChainType) -> &ReusablePromptModule {
        match chain_type {
            ChainType::Wiki => &self.verification_wiki,
            ChainType::Multi => &self.verification_multi,
            ChainType::Long => &self.verification_long,
        }
    }

    fn execute_prompt(&self, chain_type: ChainType) -> &ReusablePromptModule {
        match chain_type {
            ChainType::Wiki => &self.execute_wiki,
            ChainType::Multi => &self.execute_multi,
            ChainType::Long => &self.execute_long,
        }
    }

    fn two_step_execute_prompt(&self, chain_type: ChainType) -> &ReusablePromptModule {
        match chain_type {
            ChainType::Wiki => &self.two_step_execute_wiki,
            ChainType::Multi => &self.two_step_execute_multi,
            ChainType::Long => &self.two_step_execute_long,
        }
    }

    fn joint_prompt(&self, chain_type: ChainType) -> &ReusablePromptModule {
        match chain_type {
            ChainType::Wiki => &self.joint_wiki,
            ChainType::Multi => &self.joint_multi,
            ChainType::Long => &self.joint_long,
        }
    }

    fn final_prompt(&self, chain_type: ChainType) -> &ReusablePromptModule {
        match chain_type {
            ChainType::Wiki => &self.final_wiki,
            ChainType::Multi => &self.final_multi,
            ChainType::Long => &self.final_long,
        }
    }
}

struct ModuleCache {
    router: ReusablePromptModule,
    baseline_wiki: ReusablePromptModule,
    baseline_multi: ReusablePromptModule,
    baseline_long: ReusablePromptModule,
    verification_template_wiki: ReusablePromptModule,
    verification_wiki_fallback: ReusablePromptModule,
    verification_multi: ReusablePromptModule,
    verification_long: ReusablePromptModule,
    execute_search_wiki: ReusablePromptModule,
    execute_self_wiki: ReusablePromptModule,
    execute_search_default: ReusablePromptModule,
    execute_self_default: ReusablePromptModule,
    final_refined_default: ReusablePromptModule,
    final_refined_wiki: ReusablePromptModule,
    paper: PaperModuleCache,
}

impl ModuleCache {
    async fn new(model: &Model, wiki_max_items: usize, few_shot_examples: usize) -> Self {
        let wiki_max_items = wiki_max_items.max(1).to_string();
        let baseline_wiki = BASELINE_PROMPT_WIKI.replace("{max_items}", &wiki_max_items);
        let verification_wiki_fallback =
            VERIFICATION_QUESTION_PROMPT_WIKI_FALLBACK.replace("{max_items}", &wiki_max_items);
        let final_refined_wiki = FINAL_REFINED_PROMPT_WIKI.replace("{max_items}", &wiki_max_items);

        Self {
            router: ReusablePromptModule::load_or_build(
                model,
                "router",
                ROUTER_PROMPT,
                "{question}",
            )
            .await,
            baseline_wiki: ReusablePromptModule::load_or_build(
                model,
                "baseline_wiki",
                &baseline_wiki,
                "{question}",
            )
            .await,
            baseline_multi: ReusablePromptModule::load_or_build(
                model,
                "baseline_multi",
                BASELINE_PROMPT_MULTI,
                "{question}",
            )
            .await,
            baseline_long: ReusablePromptModule::load_or_build(
                model,
                "baseline_long",
                BASELINE_PROMPT_LONG,
                "{question}",
            )
            .await,
            verification_template_wiki: ReusablePromptModule::load_or_build(
                model,
                "verification_template_wiki",
                VERIFICATION_QUESTION_TEMPLATE_PROMPT_WIKI,
                "{question}",
            )
            .await,
            verification_wiki_fallback: ReusablePromptModule::load_or_build(
                model,
                "verification_wiki_fallback",
                &verification_wiki_fallback,
                "{question}",
            )
            .await,
            verification_multi: ReusablePromptModule::load_or_build(
                model,
                "verification_multi",
                VERIFICATION_QUESTION_PROMPT_MULTI,
                "{question}",
            )
            .await,
            verification_long: ReusablePromptModule::load_or_build(
                model,
                "verification_long",
                VERIFICATION_QUESTION_PROMPT_LONG,
                "{question}",
            )
            .await,
            execute_search_wiki: ReusablePromptModule::load_or_build(
                model,
                "execute_search_wiki",
                EXECUTE_PLAN_PROMPT_SEARCH_WIKI,
                "{search_result}",
            )
            .await,
            execute_self_wiki: ReusablePromptModule::load_or_build(
                model,
                "execute_self_wiki",
                EXECUTE_PLAN_PROMPT_SELF_WIKI,
                "{verification_question}",
            )
            .await,
            execute_search_default: ReusablePromptModule::load_or_build(
                model,
                "execute_search_default",
                EXECUTE_PLAN_PROMPT_SEARCH_DEFAULT,
                "{search_result}",
            )
            .await,
            execute_self_default: ReusablePromptModule::load_or_build(
                model,
                "execute_self_default",
                EXECUTE_PLAN_PROMPT_SELF_DEFAULT,
                "{verification_question}",
            )
            .await,
            final_refined_default: ReusablePromptModule::load_or_build(
                model,
                "final_refined_default",
                FINAL_REFINED_PROMPT,
                "{question}",
            )
            .await,
            final_refined_wiki: ReusablePromptModule::load_or_build(
                model,
                "final_refined_wiki",
                &final_refined_wiki,
                "{question}",
            )
            .await,
            paper: PaperModuleCache::new(model, few_shot_examples).await,
        }
    }

    fn baseline_prompt(&self, chain_type: ChainType) -> &ReusablePromptModule {
        match chain_type {
            ChainType::Wiki => &self.baseline_wiki,
            ChainType::Multi => &self.baseline_multi,
            ChainType::Long => &self.baseline_long,
        }
    }

    fn final_prompt(&self, chain_type: ChainType) -> &ReusablePromptModule {
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
    yes_no_text: Vec<String>,
    yes_no_tokens: Vec<Vec<u32>>,
}

impl StopSequences {
    fn new(model: &Model, wiki_max_items: usize) -> Self {
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

        let yes_no_text = YES_NO_STOP_SEQUENCES
            .iter()
            .map(|stop| (*stop).to_string())
            .collect::<Vec<_>>();
        let mut yes_no_tokens = model.eos_tokens();
        for stop in &yes_no_text {
            yes_no_tokens.push(tokenizer.tokenize(stop));
        }

        Self {
            default_text,
            default_tokens,
            wiki_list_text,
            wiki_list_tokens,
            yes_no_text,
            yes_no_tokens,
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

    fn yes_no_tokens(&self) -> &[Vec<u32>] {
        &self.yes_no_tokens
    }

    fn yes_no_text(&self) -> &[String] {
        &self.yes_no_text
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
// Entry Point
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
    let module_cache =
        ModuleCache::new(&model, cfg.cove.max_wiki_items, cfg.cove.few_shot_examples).await;
    let search_client = Client::new();
    let search_resources =
        SearchResources::new(cfg.cove.search_mode, cfg.cove.search_cache_path.clone())?;

    if let Some(prompt) = debug_prompt {
        let stop_text: Vec<String> = debug_stop_json
            .as_deref()
            .map(|raw| {
                serde_json::from_str(raw).expect("debug_stop_json must be a JSON string array")
            })
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

    if let Some(question) = single_question {
        let result = run_cove_single(
            &model,
            &module_cache,
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

    let dataset = load_dataset(&cfg.data.dataset_path);
    let validation_len = cfg.data.num_samples.min(dataset.validation.len());
    let forced_chain_type = dataset.chain_type.as_ref().map(|s| ChainType::from_str(s));

    println!(
        "Chain-of-Verification Optimized Pipeline\n\
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
    let question_concurrency = cfg.cove.question_concurrency.max(1);

    let mut in_flight = FuturesUnordered::new();
    let mut pending_examples = dataset.validation.iter().take(validation_len).enumerate();
    let mut completed = Vec::with_capacity(validation_len);

    while completed.len() < validation_len {
        while in_flight.len() < question_concurrency {
            let Some((idx, example)) = pending_examples.next() else {
                break;
            };

            let question = example.question.clone();
            let refs = example.answer_texts();
            let model_ref = &model;
            let module_cache_ref = &module_cache;
            let search_client_ref = &search_client;
            let search_resources_ref = &search_resources;
            let cove_cfg_ref = &cfg.cove;
            let stop_sequences_ref = &stop_sequences;
            let metric_spec = metric_spec;

            in_flight.push(async move {
                let request_start = Instant::now();
                let result = run_cove_single(
                    model_ref,
                    module_cache_ref,
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
            "chain_type": entry.result["chain_type"],
            "cove_variant": entry.result["cove_variant"],
            "paper_execute_strategy": entry.result["paper_execute_strategy"],
            "baseline_response": entry.result["baseline_response"],
            "verification_questions": entry.result["verification_questions"],
            "verification_answers": entry.result["verification_answers"],
            "cross_check_results": entry.result["cross_check_results"],
            "final_answer": entry.result["final_answer"],
            "baseline_score": entry.baseline_score,
            "cove_score": entry.cove_score,
            "baseline_pass": entry.baseline_pass,
            "cove_pass": entry.cove_pass,
            "request_latency_s": entry.request_latency_s,
            "timings": entry.result["timings"],
            "kv_optimization": entry.result["kv_optimization"],
        }));
    }

    let total_elapsed = total_start.elapsed();
    search_resources.persist()?;

    let mut sum_routing_ms: u64 = 0;
    let mut sum_baseline_ms: u64 = 0;
    let mut sum_vq_ms: u64 = 0;
    let mut sum_exec_ms: u64 = 0;
    let mut sum_cross_check_ms: u64 = 0;
    let mut sum_final_ms: u64 = 0;
    let mut total_tail_compactions: u64 = 0;
    let mut total_tail_tokens_masked: u64 = 0;
    let mut total_tail_pages_dropped: u64 = 0;
    let mut total_window_evictions: u64 = 0;
    let mut total_window_tokens_masked: u64 = 0;
    let mut total_window_pages_dropped: u64 = 0;
    let mut total_question_mask_attempts: u64 = 0;
    let mut total_question_mask_tokens_masked: u64 = 0;
    let mut total_question_mask_pages_dropped: u64 = 0;

    for result in &results {
        if let Some(timings) = result.get("timings") {
            sum_routing_ms += timings["routing_ms"].as_u64().unwrap_or(0);
            sum_baseline_ms += timings["baseline_ms"].as_u64().unwrap_or(0);
            sum_vq_ms += timings["verification_questions_ms"].as_u64().unwrap_or(0);
            sum_exec_ms += timings["execute_verification_ms"].as_u64().unwrap_or(0);
            sum_cross_check_ms += timings["cross_check_ms"].as_u64().unwrap_or(0);
            sum_final_ms += timings["final_answer_ms"].as_u64().unwrap_or(0);
        }
        if let Some(kv) = result.get("kv_optimization") {
            total_tail_compactions += kv["tail_compactions"].as_u64().unwrap_or(0);
            total_tail_tokens_masked += kv["tail_tokens_masked"].as_u64().unwrap_or(0);
            total_tail_pages_dropped += kv["tail_pages_dropped"].as_u64().unwrap_or(0);
            total_window_evictions += kv["window_evictions"].as_u64().unwrap_or(0);
            total_window_tokens_masked += kv["window_tokens_masked"].as_u64().unwrap_or(0);
            total_window_pages_dropped += kv["window_pages_dropped"].as_u64().unwrap_or(0);
            total_question_mask_attempts += kv["question_mask_attempts"].as_u64().unwrap_or(0);
            total_question_mask_tokens_masked +=
                kv["question_mask_tokens_masked"].as_u64().unwrap_or(0);
            total_question_mask_pages_dropped +=
                kv["question_mask_pages_dropped"].as_u64().unwrap_or(0);
        }
    }

    let n = validation_len as f64;
    let avg_routing_s = (sum_routing_ms as f64) / 1000.0 / n;
    let avg_baseline_s = (sum_baseline_ms as f64) / 1000.0 / n;
    let avg_verification_questions_s = (sum_vq_ms as f64) / 1000.0 / n;
    let avg_execute_verification_s = (sum_exec_ms as f64) / 1000.0 / n;
    let avg_cross_check_s = (sum_cross_check_ms as f64) / 1000.0 / n;
    let avg_final_answer_s = (sum_final_ms as f64) / 1000.0 / n;

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
        "paper_execute_strategy": effective_paper_execute_strategy(&cfg.cove).as_str(),
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
            "avg_cross_check_s": avg_cross_check_s,
            "avg_final_answer_s": avg_final_answer_s,
        },
        "kv_optimization": {
            "mode": cfg.cove.kv_optimization_mode.as_str(),
            "tail_compactions": total_tail_compactions,
            "tail_tokens_masked": total_tail_tokens_masked,
            "tail_pages_dropped": total_tail_pages_dropped,
            "window_evictions": total_window_evictions,
            "window_tokens_masked": total_window_tokens_masked,
            "window_pages_dropped": total_window_pages_dropped,
            "question_mask_attempts": total_question_mask_attempts,
            "question_mask_tokens_masked": total_question_mask_tokens_masked,
            "question_mask_pages_dropped": total_question_mask_pages_dropped,
        },
    });

    println!("\n{}", "=".repeat(60));
    println!("CHAIN-OF-VERIFICATION OPTIMIZED RESULTS");
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
    println!("  Cross Check:            {:.4}s", avg_cross_check_s);
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
// Pipeline
// ============================================================================

async fn run_cove_single(
    model: &Model,
    module_cache: &ModuleCache,
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

    let routing_start = Instant::now();
    let chain_type = if let Some(ct) = forced_chain_type {
        ct
    } else {
        route_question(
            module_cache,
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
        println!("[CoVe-Opt] Chain: {:?} ({}ms)", chain_type, routing_ms);
    }

    let paper_variant = resolve_paper_variant(cove_cfg, None, chain_type.as_str());
    let mut kv_optimization = KvOptimizationStats::default();

    let baseline_start = Instant::now();
    let wiki_baseline = if chain_type == ChainType::Wiki && !cove_cfg.paper_faithful {
        let (baseline, optimization) =
            generate_wiki_baseline(model, module_cache, question, cove_cfg, stop_sequences).await;
        kv_optimization.merge(&optimization);
        Some(baseline)
    } else {
        None
    };
    let baseline = if let Some(wiki_baseline) = wiki_baseline.as_ref() {
        wiki_baseline.text.clone()
    } else {
        generate_baseline_default(
            model,
            module_cache,
            question,
            chain_type,
            effective_baseline_tokens(cove_cfg, chain_type),
            effective_phase_temperature(cove_cfg.temperature_baseline, cove_cfg),
            cove_cfg,
            stop_sequences,
        )
        .await
    };
    let baseline_ms = baseline_start.elapsed().as_millis();

    if verbose {
        println!(
            "[CoVe-Opt] Baseline ({}ms): {}...",
            baseline_ms,
            &baseline[..baseline.len().min(80)]
        );
    }

    let mut cross_checks: Vec<(String, String)> = Vec::new();
    let (
        _verification_tasks,
        verification_records,
        _structured_wiki_plan,
        vq_ms,
        exec_ms,
        cross_check_ms,
        final_answer,
        final_ms,
    ) = if cove_cfg.paper_faithful && paper_variant == CoVeVariant::Joint {
        let vq_start = Instant::now();
        let verification_records = generate_joint_verification_records(
            model,
            module_cache,
            question,
            &baseline,
            chain_type,
            cove_cfg,
        )
        .await;
        let verification_tasks = verification_records
            .iter()
            .map(|record| VerificationTask {
                candidate: None,
                question: record.question.clone(),
            })
            .collect::<Vec<_>>();
        let final_start = Instant::now();
        let (final_answer, final_optimization) = generate_final_answer(
            model,
            module_cache,
            question,
            &baseline,
            &verification_records,
            &cross_checks,
            chain_type,
            cove_cfg,
            paper_variant,
            stop_sequences,
            false,
        )
        .await;
        kv_optimization.merge(&final_optimization);
        (
            verification_tasks,
            verification_records,
            false,
            vq_start.elapsed().as_millis(),
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
            module_cache,
            search_client,
            search_resources,
            question,
            &baseline,
            chain_type,
            cove_cfg,
            stop_sequences,
        )
        .await;
        kv_optimization.merge(&result.kv_optimization);
        cross_checks = result.cross_checks;
        (
            result.verification_tasks,
            result.verification_records,
            false,
            result.verification_questions_ms,
            result.execute_verification_ms,
            result.cross_check_ms,
            result.final_answer,
            result.final_answer_ms,
        )
    } else {
        let vq_start = Instant::now();
        let (verification_tasks, structured_wiki_plan) = if cove_cfg.paper_faithful {
            (
                generate_default_verification_tasks(
                    model,
                    module_cache,
                    question,
                    &baseline,
                    chain_type,
                    effective_max_verification_questions(cove_cfg, chain_type),
                    effective_verification_tokens(cove_cfg, chain_type),
                    effective_phase_temperature(cove_cfg.temperature_verification, cove_cfg),
                    cove_cfg,
                    stop_sequences,
                )
                .await,
                false,
            )
        } else {
            match chain_type {
                ChainType::Wiki => {
                    let plan = generate_wiki_verification_plan(
                        module_cache,
                        question,
                        wiki_baseline
                            .as_ref()
                            .expect("WIKI baseline should be available for WIKI verification"),
                        cove_cfg,
                        stop_sequences,
                    )
                    .await;
                    (plan.tasks, plan.used_template)
                }
                ChainType::Multi | ChainType::Long => (
                    generate_default_verification_tasks(
                        model,
                        module_cache,
                        question,
                        &baseline,
                        chain_type,
                        effective_max_verification_questions(cove_cfg, chain_type),
                        effective_verification_tokens(cove_cfg, chain_type),
                        effective_phase_temperature(cove_cfg.temperature_verification, cove_cfg),
                        cove_cfg,
                        stop_sequences,
                    )
                    .await,
                    false,
                ),
            }
        };
        let exec_start = Instant::now();
        let (verification_records, execute_optimization) = if cove_cfg.paper_faithful {
            if debug_trace && paper_variant == CoVeVariant::TwoStep {
                let questions = verification_tasks
                    .iter()
                    .map(|task| task.question.clone())
                    .collect::<Vec<_>>();
                let stop_text = paper_stop_sequences(
                    "two_step_execute",
                    chain_type.as_str(),
                    cove_cfg.max_wiki_items,
                );
                let ctx = module_cache
                    .paper
                    .two_step_execute_prompt(chain_type)
                    .fork_with_tail(&format!("{}\nResponse:", questions.join("\n")));
                let trace = generate_from_context_text_stop_trace(
                    ctx,
                    model,
                    paper_batch_tokens(cove_cfg, chain_type),
                    effective_phase_temperature(cove_cfg.temperature_verification, cove_cfg),
                    &stop_text,
                )
                .await;
                let answers = parse_numbered_answers(&trace.trimmed_text, verification_tasks.len());
                debug_two_step_execute = Some(serde_json::json!({
                    "prompt": build_paper_two_step_execute_prompt(
                        &questions,
                        chain_type.as_str(),
                        cove_cfg.few_shot_examples,
                    ),
                    "stop_text": stop_text,
                    "matched_stop_suffix": trace.matched_stop_suffix,
                    "raw_response": trace.raw_text,
                    "trimmed_response": trace.trimmed_text,
                    "raw_token_ids": trace.raw_token_ids,
                    "token_timestamps_ms": trace.token_timestamps_ms,
                    "total_elapsed_ms": trace.total_elapsed_ms,
                }));
                (
                    verification_tasks
                        .iter()
                        .cloned()
                        .zip(answers.into_iter())
                        .map(|(task, answer)| VerificationRecord {
                            candidate: task.candidate,
                            question: task.question,
                            answer,
                            verdict: VerificationVerdict::Ambiguous,
                        })
                        .collect(),
                    KvOptimizationStats::default(),
                )
            } else {
                execute_verification_parallel_default(
                    model,
                    module_cache,
                    search_client,
                    search_resources,
                    &verification_tasks,
                    chain_type,
                    cove_cfg,
                    paper_variant,
                    stop_sequences,
                )
                .await
            }
        } else {
            match chain_type {
                ChainType::Wiki => (
                    execute_verification_parallel_wiki(
                        module_cache,
                        search_client,
                        search_resources,
                        &verification_tasks,
                        effective_use_search(cove_cfg),
                        cove_cfg.max_search_results,
                        cove_cfg.search_timeout,
                        effective_verification_tokens(cove_cfg, chain_type).min(4),
                        stop_sequences,
                    )
                    .await,
                    KvOptimizationStats::default(),
                ),
                ChainType::Multi | ChainType::Long => {
                    execute_verification_parallel_default(
                        model,
                        module_cache,
                        search_client,
                        search_resources,
                        &verification_tasks,
                        chain_type,
                        cove_cfg,
                        paper_variant,
                        stop_sequences,
                    )
                    .await
                }
            }
        };
        kv_optimization.merge(&execute_optimization);
        let final_start = Instant::now();
        let (final_answer, final_optimization) = generate_final_answer(
            model,
            module_cache,
            question,
            &baseline,
            &verification_records,
            &cross_checks,
            chain_type,
            cove_cfg,
            paper_variant,
            stop_sequences,
            structured_wiki_plan,
        )
        .await;
        kv_optimization.merge(&final_optimization);
        (
            verification_tasks,
            verification_records,
            structured_wiki_plan,
            vq_start.elapsed().as_millis(),
            exec_start.elapsed().as_millis(),
            0,
            final_answer,
            final_start.elapsed().as_millis(),
        )
    };

    let total_ms = total_start.elapsed().as_millis();

    let mut result = serde_json::json!({
        "question": question,
        "chain_type": chain_type.as_str(),
        "cove_variant": paper_variant.as_str(),
        "paper_execute_strategy": effective_paper_execute_strategy(cove_cfg).as_str(),
        "baseline_response": baseline,
        "verification_questions": verification_records.iter().map(|record| record.question.clone()).collect::<Vec<_>>(),
        "verification_answers": verification_records.iter()
            .map(|record| serde_json::json!({"q": record.question, "a": record.answer}))
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
        },
        "kv_optimization": kv_optimization,
    });
    if let Some(debug) = debug_two_step_execute {
        result["debug_two_step_execute"] = debug;
    }
    result
}

async fn route_question(
    module_cache: &ModuleCache,
    question: &str,
    max_tokens: usize,
    temperature: f32,
    stop_tokens: &[Vec<u32>],
    stop_text: &[String],
) -> ChainType {
    let ctx = module_cache.router.fork_with_suffix(question);
    let response =
        generate_from_context(ctx, max_tokens, temperature, stop_tokens, stop_text).await;
    ChainType::from_str(&response)
}

async fn generate_baseline_default(
    model: &Model,
    module_cache: &ModuleCache,
    question: &str,
    chain_type: ChainType,
    max_tokens: usize,
    temperature: f32,
    cove_cfg: &CoVeConfig,
    stop_sequences: &StopSequences,
) -> String {
    if cove_cfg.paper_faithful {
        let ctx = module_cache
            .paper
            .baseline_prompt(chain_type)
            .fork_with_tail(&format!("{}\nA:", question));
        let stop_text =
            paper_stop_sequences("baseline", chain_type.as_str(), cove_cfg.max_wiki_items);
        let stop_tokens = stop_tokens_for_text(model, &stop_text);
        return generate_from_context(ctx, max_tokens, temperature, &stop_tokens, &stop_text).await;
    }

    let ctx = module_cache
        .baseline_prompt(chain_type)
        .fork_with_suffix(question);
    let (stop_tokens, stop_text) = wiki_list_stop_set(chain_type, stop_sequences);
    generate_from_context(ctx, max_tokens, temperature, stop_tokens, stop_text).await
}

async fn generate_wiki_baseline(
    _model: &Model,
    module_cache: &ModuleCache,
    question: &str,
    cove_cfg: &CoVeConfig,
    stop_sequences: &StopSequences,
) -> (WikiBaseline, KvOptimizationStats) {
    let ctx = module_cache.baseline_wiki.fork_with_suffix(question);
    let mut kv_optimization = KvOptimizationStats::default();
    let mut generated = generate_with_state(
        ctx,
        effective_baseline_tokens(cove_cfg, ChainType::Wiki),
        cove_cfg.temperature_baseline,
        stop_sequences.wiki_list_tokens(),
        stop_sequences.wiki_list_text(),
    )
    .await;

    let candidates = normalize_wiki_candidates(&generated.text, cove_cfg.max_wiki_items);
    let normalized_text = format_numbered_list(&candidates);
    if !normalized_text.is_empty() {
        maybe_compact_generated_tail(
            &mut generated.ctx,
            generated.prompt_token_len,
            &normalized_text,
            cove_cfg,
            &mut kv_optimization,
        );
    }

    (
        WikiBaseline {
            text: if normalized_text.is_empty() {
                generated.text
            } else {
                normalized_text
            },
            candidates,
        },
        kv_optimization,
    )
}

async fn generate_wiki_verification_plan(
    module_cache: &ModuleCache,
    question: &str,
    baseline: &WikiBaseline,
    cove_cfg: &CoVeConfig,
    stop_sequences: &StopSequences,
) -> WikiVerificationPlan {
    let template_ctx = module_cache
        .verification_template_wiki
        .fork_with_suffix(question);
    let template = generate_from_context(
        template_ctx,
        effective_verification_tokens(cove_cfg, ChainType::Wiki).min(24),
        0.1,
        stop_sequences.default_tokens(),
        stop_sequences.default_text(),
    )
    .await;

    let candidates = baseline
        .candidates
        .iter()
        .take(effective_max_verification_questions(
            cove_cfg,
            ChainType::Wiki,
        ))
        .cloned()
        .collect::<Vec<_>>();

    if template_contains_answer_placeholder(&template) && !candidates.is_empty() {
        let tasks = candidates
            .into_iter()
            .filter_map(|candidate| {
                instantiate_wiki_verification_question(&template, &candidate).map(|question| {
                    VerificationTask {
                        candidate: Some(candidate),
                        question,
                    }
                })
            })
            .collect::<Vec<_>>();

        if !tasks.is_empty() {
            return WikiVerificationPlan {
                tasks,
                used_template: true,
            };
        }
    }

    let tail = format!(
        "{}{}",
        question,
        module_cache
            .verification_wiki_fallback
            .suffix()
            .replace("{baseline_response}", &baseline.text)
            .replace("{verification_question_template}", &template)
    );
    let ctx = module_cache
        .verification_wiki_fallback
        .fork_with_tail(&tail);
    let response = generate_from_context(
        ctx,
        effective_verification_tokens(cove_cfg, ChainType::Wiki),
        cove_cfg.temperature_verification,
        stop_sequences.wiki_list_tokens(),
        stop_sequences.wiki_list_text(),
    )
    .await;

    let tasks = parse_verification_questions(
        &response,
        effective_max_verification_questions(cove_cfg, ChainType::Wiki),
    )
    .into_iter()
    .map(|question| VerificationTask {
        candidate: None,
        question,
    })
    .collect::<Vec<_>>();

    WikiVerificationPlan {
        tasks,
        used_template: false,
    }
}

async fn generate_default_verification_tasks(
    model: &Model,
    module_cache: &ModuleCache,
    question: &str,
    baseline: &str,
    chain_type: ChainType,
    max_questions: usize,
    max_tokens: usize,
    temperature: f32,
    cove_cfg: &CoVeConfig,
    stop_sequences: &StopSequences,
) -> Vec<VerificationTask> {
    if cove_cfg.paper_faithful {
        let ctx = module_cache
            .paper
            .verification_prompt(chain_type)
            .fork_with_tail(&format!("{}\nA: {}\nResponse:", question, baseline));
        let stop_text =
            paper_stop_sequences("verification", chain_type.as_str(), cove_cfg.max_wiki_items);
        let stop_tokens = stop_tokens_for_text(model, &stop_text);
        let response =
            generate_from_context(ctx, max_tokens, temperature, &stop_tokens, &stop_text).await;
        return parse_verification_questions(&response, max_questions)
            .into_iter()
            .map(|question| VerificationTask {
                candidate: None,
                question,
            })
            .collect();
    }

    let _ = model;
    let ctx = match chain_type {
        ChainType::Multi => {
            let tail = format!(
                "{}{}",
                question,
                module_cache
                    .verification_multi
                    .suffix()
                    .replace("{baseline_response}", baseline)
            );
            module_cache.verification_multi.fork_with_tail(&tail)
        }
        ChainType::Long => {
            let tail = format!(
                "{}{}",
                question,
                module_cache
                    .verification_long
                    .suffix()
                    .replace("{baseline_response}", baseline)
            );
            module_cache.verification_long.fork_with_tail(&tail)
        }
        ChainType::Wiki => unreachable!("WIKI uses structured verification planning"),
    };

    let response = generate_from_context(
        ctx,
        max_tokens,
        temperature,
        stop_sequences.default_tokens(),
        stop_sequences.default_text(),
    )
    .await;

    parse_verification_questions(&response, max_questions)
        .into_iter()
        .map(|question| VerificationTask {
            candidate: None,
            question,
        })
        .collect()
}

async fn generate_joint_verification_records(
    model: &Model,
    module_cache: &ModuleCache,
    question: &str,
    baseline: &str,
    chain_type: ChainType,
    cove_cfg: &CoVeConfig,
) -> Vec<VerificationRecord> {
    let ctx = module_cache
        .paper
        .joint_prompt(chain_type)
        .fork_with_tail(&format!("{}\nA: {}\nResponse:", question, baseline));
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
    .into_iter()
    .map(|(question, answer)| VerificationRecord {
        candidate: None,
        question,
        answer,
        verdict: VerificationVerdict::Ambiguous,
    })
    .collect()
}

async fn execute_verification_parallel_wiki(
    module_cache: &ModuleCache,
    search_client: &Client,
    search_resources: &SearchResources,
    tasks: &[VerificationTask],
    use_search: bool,
    max_search_results: usize,
    search_timeout: f32,
    max_tokens: usize,
    stop_sequences: &StopSequences,
) -> Vec<VerificationRecord> {
    if tasks.is_empty() {
        return Vec::new();
    }

    let futures = tasks.iter().map(|task| {
        let task = task.clone();
        let search_client = search_client;
        let search_resources = search_resources;
        let module_cache = module_cache;
        let stops = stop_sequences.yes_no_tokens().to_vec();
        let stop_text = stop_sequences.yes_no_text().to_vec();

        async move {
            let search_context = if use_search {
                match search_resources
                    .search_ddg(
                        search_client,
                        &task.question,
                        max_search_results,
                        search_timeout,
                    )
                    .await
                {
                    Ok(result) => result,
                    Err(error) => {
                        if search_resources.mode() == SearchMode::Replay {
                            panic!("Frozen search failed for '{}': {}", task.question, error);
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
                    module_cache
                        .execute_search_wiki
                        .suffix()
                        .replace("{verification_question}", &task.question)
                );
                module_cache.execute_search_wiki.fork_with_tail(&tail)
            } else {
                module_cache
                    .execute_self_wiki
                    .fork_with_suffix(&task.question)
            };

            let raw_answer =
                generate_from_context(ctx, max_tokens.max(1), 0.01, &stops, &stop_text).await;
            let verdict = parse_yes_no_verdict(&raw_answer);
            let answer = canonicalize_verdict_text(&raw_answer, verdict);

            VerificationRecord {
                candidate: task.candidate,
                question: task.question,
                answer,
                verdict,
            }
        }
    });

    join_all(futures).await
}

fn effective_paper_execute_strategy(cove_cfg: &CoVeConfig) -> PaperExecuteStrategy {
    match cove_cfg.paper_execute_strategy {
        PaperExecuteStrategy::Auto => PaperExecuteStrategy::SharedRoot,
        other => other,
    }
}

fn paper_execute_tail(question: &str) -> String {
    format!("{}\nA:", question)
}

fn select_padding_token_id(model: &Model) -> u32 {
    let tokenizer = model.get_tokenizer();
    for candidate in ["\n", " ", "\t", ".", ","] {
        let token_ids = tokenizer.tokenize(candidate);
        if token_ids.len() == 1 {
            return token_ids[0];
        }
    }
    tokenizer.tokenize("a").into_iter().next().unwrap_or(0)
}

fn select_continuation_token_id(model: &Model) -> u32 {
    let tokenizer = model.get_tokenizer();
    for candidate in [" ", "\n", "\t"] {
        let token_ids = tokenizer.tokenize(candidate);
        if token_ids.len() == 1 {
            return token_ids[0];
        }
    }
    select_padding_token_id(model)
}

async fn pad_context_to_page_boundary(ctx: &mut Context, pad_token_id: u32) {
    if ctx.kv_page_size == 0 {
        return;
    }
    let total_tokens = ctx.get_token_ids().len() + ctx.token_ids_pending.len();
    let aligned = align_up(total_tokens, ctx.kv_page_size);
    let needed = aligned.saturating_sub(total_tokens);
    if needed == 0 {
        return;
    }
    ctx.fill_tokens(vec![pad_token_id; needed]);
    ctx.flush().await;
}

async fn execute_verification_paper_fat_prompt(
    model: &Model,
    tasks: &[VerificationTask],
    chain_type: ChainType,
    cove_cfg: &CoVeConfig,
    max_tokens: usize,
    temperature: f32,
    paper_stop_text: &[String],
) -> (Vec<VerificationRecord>, KvOptimizationStats) {
    let futures = tasks.iter().cloned().map(|task| {
        let stop_text = paper_stop_text.to_vec();
        async move {
            let prompt = build_paper_execute_prompt(
                &task.question,
                chain_type.as_str(),
                cove_cfg.few_shot_examples,
            );
            let mut ctx = model.create_context();
            ctx.fill(&prompt);
            let mut generated =
                generate_with_state_text_stop(ctx, model, max_tokens, temperature, &stop_text)
                    .await;
            let answer = generated.text.clone();
            let mut kv_optimization = KvOptimizationStats::default();
            maybe_compact_generated_tail(
                &mut generated.ctx,
                generated.prompt_token_len,
                &answer,
                cove_cfg,
                &mut kv_optimization,
            );
            (
                VerificationRecord {
                    candidate: task.candidate,
                    question: task.question,
                    answer,
                    verdict: VerificationVerdict::Ambiguous,
                },
                kv_optimization,
            )
        }
    });

    let mut kv_optimization = KvOptimizationStats::default();
    let records = join_all(futures)
        .await
        .into_iter()
        .map(|(record, optimization)| {
            kv_optimization.merge(&optimization);
            record
        })
        .collect();
    (records, kv_optimization)
}

async fn execute_verification_paper_shared_root(
    model: &Model,
    module_cache: &ModuleCache,
    tasks: &[VerificationTask],
    chain_type: ChainType,
    cove_cfg: &CoVeConfig,
    max_tokens: usize,
    temperature: f32,
    paper_stop_text: &[String],
) -> (Vec<VerificationRecord>, KvOptimizationStats) {
    let futures = tasks.iter().cloned().map(|task| {
        let stop_text = paper_stop_text.to_vec();
        async move {
            let ctx = module_cache
                .paper
                .execute_prompt(chain_type)
                .fork_with_tail(&paper_execute_tail(&task.question));
            let mut generated =
                generate_with_state_text_stop(ctx, model, max_tokens, temperature, &stop_text)
                    .await;
            let answer = generated.text.clone();
            let mut kv_optimization = KvOptimizationStats::default();
            maybe_compact_generated_tail(
                &mut generated.ctx,
                generated.prompt_token_len,
                &answer,
                cove_cfg,
                &mut kv_optimization,
            );
            (
                VerificationRecord {
                    candidate: task.candidate,
                    question: task.question,
                    answer,
                    verdict: VerificationVerdict::Ambiguous,
                },
                kv_optimization,
            )
        }
    });

    let mut kv_optimization = KvOptimizationStats::default();
    let records = join_all(futures)
        .await
        .into_iter()
        .map(|(record, optimization)| {
            kv_optimization.merge(&optimization);
            record
        })
        .collect();
    (records, kv_optimization)
}

async fn execute_verification_paper_masked_chain(
    model: &Model,
    module_cache: &ModuleCache,
    tasks: &[VerificationTask],
    chain_type: ChainType,
    cove_cfg: &CoVeConfig,
    max_tokens: usize,
    temperature: f32,
    paper_stop_text: &[String],
) -> (Vec<VerificationRecord>, KvOptimizationStats) {
    if tasks.is_empty() {
        return (Vec::new(), KvOptimizationStats::default());
    }

    let pad_token_id = select_padding_token_id(model);
    let continuation_token_id = select_continuation_token_id(model);
    let mut chain_ctx = module_cache.paper.execute_prompt(chain_type).fork_with_tail("");
    let mut snapshots: Vec<(VerificationTask, Context, usize, usize)> = Vec::with_capacity(tasks.len());

    pad_context_to_page_boundary(&mut chain_ctx, pad_token_id).await;
    let mut first_block_start = chain_ctx.get_token_ids().len();

    for (idx, task) in tasks.iter().cloned().enumerate() {
        let current_start = chain_ctx.get_token_ids().len();
        if idx == 0 {
            first_block_start = current_start;
        }
        chain_ctx.fill(&paper_execute_tail(&task.question));
        chain_ctx.flush().await;
        snapshots.push((task, chain_ctx.fork(), current_start, first_block_start));
        if idx + 1 < tasks.len() {
            pad_context_to_page_boundary(&mut chain_ctx, pad_token_id).await;
        }
    }

    let futures = snapshots.into_iter().enumerate().map(|(idx, (task, mut ctx, current_start, first_start))| {
        let stop_text = paper_stop_text.to_vec();
        async move {
            let mut kv_optimization = KvOptimizationStats::default();
            if idx > 0 && current_start > first_start {
                let before_pages = ctx.kv_pages.len();
                ctx.mask_token_range(first_start, current_start, true);
                ctx.drop_masked_kv_pages();
                kv_optimization.question_mask_attempts += 1;
                kv_optimization.question_mask_tokens_masked +=
                    (current_start - first_start) as u64;
                kv_optimization.question_mask_pages_dropped +=
                    before_pages.saturating_sub(ctx.kv_pages.len()) as u64;
            }

            ctx.fill_token(continuation_token_id);
            let mut generated =
                generate_with_state_text_stop(ctx, model, max_tokens, temperature, &stop_text)
                    .await;
            let answer = generated.text.clone();
            maybe_compact_generated_tail(
                &mut generated.ctx,
                generated.prompt_token_len,
                &answer,
                cove_cfg,
                &mut kv_optimization,
            );
            (
                VerificationRecord {
                    candidate: task.candidate,
                    question: task.question,
                    answer,
                    verdict: VerificationVerdict::Ambiguous,
                },
                kv_optimization,
            )
        }
    });

    let mut kv_optimization = KvOptimizationStats::default();
    let records = join_all(futures)
        .await
        .into_iter()
        .map(|(record, optimization)| {
            kv_optimization.merge(&optimization);
            record
        })
        .collect();
    (records, kv_optimization)
}

async fn execute_verification_parallel_default(
    model: &Model,
    module_cache: &ModuleCache,
    search_client: &Client,
    search_resources: &SearchResources,
    tasks: &[VerificationTask],
    chain_type: ChainType,
    cove_cfg: &CoVeConfig,
    paper_variant: CoVeVariant,
    stop_sequences: &StopSequences,
) -> (Vec<VerificationRecord>, KvOptimizationStats) {
    if tasks.is_empty() {
        return (Vec::new(), KvOptimizationStats::default());
    }

    let use_search = effective_use_search(cove_cfg);
    let max_search_results = cove_cfg.max_search_results;
    let search_timeout = cove_cfg.search_timeout;
    let max_tokens = effective_verification_tokens(cove_cfg, chain_type);
    let temperature = effective_phase_temperature(cove_cfg.temperature_verification, cove_cfg);
    let paper_faithful = cove_cfg.paper_faithful;
    let max_wiki_items = cove_cfg.max_wiki_items;
    let chain_type_name = chain_type.as_str().to_string();

    if paper_faithful && paper_variant == CoVeVariant::TwoStep {
        let questions = tasks
            .iter()
            .map(|task| task.question.clone())
            .collect::<Vec<_>>();
        let stop_text =
            paper_stop_sequences("two_step_execute", chain_type.as_str(), max_wiki_items);
        let ctx = module_cache
            .paper
            .two_step_execute_prompt(chain_type)
            .fork_with_tail(&format!("{}\nResponse:", questions.join("\n")));
        let mut generated = generate_with_state_text_stop(
            ctx,
            model,
            paper_batch_tokens(cove_cfg, chain_type),
            temperature,
            &stop_text,
        )
        .await;
        let answers = parse_numbered_answers(&generated.text, tasks.len());
        let canonical = format_numbered_answer_list(&answers);
        let mut kv_optimization = KvOptimizationStats::default();
        if !canonical.is_empty() {
            maybe_compact_generated_tail(
                &mut generated.ctx,
                generated.prompt_token_len,
                &canonical,
                cove_cfg,
                &mut kv_optimization,
            );
        }
        return (
            tasks
            .iter()
            .cloned()
            .zip(answers.into_iter())
            .map(|(task, answer)| VerificationRecord {
                candidate: task.candidate,
                question: task.question,
                answer,
                verdict: VerificationVerdict::Ambiguous,
            })
            .collect(),
            kv_optimization,
        );
    }

    let paper_stop_text = if paper_faithful {
        paper_stop_sequences("execute", &chain_type_name, max_wiki_items)
    } else {
        Vec::new()
    };
    if paper_faithful {
        return match effective_paper_execute_strategy(cove_cfg) {
            PaperExecuteStrategy::FatPrompt => {
                execute_verification_paper_fat_prompt(
                    model,
                    tasks,
                    chain_type,
                    cove_cfg,
                    max_tokens,
                    temperature,
                    &paper_stop_text,
                )
                .await
            }
            PaperExecuteStrategy::MaskedChain => {
                execute_verification_paper_masked_chain(
                    model,
                    module_cache,
                    tasks,
                    chain_type,
                    cove_cfg,
                    max_tokens,
                    temperature,
                    &paper_stop_text,
                )
                .await
            }
            PaperExecuteStrategy::SharedRoot
            | PaperExecuteStrategy::SharedRootText
            | PaperExecuteStrategy::Auto => {
                execute_verification_paper_shared_root(
                    model,
                    module_cache,
                    tasks,
                    chain_type,
                    cove_cfg,
                    max_tokens,
                    temperature,
                    &paper_stop_text,
                )
                .await
            }
        };
    }

    let futures = tasks.iter().map(|task| {
        let task = task.clone();
        let search_client = search_client;
        let search_resources = search_resources;
        let module_cache = module_cache;
        let default_stop_tokens = stop_sequences.default_tokens().to_vec();
        let default_stop_text = stop_sequences.default_text().to_vec();

        async move {
            let search_context = if use_search {
                match search_resources
                    .search_ddg(
                        search_client,
                        &task.question,
                        max_search_results,
                        search_timeout,
                    )
                    .await
                {
                    Ok(result) => result,
                    Err(error) => {
                        if search_resources.mode() == SearchMode::Replay {
                            panic!("Frozen search failed for '{}': {}", task.question, error);
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
                    module_cache
                        .execute_search_default
                        .suffix()
                        .replace("{verification_question}", &task.question)
                );
                module_cache.execute_search_default.fork_with_tail(&tail)
            } else {
                module_cache
                    .execute_self_default
                    .fork_with_suffix(&task.question)
            };

            let answer = generate_from_context(
                ctx,
                max_tokens,
                temperature,
                &default_stop_tokens,
                &default_stop_text,
            )
            .await;
            let verdict = parse_yes_no_verdict(&answer);

            (
                VerificationRecord {
                    candidate: task.candidate,
                    question: task.question,
                    answer,
                    verdict,
                },
                KvOptimizationStats::default(),
            )
        }
    });

    let mut kv_optimization = KvOptimizationStats::default();
    let records = join_all(futures)
        .await
        .into_iter()
        .map(|(record, optimization)| {
            kv_optimization.merge(&optimization);
            record
        })
        .collect();

    (records, kv_optimization)
}

async fn generate_final_answer(
    model: &Model,
    module_cache: &ModuleCache,
    question: &str,
    baseline: &str,
    verification_records: &[VerificationRecord],
    cross_check_results: &[(String, String)],
    chain_type: ChainType,
    cove_cfg: &CoVeConfig,
    paper_variant: CoVeVariant,
    stop_sequences: &StopSequences,
    structured_wiki_plan: bool,
) -> (String, KvOptimizationStats) {
    if cove_cfg.paper_faithful {
        return generate_final_answer_generative(
            model,
            module_cache,
            question,
            baseline,
            verification_records,
            cross_check_results,
            chain_type,
            cove_cfg,
            paper_variant,
            stop_sequences,
        )
        .await;
    }

    if chain_type == ChainType::Wiki {
        if structured_wiki_plan {
            if let Some(answer) =
                build_structured_wiki_final_answer(verification_records, cove_cfg.max_wiki_items)
            {
                return (answer, KvOptimizationStats::default());
            }
        }
    }

    generate_final_answer_generative(
        model,
        module_cache,
        question,
        baseline,
        verification_records,
        cross_check_results,
        chain_type,
        cove_cfg,
        paper_variant,
        stop_sequences,
    )
    .await
}

async fn generate_final_answer_generative(
    model: &Model,
    module_cache: &ModuleCache,
    question: &str,
    baseline: &str,
    verification_records: &[VerificationRecord],
    cross_check_results: &[(String, String)],
    chain_type: ChainType,
    cove_cfg: &CoVeConfig,
    paper_variant: CoVeVariant,
    stop_sequences: &StopSequences,
) -> (String, KvOptimizationStats) {
    if cove_cfg.paper_faithful {
        let _ = paper_variant;
        let _ = cross_check_results;
        let verification_pairs = verification_records
            .iter()
            .map(|record| (record.question.clone(), record.answer.clone()))
            .collect::<Vec<_>>();
        let verification_text = verification_pairs
            .iter()
            .map(|(question, answer)| format!("Question: {}\nAnswer: {}", question, answer))
            .collect::<Vec<_>>()
            .join("\n");
        let ctx = module_cache
            .paper
            .final_prompt(chain_type)
            .fork_with_tail(&format!(
                "{}\nA: {}\nFrom another source,\n{}\nResponse:",
                question, baseline, verification_text
            ));
        let stop_text = paper_stop_sequences("final", chain_type.as_str(), cove_cfg.max_wiki_items);
        let stop_tokens = stop_tokens_for_text(model, &stop_text);
        return (
            generate_from_context(
                ctx,
                effective_final_tokens(cove_cfg, chain_type),
                effective_phase_temperature(cove_cfg.temperature_final, cove_cfg),
                &stop_tokens,
                &stop_text,
            )
            .await,
            KvOptimizationStats::default(),
        );
    }

    let verification_text = verification_records
        .iter()
        .map(|record| format!("Question: {}\nAnswer: {}", record.question, record.answer))
        .collect::<Vec<_>>()
        .join("\n");

    let tail = format!(
        "{}{}",
        question,
        module_cache
            .final_prompt(chain_type)
            .suffix()
            .replace("{baseline_response}", baseline)
            .replace("{verification_answers}", &verification_text)
    );
    let ctx = module_cache.final_prompt(chain_type).fork_with_tail(&tail);
    let (stop_tokens, stop_text) = wiki_list_stop_set(chain_type, stop_sequences);

    if chain_type != ChainType::Wiki {
        return (
            generate_from_context(
                ctx,
                effective_final_tokens(cove_cfg, chain_type),
                cove_cfg.temperature_final,
                stop_tokens,
                stop_text,
            )
            .await,
            KvOptimizationStats::default(),
        );
    }

    let mut kv_optimization = KvOptimizationStats::default();
    let mut generated = generate_with_state(
        ctx,
        effective_final_tokens(cove_cfg, ChainType::Wiki),
        cove_cfg.temperature_final,
        stop_tokens,
        stop_text,
    )
    .await;
    let candidates = normalize_wiki_candidates(&generated.text, cove_cfg.max_wiki_items);
    let normalized = format_numbered_list(&candidates);
    if !normalized.is_empty() {
        maybe_compact_generated_tail(
            &mut generated.ctx,
            generated.prompt_token_len,
            &normalized,
            cove_cfg,
            &mut kv_optimization,
        );
        (normalized, kv_optimization)
    } else {
        (generated.text, kv_optimization)
    }
}

async fn cross_check_passage(
    model: &Model,
    module_cache: &ModuleCache,
    original_fact: &str,
    verification_records: &[VerificationRecord],
    cove_cfg: &CoVeConfig,
) -> (String, KvOptimizationStats) {
    if verification_records.is_empty() {
        return ("INCONSISTENT.".to_string(), KvOptimizationStats::default());
    }

    let verification_blocks = verification_records
        .iter()
        .map(|record| format!("Question: {}\nAnswer: {}", record.question, record.answer))
        .collect::<Vec<_>>();
    let verification_text = verification_blocks.join("\n");
    let mut kv_optimization = KvOptimizationStats::default();
    let ctx = if cove_cfg.kv_optimization_mode == KvOptimizationMode::Windowed {
        let mut ctx = module_cache
            .paper
            .cross_check_long
            .fork_with_tail(&format!("{}\nFrom another source,\n", original_fact));
        let evidence_start = ctx.get_token_ids().len() + ctx.token_ids_pending.len();
        for (idx, block) in verification_blocks.iter().enumerate() {
            if idx > 0 {
                ctx.fill("\n");
            }
            ctx.fill(block);
            ctx.flush().await;
            maybe_window_evict_evidence(&mut ctx, evidence_start, cove_cfg, &mut kv_optimization);
        }
        ctx.fill("\nResponse:");
        ctx
    } else {
        module_cache.paper.cross_check_long.fork_with_tail(&format!(
            "{}\nFrom another source,\n{}\nResponse:",
            original_fact, verification_text
        ))
    };
    let stop_text = paper_stop_sequences("cross_check", "LONG_CHAIN", cove_cfg.max_wiki_items);
    let mut generated = generate_with_state_text_stop(
        ctx,
        model,
        effective_final_tokens(cove_cfg, ChainType::Long),
        effective_phase_temperature(cove_cfg.temperature_final, cove_cfg),
        &stop_text,
    )
    .await;
    let normalized = normalize_crosscheck_response(&generated.text);
    maybe_compact_generated_tail(
        &mut generated.ctx,
        generated.prompt_token_len,
        &normalized,
        cove_cfg,
        &mut kv_optimization,
    );
    (normalized, kv_optimization)
}

async fn revise_passage(cross_check_response: &str) -> String {
    let Some(consistent_body) = extract_consistent_crosscheck_body(cross_check_response) else {
        return String::new();
    };
    splice_consistent_body(&consistent_body)
}

async fn run_factor_revise_passage(
    idx: usize,
    model: &Model,
    module_cache: &ModuleCache,
    search_client: &Client,
    search_resources: &SearchResources,
    question: &str,
    passage: String,
    chain_type: ChainType,
    cove_cfg: &CoVeConfig,
    stop_sequences: &StopSequences,
) -> PassageFactorReviseResult {
    let vq_start = Instant::now();
    let tasks = generate_default_verification_tasks(
        model,
        module_cache,
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
    let (records, mut kv_optimization) = execute_verification_parallel_default(
        model,
        module_cache,
        search_client,
        search_resources,
        &tasks,
        chain_type,
        cove_cfg,
        CoVeVariant::Factored,
        stop_sequences,
    )
    .await;
    let execute_verification_ms = exec_start.elapsed().as_millis();

    let cross_start = Instant::now();
    let (cross_check, cross_check_optimization) =
        cross_check_passage(model, module_cache, &passage, &records, cove_cfg).await;
    kv_optimization.merge(&cross_check_optimization);
    let cross_check_ms = cross_start.elapsed().as_millis();

    let revise_start = Instant::now();
    let revised_passage = revise_passage(&cross_check).await;
    let final_answer_ms = revise_start.elapsed().as_millis();

    PassageFactorReviseResult {
        idx,
        passage,
        verification_tasks: tasks,
        verification_records: records,
        cross_check,
        revised_passage,
        verification_questions_ms,
        execute_verification_ms,
        cross_check_ms,
        final_answer_ms,
        kv_optimization,
    }
}

async fn run_factor_revise(
    model: &Model,
    module_cache: &ModuleCache,
    search_client: &Client,
    search_resources: &SearchResources,
    question: &str,
    baseline: &str,
    chain_type: ChainType,
    cove_cfg: &CoVeConfig,
    stop_sequences: &StopSequences,
) -> FactorReviseResult {
    let passages = split_passages(baseline);
    let mut verification_tasks = Vec::new();
    let mut verification_records = Vec::new();
    let mut cross_checks = Vec::new();
    let mut revised_passages = Vec::new();
    let mut verification_questions_ms = 0u128;
    let mut execute_verification_ms = 0u128;
    let mut cross_check_ms = 0u128;
    let mut final_answer_ms = 0u128;
    let mut kv_optimization = KvOptimizationStats::default();

    let total_passages = passages.len();
    let max_in_flight_passages =
        resolve_factor_revise_passage_concurrency(cove_cfg, total_passages, true);
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
                module_cache,
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
        kv_optimization.merge(&result.kv_optimization);
        verification_tasks.extend(result.verification_tasks);
        verification_records.extend(result.verification_records);
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
        verification_tasks,
        verification_records,
        cross_checks,
        final_answer,
        verification_questions_ms,
        execute_verification_ms,
        cross_check_ms,
        final_answer_ms,
        kv_optimization,
    }
}

// ============================================================================
// Helpers
// ============================================================================

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
    ctx: Context,
    max_tokens: usize,
    temperature: f32,
    stop_tokens: &[Vec<u32>],
    stop_text: &[String],
) -> String {
    let stop = max_len(max_tokens.max(1)).or(ends_with_any(stop_tokens.to_vec()));
    let raw_text = {
        let mut ctx = ctx;
        ctx.generate(make_sampler(temperature), stop).await
    };
    trim_stop_suffixes(raw_text, stop_text)
}

async fn generate_with_state_text_stop(
    mut ctx: Context,
    model: &Model,
    max_tokens: usize,
    temperature: f32,
    stop_text: &[String],
) -> GeneratedOutput {
    let prompt_token_len = ctx.get_token_ids().len() + ctx.token_ids_pending.len();
    let tokenizer = model.get_tokenizer();
    let sampler = make_sampler(temperature);
    let mut generated_token_ids = Vec::new();

    for _ in 0..max_tokens.max(1) {
        let next_token_id = ctx.decode_step(&sampler).await;
        ctx.fill_token(next_token_id);
        generated_token_ids.push(next_token_id);

        let raw_text = tokenizer.detokenize(&generated_token_ids);
        if matched_stop_suffix(&raw_text, stop_text).is_some() {
            let text = trim_stop_suffixes(raw_text, stop_text);
            ctx.flush().await;
            return GeneratedOutput {
                text,
                ctx,
                prompt_token_len,
            };
        }
    }

    let raw_text = tokenizer.detokenize(&generated_token_ids);
    let text = trim_stop_suffixes(raw_text, stop_text);
    ctx.flush().await;
    GeneratedOutput {
        text,
        ctx,
        prompt_token_len,
    }
}

async fn generate_from_context_text_stop_trace(
    mut ctx: Context,
    model: &Model,
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

async fn generate_with_state(
    mut ctx: Context,
    max_tokens: usize,
    temperature: f32,
    stop_tokens: &[Vec<u32>],
    stop_text: &[String],
) -> GeneratedOutput {
    let prompt_token_len = ctx.get_token_ids().len() + ctx.token_ids_pending.len();
    let stop = max_len(max_tokens.max(1)).or(ends_with_any(stop_tokens.to_vec()));
    let raw_text = ctx.generate(make_sampler(temperature), stop).await;
    ctx.flush().await;
    let text = trim_stop_suffixes(raw_text, stop_text);
    GeneratedOutput {
        text,
        ctx,
        prompt_token_len,
    }
}

fn make_sampler(temperature: f32) -> Sampler {
    if temperature <= 0.0 {
        Sampler::greedy()
    } else {
        Sampler::top_p(temperature, 1.0)
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

fn align_up(value: usize, alignment: usize) -> usize {
    if alignment == 0 {
        value
    } else {
        value.div_ceil(alignment) * alignment
    }
}

fn align_down(value: usize, alignment: usize) -> usize {
    if alignment == 0 {
        value
    } else {
        (value / alignment) * alignment
    }
}

fn apply_masked_page_drop(
    ctx: &mut Context,
    mask_start: usize,
    mask_end: usize,
) -> Option<(u64, u64)> {
    if mask_end <= mask_start {
        return None;
    }

    let before_pages = ctx.kv_pages.len();
    ctx.mask_token_range(mask_start, mask_end, true);
    ctx.drop_masked_kv_pages();
    let dropped_pages = before_pages.saturating_sub(ctx.kv_pages.len()) as u64;
    if dropped_pages == 0 {
        None
    } else {
        Some(((mask_end - mask_start) as u64, dropped_pages))
    }
}

fn compute_page_aligned_trim_mask_range(
    prompt_token_len: usize,
    total_generated_tokens: usize,
    kept_generated_tokens: usize,
    kv_page_size: usize,
    min_tokens: usize,
) -> Option<(usize, usize)> {
    if kept_generated_tokens >= total_generated_tokens || kv_page_size == 0 {
        return None;
    }

    let total_committed = prompt_token_len + total_generated_tokens;
    let mask_start = align_up(prompt_token_len + kept_generated_tokens, kv_page_size);
    let mask_end = align_down(total_committed, kv_page_size);
    if mask_end <= mask_start {
        return None;
    }

    let masked_tokens = mask_end - mask_start;
    if masked_tokens < min_tokens.max(1) {
        return None;
    }

    Some((mask_start, mask_end))
}

fn maybe_compact_generated_tail(
    ctx: &mut Context,
    prompt_token_len: usize,
    kept_text: &str,
    cove_cfg: &CoVeConfig,
    stats: &mut KvOptimizationStats,
) {
    if matches!(cove_cfg.kv_optimization_mode, KvOptimizationMode::Off) {
        return;
    }

    let total_committed = ctx.get_token_ids().len();
    let total_generated_tokens = total_committed.saturating_sub(prompt_token_len);
    let kept_generated_tokens = ctx.model().get_tokenizer().tokenize(kept_text).len();
    if let Some((mask_start, mask_end)) = compute_page_aligned_trim_mask_range(
        prompt_token_len,
        total_generated_tokens,
        kept_generated_tokens,
        ctx.kv_page_size,
        cove_cfg.kv_compaction_min_tokens,
    ) {
        if let Some((masked_tokens, dropped_pages)) = apply_masked_page_drop(ctx, mask_start, mask_end)
        {
            stats.tail_compactions += 1;
            stats.tail_tokens_masked += masked_tokens;
            stats.tail_pages_dropped += dropped_pages;
        }
    }
}

fn compute_windowed_evidence_mask_range(
    evidence_start: usize,
    total_tokens: usize,
    kv_page_size: usize,
    sink_tokens: usize,
    keep_recent_tokens: usize,
    min_tokens: usize,
) -> Option<(usize, usize)> {
    if kv_page_size == 0 || total_tokens <= evidence_start {
        return None;
    }

    let protected_prefix = evidence_start.max(sink_tokens);
    let mask_start = align_up(protected_prefix, kv_page_size);
    let mask_end = align_down(total_tokens.saturating_sub(keep_recent_tokens), kv_page_size);
    if mask_end <= mask_start {
        return None;
    }

    let masked_tokens = mask_end - mask_start;
    if masked_tokens < min_tokens.max(1) {
        return None;
    }

    Some((mask_start, mask_end))
}

fn maybe_window_evict_evidence(
    ctx: &mut Context,
    evidence_start: usize,
    cove_cfg: &CoVeConfig,
    stats: &mut KvOptimizationStats,
) {
    if cove_cfg.kv_optimization_mode != KvOptimizationMode::Windowed {
        return;
    }

    let total_tokens = ctx.get_token_ids().len();
    if let Some((mask_start, mask_end)) = compute_windowed_evidence_mask_range(
        evidence_start,
        total_tokens,
        ctx.kv_page_size,
        cove_cfg.kv_window_sink_tokens,
        cove_cfg.kv_window_recent_tokens,
        cove_cfg.kv_compaction_min_tokens,
    ) {
        if let Some((masked_tokens, dropped_pages)) = apply_masked_page_drop(ctx, mask_start, mask_end)
        {
            stats.window_evictions += 1;
            stats.window_tokens_masked += masked_tokens;
            stats.window_pages_dropped += dropped_pages;
        }
    }
}

fn effective_phase_temperature(temperature: f32, cove_cfg: &CoVeConfig) -> f32 {
    if cove_cfg.paper_faithful {
        0.0
    } else {
        temperature
    }
}

fn normalize_wiki_candidates(text: &str, max_items: usize) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut candidates = Vec::new();

    for line in text.lines() {
        let cleaned = canonicalize_wiki_candidate(line);
        if cleaned.is_empty() {
            continue;
        }

        let key = canonical_key(&cleaned);
        if key.is_empty() || !seen.insert(key) {
            continue;
        }

        candidates.push(cleaned);
        if candidates.len() >= max_items.max(1) {
            break;
        }
    }

    candidates
}

fn canonicalize_wiki_candidate(line: &str) -> String {
    let mut cleaned = strip_number_prefix(line);
    if cleaned.is_empty() {
        return String::new();
    }

    for sep in [" - ", " – ", " — "] {
        if let Some((head, _)) = cleaned.split_once(sep) {
            cleaned = head.trim().to_string();
        }
    }

    if let Some((head, tail)) = cleaned.split_once(": ") {
        if head.split_whitespace().count() <= 8 && tail.split_whitespace().count() > 2 {
            cleaned = head.trim().to_string();
        }
    }

    cleaned = cleaned
        .trim_matches(|ch: char| {
            ch.is_whitespace() || matches!(ch, '"' | '\'' | '`' | '[' | ']' | '(' | ')' | '*' | '-')
        })
        .trim_end_matches(|ch: char| matches!(ch, '.' | ';' | ','))
        .to_string();

    cleaned
}

fn format_numbered_list(items: &[String]) -> String {
    items
        .iter()
        .enumerate()
        .map(|(idx, item)| format!("{}. {}", idx + 1, item))
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_numbered_answer_list(items: &[String]) -> String {
    items
        .iter()
        .enumerate()
        .map(|(idx, item)| format!("{}. {}", idx + 1, item.trim()))
        .collect::<Vec<_>>()
        .join("\n")
}

fn template_contains_answer_placeholder(template: &str) -> bool {
    let count = template.matches("[ANSWER_ENTITY]").count();
    count == 1
}

fn instantiate_wiki_verification_question(template: &str, candidate: &str) -> Option<String> {
    if !template_contains_answer_placeholder(template) {
        return None;
    }

    let question = template
        .replace("[ANSWER_ENTITY]", candidate.trim())
        .trim()
        .to_string();
    if question.is_empty() {
        None
    } else {
        Some(compact_whitespace(&question))
    }
}

fn build_structured_wiki_final_answer(
    verification_records: &[VerificationRecord],
    max_items: usize,
) -> Option<String> {
    let mut positives = Vec::new();
    let mut seen = BTreeSet::new();
    let mut saw_candidate = false;
    let mut saw_ambiguous = false;

    for record in verification_records {
        let Some(candidate) = record.candidate.as_ref() else {
            saw_ambiguous = true;
            continue;
        };
        saw_candidate = true;
        match record.verdict {
            VerificationVerdict::Yes => {
                let key = canonical_key(candidate);
                if !key.is_empty() && seen.insert(key) {
                    positives.push(candidate.clone());
                }
            }
            VerificationVerdict::No => {}
            VerificationVerdict::Ambiguous => saw_ambiguous = true,
        }
    }

    if positives.is_empty() {
        if saw_candidate && !saw_ambiguous {
            return None;
        }
        return None;
    }

    positives.truncate(max_items.max(1));
    Some(format_numbered_list(&positives))
}

fn parse_yes_no_verdict(text: &str) -> VerificationVerdict {
    let first_word = text
        .trim()
        .trim_start_matches(|ch: char| !ch.is_ascii_alphabetic())
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_matches(|ch: char| !ch.is_ascii_alphabetic())
        .to_ascii_uppercase();

    if first_word.starts_with("YES") {
        VerificationVerdict::Yes
    } else if first_word.starts_with("NO") {
        VerificationVerdict::No
    } else {
        VerificationVerdict::Ambiguous
    }
}

fn canonicalize_verdict_text(raw_answer: &str, verdict: VerificationVerdict) -> String {
    match verdict {
        VerificationVerdict::Yes => "YES".to_string(),
        VerificationVerdict::No => "NO".to_string(),
        VerificationVerdict::Ambiguous => raw_answer.trim().to_string(),
    }
}

fn strip_number_prefix(line: &str) -> String {
    let trimmed = line.trim();
    let mut prefix_end = 0usize;
    for (idx, ch) in trimmed.char_indices() {
        if ch.is_ascii_digit() {
            prefix_end = idx + ch.len_utf8();
        } else {
            break;
        }
    }

    if prefix_end == 0 {
        return trimmed.to_string();
    }

    let mut rest = &trimmed[prefix_end..];
    if let Some(ch) = rest.chars().next() {
        if ch == '.' || ch == '\\' || ch == ')' {
            rest = &rest[ch.len_utf8()..];
        } else {
            return trimmed.to_string();
        }
    } else {
        return trimmed.to_string();
    }

    rest.trim_start().to_string()
}

fn canonical_key(text: &str) -> String {
    text.to_ascii_lowercase()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch.is_ascii_whitespace() {
                ch
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn compact_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn build_paper_baseline_prefix(chain_type: &str, few_shot_examples: usize) -> String {
    paper_prompt_prefix(&build_paper_baseline_prompt(
        PAPER_DYNAMIC_MARKER,
        chain_type,
        few_shot_examples,
    ))
}

fn build_paper_verification_prefix(chain_type: &str, few_shot_examples: usize) -> String {
    paper_prompt_prefix(&build_paper_verification_prompt(
        PAPER_DYNAMIC_MARKER,
        "baseline",
        chain_type,
        few_shot_examples,
    ))
}

fn build_paper_execute_prefix(chain_type: &str, few_shot_examples: usize) -> String {
    paper_prompt_prefix(&build_paper_execute_prompt(
        PAPER_DYNAMIC_MARKER,
        chain_type,
        few_shot_examples,
    ))
}

fn build_paper_two_step_execute_prefix(chain_type: &str, few_shot_examples: usize) -> String {
    paper_prompt_prefix(&build_paper_two_step_execute_prompt(
        &[PAPER_DYNAMIC_MARKER.to_string()],
        chain_type,
        few_shot_examples,
    ))
}

fn build_paper_joint_prefix(chain_type: &str, few_shot_examples: usize) -> String {
    paper_prompt_prefix(&build_paper_joint_prompt(
        PAPER_DYNAMIC_MARKER,
        "baseline",
        chain_type,
        few_shot_examples,
    ))
}

fn build_paper_final_prefix(chain_type: &str, few_shot_examples: usize) -> String {
    paper_prompt_prefix(&build_paper_final_prompt(
        PAPER_DYNAMIC_MARKER,
        "baseline",
        &[],
        chain_type,
        few_shot_examples,
    ))
}

fn build_paper_cross_check_prefix(few_shot_examples: usize) -> String {
    paper_prompt_prefix(&build_paper_crosscheck_prompt(
        PAPER_DYNAMIC_MARKER,
        &[],
        few_shot_examples,
    ))
}

fn paper_prompt_prefix(prompt: &str) -> String {
    prompt
        .split_once(PAPER_DYNAMIC_MARKER)
        .map(|(prefix, _)| prefix.to_string())
        .unwrap_or_else(|| {
            panic!(
                "Paper prompt missing dynamic marker '{}': {}",
                PAPER_DYNAMIC_MARKER, prompt
            )
        })
}

fn make_module_cache_key(
    model_name: &str,
    logical_name: &str,
    prompt_prefix: &str,
    prompt_suffix: &str,
) -> String {
    let mut hasher = DefaultHasher::new();
    MODULE_CACHE_VERSION.hash(&mut hasher);
    model_name.hash(&mut hasher);
    logical_name.hash(&mut hasher);
    prompt_prefix.hash(&mut hasher);
    prompt_suffix.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn make_module_export_name(logical_name: &str, cache_key: &str) -> String {
    format!(
        "{}.module.{}.{}",
        MODULE_CACHE_VERSION, logical_name, cache_key
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_wiki_candidates_trims_noise_and_caps_items() {
        let text = "1. Matt Damon - Academy Award winner\n2. Chris Evans\n3. Matt Damon\n4. Ben Affleck: actor and writer";
        let candidates = normalize_wiki_candidates(text, 3);
        assert_eq!(
            candidates,
            vec![
                "Matt Damon".to_string(),
                "Chris Evans".to_string(),
                "Ben Affleck".to_string()
            ]
        );
    }

    #[test]
    fn template_instantiation_requires_answer_entity_placeholder() {
        let template = "Was [ANSWER_ENTITY] born in Boston?";
        assert!(template_contains_answer_placeholder(template));
        assert_eq!(
            instantiate_wiki_verification_question(template, "Matt Damon").as_deref(),
            Some("Was Matt Damon born in Boston?")
        );
        assert!(
            instantiate_wiki_verification_question("Was he born in Boston?", "Matt Damon")
                .is_none()
        );
    }

    #[test]
    fn verdict_parser_handles_yes_no_and_ambiguous() {
        assert_eq!(parse_yes_no_verdict("YES"), VerificationVerdict::Yes);
        assert_eq!(parse_yes_no_verdict("No."), VerificationVerdict::No);
        assert_eq!(
            parse_yes_no_verdict("It appears so"),
            VerificationVerdict::Ambiguous
        );
    }

    #[test]
    fn page_aligned_trim_mask_range_only_drops_full_pages() {
        assert_eq!(
            compute_page_aligned_trim_mask_range(120, 12, 12, 16, 1),
            None
        );
        assert_eq!(
            compute_page_aligned_trim_mask_range(120, 12, 5, 16, 1),
            None
        );
        assert_eq!(
            compute_page_aligned_trim_mask_range(120, 40, 5, 16, 1),
            Some((128, 160))
        );
    }

    #[test]
    fn windowed_mask_range_preserves_sink_and_recent_tail() {
        assert_eq!(
            compute_windowed_evidence_mask_range(96, 224, 16, 128, 48, 1),
            Some((128, 176))
        );
    }

    #[test]
    fn windowed_mask_range_skips_partial_pages() {
        assert_eq!(
            compute_windowed_evidence_mask_range(98, 170, 16, 98, 49, 1),
            None
        );
    }

    #[test]
    fn module_cache_key_changes_when_prompt_changes() {
        let key_a = make_module_cache_key("model-a", "router", ROUTER_PROMPT, "{question}");
        let key_b = make_module_cache_key("model-a", "router", ROUTER_PROMPT, "{question}");
        let key_c = make_module_cache_key("model-a", "router", BASELINE_PROMPT_MULTI, "");
        assert_eq!(key_a, key_b);
        assert_ne!(key_a, key_c);
    }

    #[test]
    fn structured_wiki_final_answer_preserves_positive_order() {
        let records = vec![
            VerificationRecord {
                candidate: Some("Matt Damon".to_string()),
                question: "Was Matt Damon born in Boston?".to_string(),
                answer: "YES".to_string(),
                verdict: VerificationVerdict::Yes,
            },
            VerificationRecord {
                candidate: Some("Chris Evans".to_string()),
                question: "Was Chris Evans born in Boston?".to_string(),
                answer: "NO".to_string(),
                verdict: VerificationVerdict::No,
            },
            VerificationRecord {
                candidate: Some("Ben Affleck".to_string()),
                question: "Was Ben Affleck born in Boston?".to_string(),
                answer: "YES".to_string(),
                verdict: VerificationVerdict::Yes,
            },
        ];

        assert_eq!(
            build_structured_wiki_final_answer(&records, 5).as_deref(),
            Some("1. Matt Damon\n2. Ben Affleck")
        );
    }
}
