//! Utilities for Chain-of-Verification inferlet.
//!
//! Includes config structures, shared search-cache support, and evaluation metrics.

use futures::{
    future::{select, Either},
    pin_mut,
};
use inferlet::wstd::http::{Client, Method, Request};
use inferlet::wstd::io::{empty, AsyncRead};
use inferlet::wstd::time::Duration;
use inferlet::Result;
use rand::{Error, RngCore};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

// ============================================================================
// Configuration
// ============================================================================

fn default_max_tokens_routing() -> usize {
    8
}

fn default_question_concurrency() -> usize {
    1
}

fn default_search_mode() -> SearchMode {
    SearchMode::Live
}

fn default_metric_name() -> String {
    "auto".to_string()
}

fn default_max_wiki_items() -> usize {
    5
}

fn default_max_tokens_wiki_baseline() -> usize {
    48
}

fn default_max_tokens_wiki_verification() -> usize {
    48
}

fn default_max_tokens_wiki_final() -> usize {
    64
}

fn default_paper_faithful() -> bool {
    true
}

fn default_few_shot_examples() -> usize {
    3
}

fn default_paper_task() -> PaperTask {
    PaperTask::Auto
}

fn default_cove_variant() -> CoVeVariant {
    CoVeVariant::Auto
}

fn default_paper_execute_strategy() -> PaperExecuteStrategy {
    PaperExecuteStrategy::Auto
}

fn default_factor_revise_passage_mode() -> FactorRevisePassageMode {
    FactorRevisePassageMode::Auto
}

fn default_kv_optimization_mode() -> KvOptimizationMode {
    KvOptimizationMode::Off
}

fn default_kv_compaction_min_tokens() -> usize {
    16
}

fn default_kv_window_sink_tokens() -> usize {
    256
}

fn default_kv_window_recent_tokens() -> usize {
    256
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    Live,
    Record,
    Replay,
}

impl SearchMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            SearchMode::Live => "live",
            SearchMode::Record => "record",
            SearchMode::Replay => "replay",
        }
    }
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PaperTask {
    Auto,
    Wikidata,
    WikiCategory,
    Multispanqa,
    Biography,
}

impl PaperTask {
    pub fn as_str(&self) -> &'static str {
        match self {
            PaperTask::Auto => "auto",
            PaperTask::Wikidata => "wikidata",
            PaperTask::WikiCategory => "wiki_category",
            PaperTask::Multispanqa => "multispanqa",
            PaperTask::Biography => "biography",
        }
    }
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CoVeVariant {
    Auto,
    Joint,
    TwoStep,
    Factored,
    FactorRevise,
}

impl CoVeVariant {
    pub fn as_str(&self) -> &'static str {
        match self {
            CoVeVariant::Auto => "auto",
            CoVeVariant::Joint => "joint",
            CoVeVariant::TwoStep => "two_step",
            CoVeVariant::Factored => "factored",
            CoVeVariant::FactorRevise => "factor_revise",
        }
    }
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PaperExecuteStrategy {
    Auto,
    FatPrompt,
    MaskedChain,
    MaskedChainNoDrop,
    SharedRoot,
    SharedRootText,
}

impl PaperExecuteStrategy {
    pub fn as_str(&self) -> &'static str {
        match self {
            PaperExecuteStrategy::Auto => "auto",
            PaperExecuteStrategy::FatPrompt => "fat_prompt",
            PaperExecuteStrategy::MaskedChain => "masked_chain",
            PaperExecuteStrategy::MaskedChainNoDrop => "masked_chain_no_drop",
            PaperExecuteStrategy::SharedRoot => "shared_root",
            PaperExecuteStrategy::SharedRootText => "shared_root_text",
        }
    }
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FactorRevisePassageMode {
    Auto,
    Serial,
    #[serde(alias = "concurrent")]
    Parallel,
}

impl FactorRevisePassageMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            FactorRevisePassageMode::Auto => "auto",
            FactorRevisePassageMode::Serial => "serial",
            FactorRevisePassageMode::Parallel => "parallel",
        }
    }
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KvOptimizationMode {
    Off,
    Tail,
    Windowed,
}

impl KvOptimizationMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            KvOptimizationMode::Off => "off",
            KvOptimizationMode::Tail => "tail",
            KvOptimizationMode::Windowed => "windowed",
        }
    }
}

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
    #[serde(default = "default_paper_faithful")]
    pub paper_faithful: bool,
    #[serde(default = "default_few_shot_examples")]
    pub few_shot_examples: usize,
    #[serde(default = "default_paper_task")]
    pub paper_task: PaperTask,
    #[serde(default = "default_cove_variant")]
    pub cove_variant: CoVeVariant,
    #[serde(default = "default_paper_execute_strategy")]
    pub paper_execute_strategy: PaperExecuteStrategy,
    pub use_search: bool,
    pub max_search_results: usize,
    pub search_timeout: f32,
    #[serde(default = "default_search_mode")]
    pub search_mode: SearchMode,
    #[serde(default)]
    pub search_cache_path: Option<String>,
    #[serde(default = "default_max_tokens_routing")]
    pub max_tokens_routing: usize,
    #[serde(default = "default_question_concurrency")]
    pub question_concurrency: usize,
    pub max_tokens_baseline: usize,
    pub max_tokens_verification: usize,
    pub max_tokens_final: usize,
    #[serde(default = "default_max_wiki_items")]
    pub max_wiki_items: usize,
    #[serde(default = "default_max_tokens_wiki_baseline")]
    pub max_tokens_wiki_baseline: usize,
    #[serde(default = "default_max_tokens_wiki_verification")]
    pub max_tokens_wiki_verification: usize,
    #[serde(default = "default_max_tokens_wiki_final")]
    pub max_tokens_wiki_final: usize,
    pub temperature_routing: f32,
    pub temperature_baseline: f32,
    pub temperature_verification: f32,
    pub temperature_final: f32,
    pub max_verification_questions: usize,
    #[serde(default = "default_factor_revise_passage_mode")]
    pub factor_revise_passage_mode: FactorRevisePassageMode,
    #[serde(default)]
    pub factor_revise_passage_concurrency: Option<usize>,
    #[serde(default = "default_kv_optimization_mode")]
    pub kv_optimization_mode: KvOptimizationMode,
    #[serde(default = "default_kv_compaction_min_tokens")]
    pub kv_compaction_min_tokens: usize,
    #[serde(default = "default_kv_window_sink_tokens")]
    pub kv_window_sink_tokens: usize,
    #[serde(default = "default_kv_window_recent_tokens")]
    pub kv_window_recent_tokens: usize,
}

#[derive(Deserialize, Clone, Debug)]
pub struct DataConfig {
    pub dataset: String,
    pub dataset_path: String,
    pub num_samples: usize,
    pub random_seed: u64,
    #[serde(default = "default_metric_name")]
    pub metric: String,
}

#[derive(Deserialize, Clone, Debug)]
pub struct RootConfig {
    pub generation: GenerationConfig,
    pub cove: CoVeConfig,
    pub data: DataConfig,
}

fn debug_path_state(label: &str, path: &str) {
    let candidate = Path::new(path);
    eprintln!(
        "[cove-debug] {} path='{}' exists={} is_file={} is_dir={}",
        label,
        path,
        candidate.exists(),
        candidate.is_file(),
        candidate.is_dir()
    );
}

pub fn default_single_question_config() -> RootConfig {
    RootConfig {
        generation: GenerationConfig {
            model_name: "llama-3.1-8b-instruct".to_string(),
            model_id: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
            tokenizer_id: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
            torch_dtype: "bfloat16".to_string(),
            max_new_tokens: 500,
            temperature: 0.1,
        },
        cove: CoVeConfig {
            paper_faithful: true,
            few_shot_examples: default_few_shot_examples(),
            paper_task: default_paper_task(),
            cove_variant: default_cove_variant(),
            paper_execute_strategy: default_paper_execute_strategy(),
            use_search: false,
            max_search_results: 3,
            search_timeout: 10.0,
            search_mode: SearchMode::Live,
            search_cache_path: None,
            max_tokens_routing: default_max_tokens_routing(),
            question_concurrency: default_question_concurrency(),
            max_tokens_baseline: 256,
            max_tokens_verification: 64,
            max_tokens_final: 512,
            max_wiki_items: default_max_wiki_items(),
            max_tokens_wiki_baseline: default_max_tokens_wiki_baseline(),
            max_tokens_wiki_verification: default_max_tokens_wiki_verification(),
            max_tokens_wiki_final: default_max_tokens_wiki_final(),
            temperature_routing: 0.0,
            temperature_baseline: 0.0,
            temperature_verification: 0.0,
            temperature_final: 0.0,
            max_verification_questions: 5,
            factor_revise_passage_mode: default_factor_revise_passage_mode(),
            factor_revise_passage_concurrency: None,
            kv_optimization_mode: default_kv_optimization_mode(),
            kv_compaction_min_tokens: default_kv_compaction_min_tokens(),
            kv_window_sink_tokens: default_kv_window_sink_tokens(),
            kv_window_recent_tokens: default_kv_window_recent_tokens(),
        },
        data: DataConfig {
            dataset: "trivia_qa".to_string(),
            dataset_path: String::new(),
            num_samples: 0,
            random_seed: 42,
            metric: default_metric_name(),
        },
    }
}

pub fn load_config(path_or_inline_toml: &str) -> RootConfig {
    if Path::new(path_or_inline_toml).exists() {
        debug_path_state("config", path_or_inline_toml);
        let content = fs::read_to_string(path_or_inline_toml).unwrap_or_else(|e| {
            panic!(
                "Failed to read config file '{}': {}",
                path_or_inline_toml, e
            )
        });
        toml::from_str(&content).unwrap_or_else(|e| {
            panic!(
                "Failed to parse config file '{}': {}",
                path_or_inline_toml, e
            )
        })
    } else {
        eprintln!("[cove-debug] using inline config TOML");
        toml::from_str(path_or_inline_toml)
            .unwrap_or_else(|e| panic!("Failed to parse inline config TOML: {}", e))
    }
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
            .map(|answers| answers.text.clone())
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
    debug_path_state("dataset", path);
    let content = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read dataset file '{}': {}", path, e));

    if let Ok(dataset) = serde_json::from_str::<DatasetFile>(&content) {
        return dataset;
    }

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
// Paper-faithful prompts and parsing
// ============================================================================

pub const PAPER_QA_STOP_SEQUENCES: [&str; 4] = ["\nQ:", "Q:", "\nA:", "A:"];
pub const PAPER_CONTEXT_STOP_SEQUENCES: [&str; 4] =
    ["\nContext:", "Context:", "\nResponse:", "Response:"];

#[derive(Clone, Copy)]
struct PaperExample {
    question: &'static str,
    answer: &'static str,
    baseline_answer: &'static str,
    verification_questions: &'static [&'static str],
    verification_answers: &'static [(&'static str, &'static str)],
    final_answer: &'static str,
}

#[derive(Clone, Copy)]
struct CrossCheckExample {
    original_fact: &'static str,
    verification_answers: &'static [(&'static str, &'static str)],
    response: &'static str,
}

#[derive(Clone, Copy)]
struct FactorReviseExample {
    original_passage: &'static str,
    verification_answers: &'static [(&'static str, &'static str)],
    cross_check_response: &'static str,
    final_answer: &'static str,
}

#[derive(Clone, Copy)]
struct LongPlanExample {
    question: &'static str,
    baseline_answer: &'static str,
    plan_lines: &'static [&'static str],
}

const WIKI_EXAMPLE_1_VERIFICATION_QUESTIONS: [&str; 3] = [
    "Where was Matt Damon born?",
    "Where was Chris Evans born?",
    "Where was Ben Affleck born?",
];
const WIKI_EXAMPLE_1_VERIFICATION_ANSWERS: [(&str, &str); 3] = [
    (
        "Where was Matt Damon born?",
        "Matt Damon was born in Cambridge, Massachusetts.",
    ),
    (
        "Where was Chris Evans born?",
        "Chris Evans was born in Boston, Massachusetts.",
    ),
    (
        "Where was Ben Affleck born?",
        "Ben Affleck was born in Berkeley, California.",
    ),
];
const WIKI_EXAMPLE_2_VERIFICATION_QUESTIONS: [&str; 3] = [
    "Where was Alicia Keys born?",
    "Where was Taylor Swift born?",
    "Where was Lady Gaga born?",
];
const WIKI_EXAMPLE_2_VERIFICATION_ANSWERS: [(&str, &str); 3] = [
    (
        "Where was Alicia Keys born?",
        "Alicia Keys was born in New York City, New York.",
    ),
    (
        "Where was Taylor Swift born?",
        "Taylor Swift was born in West Reading, Pennsylvania.",
    ),
    (
        "Where was Lady Gaga born?",
        "Lady Gaga was born in New York City, New York.",
    ),
];
const WIKI_EXAMPLE_3_VERIFICATION_QUESTIONS: [&str; 3] = [
    "Who won the Nobel Prize in Physics in 1903?",
    "In which year did Albert Einstein win the Nobel Prize?",
    "Did Henri Becquerel win a Nobel Prize in 1903?",
];
const WIKI_EXAMPLE_3_VERIFICATION_ANSWERS: [(&str, &str); 3] = [
    (
        "Who won the Nobel Prize in Physics in 1903?",
        "Henri Becquerel, Pierre Curie, and Marie Curie won the Nobel Prize in Physics in 1903.",
    ),
    (
        "In which year did Albert Einstein win the Nobel Prize?",
        "Albert Einstein won the Nobel Prize in Physics in 1921.",
    ),
    (
        "Did Henri Becquerel win a Nobel Prize in 1903?",
        "Yes. Henri Becquerel shared the 1903 Nobel Prize in Physics.",
    ),
];
const WIKI_PAPER_EXAMPLES: [PaperExample; 3] = [
    PaperExample {
        question: "Name some movie actors who were born in Boston.",
        answer: "1. Chris Evans\n2. Matt Damon",
        baseline_answer: "1. Matt Damon\n2. Chris Evans\n3. Ben Affleck",
        verification_questions: &WIKI_EXAMPLE_1_VERIFICATION_QUESTIONS,
        verification_answers: &WIKI_EXAMPLE_1_VERIFICATION_ANSWERS,
        final_answer: "1. Chris Evans\n2. Matt Damon",
    },
    PaperExample {
        question: "Name some singers who were born in New York City.",
        answer: "1. Alicia Keys\n2. Lady Gaga",
        baseline_answer: "1. Alicia Keys\n2. Taylor Swift\n3. Lady Gaga",
        verification_questions: &WIKI_EXAMPLE_2_VERIFICATION_QUESTIONS,
        verification_answers: &WIKI_EXAMPLE_2_VERIFICATION_ANSWERS,
        final_answer: "1. Alicia Keys\n2. Lady Gaga",
    },
    PaperExample {
        question: "Name some scientists who won the Nobel Prize in 1903.",
        answer: "1. Marie Curie\n2. Pierre Curie\n3. Henri Becquerel",
        baseline_answer: "1. Marie Curie\n2. Pierre Curie\n3. Albert Einstein",
        verification_questions: &WIKI_EXAMPLE_3_VERIFICATION_QUESTIONS,
        verification_answers: &WIKI_EXAMPLE_3_VERIFICATION_ANSWERS,
        final_answer: "1. Marie Curie\n2. Pierre Curie\n3. Henri Becquerel",
    },
];

const MULTI_EXAMPLE_1_VERIFICATION_QUESTIONS: [&str; 2] = [
    "Who invented the first printing press?",
    "In what year did Johannes Gutenberg invent the first printing press?",
];
const MULTI_EXAMPLE_1_VERIFICATION_ANSWERS: [(&str, &str); 2] = [
    (
        "Who invented the first printing press?",
        "Johannes Gutenberg is credited with inventing the first movable-type printing press in Europe.",
    ),
    (
        "In what year did Johannes Gutenberg invent the first printing press?",
        "Johannes Gutenberg developed the printing press around 1440.",
    ),
];
const MULTI_EXAMPLE_2_VERIFICATION_QUESTIONS: [&str; 2] = [
    "Who wrote \"I Can't Make You Love Me\"?",
    "Who recorded the hit version of \"I Can't Make You Love Me\"?",
];
const MULTI_EXAMPLE_2_VERIFICATION_ANSWERS: [(&str, &str); 2] = [
    (
        "Who wrote \"I Can't Make You Love Me\"?",
        "\"I Can't Make You Love Me\" was written by Mike Reid and Allen Shamblin.",
    ),
    (
        "Who recorded the hit version of \"I Can't Make You Love Me\"?",
        "Bonnie Raitt recorded the best-known hit version of \"I Can't Make You Love Me\".",
    ),
];
const MULTI_EXAMPLE_3_VERIFICATION_QUESTIONS: [&str; 1] = [
    "On which dates does the North American Butterfly Association count butterflies in the United States each year?",
];
const MULTI_EXAMPLE_3_VERIFICATION_ANSWERS: [(&str, &str); 1] = [(
    "On which dates does the North American Butterfly Association count butterflies in the United States each year?",
    "The official Fourth of July Butterfly Count runs from July 1 through July 4 each year.",
)];
const MULTI_PAPER_EXAMPLES: [PaperExample; 3] = [
    PaperExample {
        question: "Who invented the first printing press and in what year?",
        answer: "Johannes Gutenberg, around 1440.",
        baseline_answer: "Johannes Gutenberg, 1450.",
        verification_questions: &MULTI_EXAMPLE_1_VERIFICATION_QUESTIONS,
        verification_answers: &MULTI_EXAMPLE_1_VERIFICATION_ANSWERS,
        final_answer: "Johannes Gutenberg, around 1440.",
    },
    PaperExample {
        question: "Who wrote I Can't Make You Love Me?",
        answer: "Mike Reid and Allen Shamblin.",
        baseline_answer: "Bonnie Raitt.",
        verification_questions: &MULTI_EXAMPLE_2_VERIFICATION_QUESTIONS,
        verification_answers: &MULTI_EXAMPLE_2_VERIFICATION_ANSWERS,
        final_answer: "Mike Reid and Allen Shamblin.",
    },
    PaperExample {
        question: "When does the North American Butterfly Association count butterflies in the U.S. each year?",
        answer: "July 1 through July 4.",
        baseline_answer: "July.",
        verification_questions: &MULTI_EXAMPLE_3_VERIFICATION_QUESTIONS,
        verification_answers: &MULTI_EXAMPLE_3_VERIFICATION_ANSWERS,
        final_answer: "July 1 through July 4.",
    },
];

const LONG_EXAMPLE_1_VERIFICATION_QUESTIONS: [&str; 3] = [
    "Where was Ada Lovelace born?",
    "What was Ada Lovelace known for?",
    "Did Ada Lovelace build the Analytical Engine herself?",
];
const LONG_EXAMPLE_1_VERIFICATION_ANSWERS: [(&str, &str); 3] = [
    (
        "Where was Ada Lovelace born?",
        "Ada Lovelace was born in London, England.",
    ),
    (
        "What was Ada Lovelace known for?",
        "Ada Lovelace is known for her notes on Charles Babbage's Analytical Engine, including an algorithm for the machine.",
    ),
    (
        "Did Ada Lovelace build the Analytical Engine herself?",
        "No. Charles Babbage designed the Analytical Engine; Ada Lovelace wrote influential notes about it.",
    ),
];
const LONG_EXAMPLE_2_VERIFICATION_QUESTIONS: [&str; 3] = [
    "Where was Grace Hopper born?",
    "What is Grace Hopper known for in computing?",
    "Did Grace Hopper invent COBOL by herself?",
];
const LONG_EXAMPLE_2_VERIFICATION_ANSWERS: [(&str, &str); 3] = [
    (
        "Where was Grace Hopper born?",
        "Grace Hopper was born in New York City, New York.",
    ),
    (
        "What is Grace Hopper known for in computing?",
        "Grace Hopper helped develop early compilers and was a leading figure in the development of COBOL.",
    ),
    (
        "Did Grace Hopper invent COBOL by herself?",
        "No. Grace Hopper contributed to COBOL, but it was developed by a committee and broader team effort.",
    ),
];
const LONG_EXAMPLE_3_VERIFICATION_QUESTIONS: [&str; 3] = [
    "What field is Marie Curie known for?",
    "How many Nobel Prizes did Marie Curie win?",
    "What discoveries is Marie Curie associated with?",
];
const LONG_EXAMPLE_3_VERIFICATION_ANSWERS: [(&str, &str); 3] = [
    (
        "What field is Marie Curie known for?",
        "Marie Curie is known for pioneering research on radioactivity in physics and chemistry.",
    ),
    (
        "How many Nobel Prizes did Marie Curie win?",
        "Marie Curie won two Nobel Prizes.",
    ),
    (
        "What discoveries is Marie Curie associated with?",
        "Marie Curie is associated with the discoveries of polonium and radium.",
    ),
];
const LONG_PAPER_EXAMPLES: [PaperExample; 3] = [
    PaperExample {
        question: "Tell me a bio of Ada Lovelace.",
        answer: "Ada Lovelace was an English mathematician and writer best known for her work on Charles Babbage's Analytical Engine. Born in London in 1815, she is often described as one of the first computer programmers because she published notes describing how the machine could execute a sequence of operations.",
        baseline_answer: "Ada Lovelace was a French mathematician born in Paris in 1815 who built the Analytical Engine herself.",
        verification_questions: &LONG_EXAMPLE_1_VERIFICATION_QUESTIONS,
        verification_answers: &LONG_EXAMPLE_1_VERIFICATION_ANSWERS,
        final_answer: "Ada Lovelace was an English mathematician and writer born in London in 1815. She is best known for her notes on Charles Babbage's Analytical Engine, including an algorithm that is often cited as an early computer program.",
    },
    PaperExample {
        question: "Write a short biography of Grace Hopper.",
        answer: "Grace Hopper was an American computer scientist and U.S. Navy rear admiral. Born in New York City in 1906, she helped develop early compilers and played a major role in the development of COBOL.",
        baseline_answer: "Grace Hopper was a British computer scientist born in London in 1906 who invented COBOL by herself.",
        verification_questions: &LONG_EXAMPLE_2_VERIFICATION_QUESTIONS,
        verification_answers: &LONG_EXAMPLE_2_VERIFICATION_ANSWERS,
        final_answer: "Grace Hopper was an American computer scientist and U.S. Navy rear admiral born in New York City in 1906. She is best known for helping develop early compilers and for her leadership in the development of COBOL.",
    },
    PaperExample {
        question: "Tell me in short about Marie Curie.",
        answer: "Marie Curie was a Polish-born physicist and chemist who conducted pioneering research on radioactivity. She won Nobel Prizes in both Physics and Chemistry and is remembered for discoveries including polonium and radium.",
        baseline_answer: "Marie Curie was a French biologist who won one Nobel Prize for discovering electricity.",
        verification_questions: &LONG_EXAMPLE_3_VERIFICATION_QUESTIONS,
        verification_answers: &LONG_EXAMPLE_3_VERIFICATION_ANSWERS,
        final_answer: "Marie Curie was a Polish-born physicist and chemist known for pioneering research on radioactivity. She won two Nobel Prizes and is especially associated with the discoveries of polonium and radium.",
    },
];

const LONG_PLAN_EXAMPLE_1_LINES: [&str; 6] = [
    "Fact: Ada Lovelace was a French mathematician.",
    "Verification Question: What nationality was Ada Lovelace?",
    "Fact: Ada Lovelace was born in Paris in 1815.",
    "Verification Question: Where was Ada Lovelace born?",
    "Fact: Ada Lovelace built the Analytical Engine herself.",
    "Verification Question: Did Ada Lovelace build the Analytical Engine herself?",
];
const LONG_PLAN_EXAMPLE_2_LINES: [&str; 6] = [
    "Fact: Grace Hopper was a British computer scientist.",
    "Verification Question: What nationality was Grace Hopper?",
    "Fact: Grace Hopper was born in London in 1906.",
    "Verification Question: Where was Grace Hopper born?",
    "Fact: Grace Hopper invented COBOL by herself.",
    "Verification Question: Did Grace Hopper invent COBOL by herself?",
];
const LONG_PLAN_EXAMPLE_3_LINES: [&str; 6] = [
    "Fact: Marie Curie was a French biologist.",
    "Verification Question: What field is Marie Curie known for?",
    "Fact: Marie Curie won one Nobel Prize.",
    "Verification Question: How many Nobel Prizes did Marie Curie win?",
    "Fact: Marie Curie is associated with discovering electricity.",
    "Verification Question: What discoveries is Marie Curie associated with?",
];
const LONG_PLAN_EXAMPLES: [LongPlanExample; 3] = [
    LongPlanExample {
        question: "Tell me a bio of Ada Lovelace.",
        baseline_answer: "Ada Lovelace was a French mathematician born in Paris in 1815 who built the Analytical Engine herself.",
        plan_lines: &LONG_PLAN_EXAMPLE_1_LINES,
    },
    LongPlanExample {
        question: "Write a short biography of Grace Hopper.",
        baseline_answer: "Grace Hopper was a British computer scientist born in London in 1906 who invented COBOL by herself.",
        plan_lines: &LONG_PLAN_EXAMPLE_2_LINES,
    },
    LongPlanExample {
        question: "Tell me in short about Marie Curie.",
        baseline_answer: "Marie Curie was a French biologist who won one Nobel Prize for discovering electricity.",
        plan_lines: &LONG_PLAN_EXAMPLE_3_LINES,
    },
];

const LONG_CROSSCHECK_EXAMPLE_1_VERIFICATION_ANSWERS: [(&str, &str); 2] = [
    (
        "Where was Alan Turing born?",
        "Alan Turing was born in London, England, in 1912.",
    ),
    (
        "What is Alan Turing known for?",
        "Alan Turing helped lay the foundations of computer science and modern computing.",
    ),
];
const LONG_CROSSCHECK_EXAMPLE_2_VERIFICATION_ANSWERS: [(&str, &str); 3] = [
    (
        "Where was Grace Hopper born?",
        "Grace Hopper was born in New York City, New York.",
    ),
    (
        "What is Grace Hopper known for in computing?",
        "Grace Hopper helped develop early compilers and was a leading figure in the development of COBOL.",
    ),
    (
        "Did Grace Hopper invent COBOL by herself?",
        "No. Grace Hopper contributed to COBOL, but it was developed by a committee and broader team effort.",
    ),
];
const LONG_CROSSCHECK_EXAMPLE_3_VERIFICATION_ANSWERS: [(&str, &str); 3] = [
    (
        "What field is Marie Curie known for?",
        "Marie Curie is known for pioneering research on radioactivity in physics and chemistry.",
    ),
    (
        "How many Nobel Prizes did Marie Curie win?",
        "Marie Curie won two Nobel Prizes.",
    ),
    (
        "What discoveries is Marie Curie associated with?",
        "Marie Curie is associated with the discoveries of polonium and radium.",
    ),
];
const LONG_CROSSCHECK_EXAMPLES: [CrossCheckExample; 3] = [
    CrossCheckExample {
        original_fact: "Alan Turing was a British mathematician born in London in 1912 who helped lay the foundations of computer science.",
        verification_answers: &LONG_CROSSCHECK_EXAMPLE_1_VERIFICATION_ANSWERS,
        response: "CONSISTENT. Alan Turing was a British mathematician born in London in 1912 who helped lay the foundations of computer science.",
    },
    CrossCheckExample {
        original_fact: "Grace Hopper was a British computer scientist born in London in 1906 who invented COBOL by herself.",
        verification_answers: &LONG_CROSSCHECK_EXAMPLE_2_VERIFICATION_ANSWERS,
        response: "PARTIALLY CONSISTENT. Grace Hopper was a computer scientist born in 1906 who contributed to the development of COBOL.",
    },
    CrossCheckExample {
        original_fact: "Marie Curie was a French biologist who won one Nobel Prize for discovering electricity.",
        verification_answers: &LONG_CROSSCHECK_EXAMPLE_3_VERIFICATION_ANSWERS,
        response: "INCONSISTENT.",
    },
];

const LONG_FACTOR_REVISE_EXAMPLE_1_VERIFICATION_ANSWERS: [(&str, &str); 3] = [
    (
        "Where was Ada Lovelace born?",
        "Ada Lovelace was born in London, England, in 1815.",
    ),
    (
        "What is Ada Lovelace known for?",
        "Ada Lovelace is known for her notes on Charles Babbage's Analytical Engine and an early computer algorithm.",
    ),
    (
        "Did Ada Lovelace build the Analytical Engine herself?",
        "No. The Analytical Engine was designed by Charles Babbage.",
    ),
];
const LONG_FACTOR_REVISE_EXAMPLES: [FactorReviseExample; 3] = [
    FactorReviseExample {
        original_passage: "Ada Lovelace was a French mathematician born in Paris in 1815 who built the Analytical Engine herself.",
        verification_answers: &LONG_FACTOR_REVISE_EXAMPLE_1_VERIFICATION_ANSWERS,
        cross_check_response: "PARTIALLY CONSISTENT. Ada Lovelace was a mathematician born in 1815 who is known for her notes on Charles Babbage's Analytical Engine.",
        final_answer: "Ada Lovelace was a mathematician born in 1815 who is known for her notes on Charles Babbage's Analytical Engine.",
    },
    FactorReviseExample {
        original_passage: "Grace Hopper was a British computer scientist born in London in 1906 who invented COBOL by herself.",
        verification_answers: &LONG_CROSSCHECK_EXAMPLE_2_VERIFICATION_ANSWERS,
        cross_check_response: "PARTIALLY CONSISTENT. Grace Hopper was a computer scientist born in 1906 who contributed to the development of COBOL.",
        final_answer: "Grace Hopper was a computer scientist born in 1906 who contributed to the development of COBOL.",
    },
    FactorReviseExample {
        original_passage: "Alan Turing was a British mathematician born in London in 1912 who helped lay the foundations of computer science.",
        verification_answers: &LONG_CROSSCHECK_EXAMPLE_1_VERIFICATION_ANSWERS,
        cross_check_response: "CONSISTENT. Alan Turing was a British mathematician born in London in 1912 who helped lay the foundations of computer science.",
        final_answer: "Alan Turing was a British mathematician born in London in 1912 who helped lay the foundations of computer science.",
    },
];

fn normalize_paper_chain_type(chain_type: &str) -> &'static str {
    let chain_upper = chain_type.to_ascii_uppercase();
    if chain_upper.contains("WIKI") {
        "WIKI_CHAIN"
    } else if chain_upper.contains("LONG") {
        "LONG_CHAIN"
    } else {
        "MULTI_CHAIN"
    }
}

fn paper_examples_for_chain(chain_type: &str) -> &'static [PaperExample] {
    match normalize_paper_chain_type(chain_type) {
        "WIKI_CHAIN" => &WIKI_PAPER_EXAMPLES,
        "LONG_CHAIN" => &LONG_PAPER_EXAMPLES,
        _ => &MULTI_PAPER_EXAMPLES,
    }
}

fn join_nonempty_blocks(blocks: Vec<String>) -> String {
    blocks
        .into_iter()
        .filter(|block| !block.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn format_verification_question_list(questions: &[&str]) -> String {
    questions
        .iter()
        .map(|question| question.trim().to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_verification_pairs(pairs: &[(&str, &str)]) -> String {
    pairs
        .iter()
        .map(|(question, answer)| format!("Question: {}\nAnswer: {}", question, answer))
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_numbered_answers(answers: &[&str]) -> String {
    answers
        .iter()
        .enumerate()
        .map(|(index, answer)| format!("{}. {}", index + 1, answer))
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_factor_revise_evidence(
    verification_pairs: &[(&str, &str)],
    cross_check_response: &str,
) -> String {
    let mut blocks = Vec::new();
    let verification_text = format_verification_pairs(verification_pairs);
    if !verification_text.trim().is_empty() {
        blocks.push(verification_text);
    }
    if !cross_check_response.trim().is_empty() {
        blocks.push(format!("Cross-check: {}", cross_check_response.trim()));
    }
    blocks.join("\n")
}

pub fn build_paper_baseline_prompt(
    question: &str,
    chain_type: &str,
    few_shot_examples: usize,
) -> String {
    let mut blocks = paper_examples_for_chain(chain_type)
        .iter()
        .take(few_shot_examples)
        .map(|example| format!("Q: {}\nA: {}", example.question, example.answer))
        .collect::<Vec<_>>();
    blocks.push(format!("Q: {}\nA:", question));
    join_nonempty_blocks(blocks)
}

pub fn build_paper_verification_prompt(
    question: &str,
    baseline_response: &str,
    chain_type: &str,
    few_shot_examples: usize,
) -> String {
    if normalize_paper_chain_type(chain_type) == "LONG_CHAIN" {
        let mut blocks = LONG_PLAN_EXAMPLES
            .iter()
            .take(few_shot_examples)
            .map(|example| {
                format!(
                    "Context: Q: {}\nA: {}\nResponse:\n{}",
                    example.question,
                    example.baseline_answer,
                    example.plan_lines.join("\n"),
                )
            })
            .collect::<Vec<_>>();
        blocks.push(format!(
            "Context: Q: {}\nA: {}\nResponse:",
            question, baseline_response
        ));
        return join_nonempty_blocks(blocks);
    }

    let mut blocks = paper_examples_for_chain(chain_type)
        .iter()
        .take(few_shot_examples)
        .map(|example| {
            format!(
                "Context: Q: {}\nA: {}\nResponse:\n{}",
                example.question,
                example.baseline_answer,
                format_verification_question_list(example.verification_questions)
            )
        })
        .collect::<Vec<_>>();
    blocks.push(format!(
        "Context: Q: {}\nA: {}\nResponse:",
        question, baseline_response
    ));
    join_nonempty_blocks(blocks)
}

pub fn build_paper_execute_prompt(
    verification_question: &str,
    chain_type: &str,
    few_shot_examples: usize,
) -> String {
    let mut blocks = Vec::new();
    for example in paper_examples_for_chain(chain_type)
        .iter()
        .take(few_shot_examples)
    {
        for (question, answer) in example.verification_answers {
            blocks.push(format!("Q: {}\nA: {}", question, answer));
        }
    }
    blocks.push(format!("Q: {}\nA:", verification_question));
    join_nonempty_blocks(blocks)
}

pub fn build_paper_two_step_execute_prompt(
    verification_questions: &[String],
    chain_type: &str,
    few_shot_examples: usize,
) -> String {
    let mut blocks = paper_examples_for_chain(chain_type)
        .iter()
        .take(few_shot_examples)
        .map(|example| {
            let answers = example
                .verification_answers
                .iter()
                .map(|(_, answer)| *answer)
                .collect::<Vec<_>>();
            format!(
                "Context:\n{}\nResponse:\n{}",
                format_verification_question_list(example.verification_questions),
                format_numbered_answers(&answers),
            )
        })
        .collect::<Vec<_>>();
    let current_questions = verification_questions
        .iter()
        .map(|question| question.trim().to_string())
        .collect::<Vec<_>>()
        .join("\n");
    blocks.push(format!("Context:\n{}\nResponse:", current_questions));
    join_nonempty_blocks(blocks)
}

pub fn build_paper_joint_prompt(
    question: &str,
    baseline_response: &str,
    chain_type: &str,
    few_shot_examples: usize,
) -> String {
    let mut blocks = paper_examples_for_chain(chain_type)
        .iter()
        .take(few_shot_examples)
        .map(|example| {
            format!(
                "Context: Q: {}\nA: {}\nResponse:\n{}",
                example.question,
                example.baseline_answer,
                format_verification_pairs(example.verification_answers),
            )
        })
        .collect::<Vec<_>>();
    blocks.push(format!(
        "Context: Q: {}\nA: {}\nResponse:",
        question, baseline_response
    ));
    join_nonempty_blocks(blocks)
}

pub fn build_paper_final_prompt(
    question: &str,
    baseline_response: &str,
    verification_qa_pairs: &[(String, String)],
    chain_type: &str,
    few_shot_examples: usize,
) -> String {
    let mut blocks = paper_examples_for_chain(chain_type)
        .iter()
        .take(few_shot_examples)
        .map(|example| {
            format!(
                "Context: Q: {}\nA: {}\nFrom another source,\n{}\nResponse: {}",
                example.question,
                example.baseline_answer,
                format_verification_pairs(example.verification_answers),
                example.final_answer
            )
        })
        .collect::<Vec<_>>();
    let pairs = verification_qa_pairs
        .iter()
        .map(|(question, answer)| format!("Question: {}\nAnswer: {}", question, answer))
        .collect::<Vec<_>>()
        .join("\n");
    blocks.push(format!(
        "Context: Q: {}\nA: {}\nFrom another source,\n{}\nResponse:",
        question, baseline_response, pairs
    ));
    join_nonempty_blocks(blocks)
}

pub fn build_paper_crosscheck_prompt(
    original_fact: &str,
    verification_qa_pairs: &[(String, String)],
    few_shot_examples: usize,
) -> String {
    let mut blocks = LONG_CROSSCHECK_EXAMPLES
        .iter()
        .take(few_shot_examples)
        .map(|example| {
            format!(
                "Context: {}\nFrom another source,\n{}\nResponse: {}",
                example.original_fact,
                format_verification_pairs(example.verification_answers),
                example.response,
            )
        })
        .collect::<Vec<_>>();
    let pairs = verification_qa_pairs
        .iter()
        .map(|(question, answer)| format!("Question: {}\nAnswer: {}", question, answer))
        .collect::<Vec<_>>()
        .join("\n");
    blocks.push(format!(
        "Context: {}\nFrom another source,\n{}\nResponse:",
        original_fact, pairs
    ));
    join_nonempty_blocks(blocks)
}

pub fn build_paper_factor_revise_passage_prompt(
    original_passage: &str,
    verification_qa_pairs: &[(String, String)],
    cross_check_response: &str,
    few_shot_examples: usize,
) -> String {
    let mut blocks = LONG_FACTOR_REVISE_EXAMPLES
        .iter()
        .take(few_shot_examples)
        .map(|example| {
            format!(
                "Context: {}\nFrom another source,\n{}\nResponse: {}",
                example.original_passage,
                format_factor_revise_evidence(
                    example.verification_answers,
                    example.cross_check_response,
                ),
                example.final_answer,
            )
        })
        .collect::<Vec<_>>();
    let verification_pairs = verification_qa_pairs
        .iter()
        .map(|(question, answer)| (question.as_str(), answer.as_str()))
        .collect::<Vec<_>>();
    blocks.push(format!(
        "Context: {}\nFrom another source,\n{}\nResponse:",
        original_passage,
        format_factor_revise_evidence(&verification_pairs, cross_check_response),
    ));
    join_nonempty_blocks(blocks)
}

pub fn resolve_paper_variant(
    cove_cfg: &CoVeConfig,
    dataset_name: Option<&str>,
    chain_type: &str,
) -> CoVeVariant {
    if !cove_cfg.paper_faithful {
        return CoVeVariant::Factored;
    }
    if cove_cfg.cove_variant != CoVeVariant::Auto {
        return cove_cfg.cove_variant;
    }

    match cove_cfg.paper_task {
        PaperTask::Wikidata => CoVeVariant::TwoStep,
        PaperTask::WikiCategory => CoVeVariant::Factored,
        PaperTask::Multispanqa => CoVeVariant::Factored,
        PaperTask::Biography => CoVeVariant::FactorRevise,
        PaperTask::Auto => {
            let dataset = dataset_name
                .unwrap_or_default()
                .to_ascii_lowercase()
                .replace('-', "_");
            match dataset.as_str() {
                "wikidata" | "cove_wikidata" | "wikidata_list" => CoVeVariant::TwoStep,
                "quest" | "wiki_category" | "wiki_category_list" | "cove_quest" => {
                    CoVeVariant::Factored
                }
                "multispanqa" | "cove_multispanqa" => CoVeVariant::Factored,
                "biography" | "biographies" | "factscore" | "cove_biography" => {
                    CoVeVariant::FactorRevise
                }
                _ => {
                    if normalize_paper_chain_type(chain_type) == "LONG_CHAIN" {
                        CoVeVariant::FactorRevise
                    } else {
                        CoVeVariant::Factored
                    }
                }
            }
        }
    }
}

pub fn paper_stop_sequences(phase: &str, chain_type: &str, max_wiki_items: usize) -> Vec<String> {
    let mut merged = match phase {
        "baseline" | "execute" => PAPER_QA_STOP_SEQUENCES
            .iter()
            .map(|item| (*item).to_string())
            .collect::<Vec<_>>(),
        _ => PAPER_CONTEXT_STOP_SEQUENCES
            .iter()
            .map(|item| (*item).to_string())
            .collect::<Vec<_>>(),
    };

    if normalize_paper_chain_type(chain_type) == "WIKI_CHAIN"
        && matches!(phase, "baseline" | "verification" | "final")
    {
        let next_item = max_wiki_items.max(1) + 1;
        for stop in [format!("\n{}.", next_item), format!("\n{})", next_item)] {
            if !merged.iter().any(|existing| existing == &stop) {
                merged.push(stop);
            }
        }
    }

    merged
}

pub fn stop_tokens_for_text(model: &inferlet::Model, stop_text: &[String]) -> Vec<Vec<u32>> {
    let tokenizer = model.get_tokenizer();
    let mut stop_tokens = model.eos_tokens();
    for stop in stop_text {
        stop_tokens.push(tokenizer.tokenize(stop));
    }
    stop_tokens
}

pub fn effective_use_search(cove_cfg: &CoVeConfig) -> bool {
    cove_cfg.use_search && !cove_cfg.paper_faithful
}

pub fn resolve_factor_revise_passage_concurrency(
    cove_cfg: &CoVeConfig,
    total_passages: usize,
    default_parallel: bool,
) -> usize {
    if total_passages == 0 {
        return 1;
    }

    let configured_parallelism = cove_cfg
        .factor_revise_passage_concurrency
        .unwrap_or_else(|| cove_cfg.question_concurrency.max(1).min(2))
        .max(1);

    let desired = match cove_cfg.factor_revise_passage_mode {
        FactorRevisePassageMode::Auto => {
            if default_parallel {
                configured_parallelism
            } else {
                1
            }
        }
        FactorRevisePassageMode::Serial => 1,
        FactorRevisePassageMode::Parallel => configured_parallelism,
    };

    total_passages.min(desired)
}

pub fn parse_verification_questions(text: &str, max: usize) -> Vec<String> {
    let raw_text = text.trim();
    if raw_text.is_empty() || max == 0 {
        return Vec::new();
    }

    let mut candidates = Vec::new();

    for line in raw_text.lines() {
        candidates.extend(extract_question_candidates(line));
    }

    if candidates.is_empty() {
        for chunk in split_numbered_chunks(raw_text) {
            candidates.extend(extract_question_candidates(chunk));
        }
    }

    if candidates.is_empty() {
        candidates.extend(extract_question_candidates(raw_text));
    }

    let mut deduped = Vec::new();
    let mut seen = BTreeSet::new();
    for candidate in candidates {
        if seen.insert(candidate.clone()) {
            deduped.push(candidate);
            if deduped.len() >= max {
                break;
            }
        }
    }

    deduped
}

fn extract_question_candidates(text: &str) -> Vec<String> {
    let stripped = strip_list_prefix(text);
    if stripped.is_empty() {
        return Vec::new();
    }

    let cleaned = if let Some(rest) = strip_question_label(&stripped) {
        rest.to_string()
    } else if stripped.to_ascii_lowercase().starts_with("fact:") {
        return Vec::new();
    } else {
        stripped
    };

    let parts: Vec<String> = if cleaned.contains('?') {
        split_after_question_marks(&cleaned)
            .into_iter()
            .map(|chunk: &str| chunk.trim().to_string())
            .collect::<Vec<_>>()
    } else if cleaned.contains(',') || cleaned.contains(';') {
        cleaned
            .split(|ch| ch == ',' || ch == ';')
            .map(|chunk: &str| chunk.trim().to_string())
            .collect::<Vec<_>>()
    } else {
        vec![cleaned.to_string()]
    };

    parts
        .into_iter()
        .filter(|part: &String| part.len() > 5)
        .collect()
}

fn strip_question_label(text: &str) -> Option<&str> {
    const LABELS: [&str; 4] = [
        "Verification Question:",
        "verification question:",
        "Question:",
        "question:",
    ];

    for label in LABELS {
        if let Some(rest) = text.strip_prefix(label) {
            return Some(rest.trim());
        }
    }
    None
}

pub fn parse_numbered_answers(text: &str, max: usize) -> Vec<String> {
    let raw_text = text.trim();
    if raw_text.is_empty() || max == 0 {
        return Vec::new();
    }

    let mut answers = Vec::new();

    for line in raw_text.lines() {
        let cleaned = strip_list_prefix(line);
        if !cleaned.is_empty() {
            answers.push(cleaned);
        }
    }

    if answers.is_empty() {
        for chunk in split_numbered_chunks(raw_text) {
            let cleaned = strip_list_prefix(chunk);
            if !cleaned.is_empty() {
                answers.push(cleaned);
            }
        }
    }

    answers.into_iter().take(max).collect()
}

pub fn parse_verification_pairs(text: &str, max: usize) -> Vec<(String, String)> {
    let raw_text = text.trim();
    if raw_text.is_empty() || max == 0 {
        return Vec::new();
    }

    let mut pairs = Vec::new();
    for block in raw_text.split("Question:").skip(1) {
        let trimmed = block.trim();
        let Some((question, answer)) = trimmed.split_once("Answer:") else {
            continue;
        };
        let question = question.trim().to_string();
        let answer = answer.trim().to_string();
        if !question.is_empty() && !answer.is_empty() {
            pairs.push((question, answer));
        }
        if pairs.len() >= max {
            break;
        }
    }

    pairs
}

pub fn split_passages(text: &str) -> Vec<String> {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.trim().is_empty() {
        return Vec::new();
    }

    let mut passages = Vec::new();
    let mut start = 0usize;
    for (idx, ch) in normalized.char_indices() {
        if !matches!(ch, '.' | '!' | '?') {
            continue;
        }
        let end = idx + ch.len_utf8();
        let passage = normalized[start..end].trim();
        if !passage.is_empty() {
            passages.push(passage.to_string());
        }

        start = end;
        while start < normalized.len() {
            let next = normalized[start..].chars().next().unwrap();
            if next.is_whitespace() {
                start += next.len_utf8();
            } else {
                break;
            }
        }
    }

    let tail = normalized[start..].trim();
    if !tail.is_empty() {
        passages.push(tail.to_string());
    }

    if passages.is_empty() {
        vec![normalized]
    } else {
        passages
    }
}

pub fn normalize_crosscheck_response(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return "INCONSISTENT.".to_string();
    }

    let upper = trimmed.to_ascii_uppercase();
    let (label, rest) = if upper.starts_with("PARTIALLY CONSISTENT") {
        (
            "PARTIALLY CONSISTENT",
            trimmed["PARTIALLY CONSISTENT".len()..].trim(),
        )
    } else if upper.starts_with("INCONSISTENT") {
        ("INCONSISTENT", trimmed["INCONSISTENT".len()..].trim())
    } else if upper.starts_with("CONSISTENT") {
        ("CONSISTENT", trimmed["CONSISTENT".len()..].trim())
    } else {
        return trimmed.to_string();
    };

    let rest = rest.trim_start_matches('.').trim();
    if rest.is_empty() {
        format!("{}.", label)
    } else {
        format!("{}. {}", label, rest)
    }
}

pub fn extract_consistent_crosscheck_body(text: &str) -> Option<String> {
    let normalized = normalize_crosscheck_response(text);
    let upper = normalized.to_ascii_uppercase();
    if !(upper.starts_with("CONSISTENT.") || upper.starts_with("PARTIALLY CONSISTENT.")) {
        return None;
    }
    let body = normalized
        .split_once('.')
        .map(|(_, suffix)| suffix.trim().to_string())
        .unwrap_or_default();
    if body.is_empty() {
        None
    } else {
        Some(body)
    }
}

pub fn splice_consistent_body(text: &str) -> String {
    let mut seen = BTreeSet::new();
    let mut passages = Vec::new();
    for passage in split_passages(text) {
        let normalized = passage.split_whitespace().collect::<Vec<_>>().join(" ");
        if !normalized.is_empty() && seen.insert(normalized.clone()) {
            passages.push(normalized);
        }
    }
    passages.join(" ")
}

fn split_numbered_chunks(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut starts = Vec::new();

    for idx in 0..bytes.len() {
        if is_numbered_chunk_start(bytes, idx) {
            starts.push(idx);
        }
    }

    if starts.is_empty() {
        return vec![text];
    }

    let mut chunks = Vec::new();
    for (pos, start) in starts.iter().enumerate() {
        let end = starts.get(pos + 1).copied().unwrap_or(text.len());
        let chunk = text[*start..end].trim();
        if !chunk.is_empty() {
            chunks.push(chunk);
        }
    }
    chunks
}

fn is_numbered_chunk_start(bytes: &[u8], idx: usize) -> bool {
    if !bytes[idx].is_ascii_digit() {
        return false;
    }
    if idx > 0 {
        let prev = bytes[idx - 1];
        if !(prev.is_ascii_whitespace() || matches!(prev, b',' | b';' | b'(' | b'[')) {
            return false;
        }
    }

    let mut end = idx;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    if end >= bytes.len() || !matches!(bytes[end], b'.' | b')') {
        return false;
    }
    if end + 1 >= bytes.len() {
        return false;
    }

    bytes[end + 1].is_ascii_whitespace()
}

fn split_after_question_marks(text: &str) -> Vec<&str> {
    let mut chunks = Vec::new();
    let mut start = 0usize;

    for (idx, ch) in text.char_indices() {
        if ch != '?' {
            continue;
        }
        let end = idx + ch.len_utf8();
        let chunk = text[start..end].trim();
        if !chunk.is_empty() {
            chunks.push(chunk);
        }

        start = end;
        while start < text.len() {
            let next = text[start..].chars().next().unwrap();
            if next.is_whitespace() || next == ',' || next == ';' {
                start += next.len_utf8();
            } else {
                break;
            }
        }
    }

    let tail = text[start..].trim();
    if !tail.is_empty() {
        chunks.push(tail);
    }

    chunks
}

fn strip_list_prefix(text: &str) -> String {
    let trimmed = text.trim();
    if let Some(first) = trimmed.chars().next() {
        if matches!(first, '-' | '*' | '•') {
            return trimmed[first.len_utf8()..].trim_start().to_string();
        }
    }
    strip_number_prefix_local(trimmed)
}

// ============================================================================
// DuckDuckGo HTTP Search with Shared Cache
// ============================================================================

#[derive(Clone, Debug)]
pub struct SearchResources {
    mode: SearchMode,
    cache_path: Option<String>,
    cache: Arc<Mutex<BTreeMap<String, String>>>,
    dirty: Arc<Mutex<bool>>,
}

impl SearchResources {
    pub fn new(mode: SearchMode, cache_path: Option<String>) -> Result<Self> {
        let cache = if let Some(path) = cache_path.as_deref() {
            debug_path_state("search_cache", path);
            if Path::new(path).exists() {
                load_search_cache(path)?
            } else if mode == SearchMode::Replay {
                return Err(inferlet::anyhow!(
                    "Frozen search cache file '{}' does not exist in replay mode.",
                    path
                ));
            } else {
                eprintln!(
                    "[cove-debug] search cache file missing; starting empty cache in {} mode",
                    mode.as_str()
                );
                BTreeMap::new()
            }
        } else {
            eprintln!("[cove-debug] search_cache disabled");
            BTreeMap::new()
        };

        Ok(Self {
            mode,
            cache_path,
            cache: Arc::new(Mutex::new(cache)),
            dirty: Arc::new(Mutex::new(false)),
        })
    }

    pub fn mode(&self) -> SearchMode {
        self.mode
    }

    pub fn cache_path(&self) -> Option<&str> {
        self.cache_path.as_deref()
    }

    fn get_cached(&self, query: &str) -> Option<String> {
        self.cache.lock().unwrap().get(query).cloned()
    }

    fn store_cached(&self, query: &str, result: String) {
        let mut cache = self.cache.lock().unwrap();
        if cache.contains_key(query) {
            return;
        }
        cache.insert(query.to_string(), result);
        *self.dirty.lock().unwrap() = true;
    }

    pub async fn search_ddg(
        &self,
        client: &Client,
        query: &str,
        max_results: usize,
        timeout_secs: f32,
    ) -> Result<String> {
        if matches!(self.mode, SearchMode::Record | SearchMode::Replay) {
            if let Some(cached) = self.get_cached(query) {
                return Ok(cached);
            }
        }

        if self.mode == SearchMode::Replay {
            return Err(inferlet::anyhow!(
                "Frozen search cache miss for query: {}. Run once with search_mode='record' first.",
                query
            ));
        }

        let result = search_ddg_live(client, query, max_results, timeout_secs).await?;

        if self.mode == SearchMode::Record {
            self.store_cached(query, result.clone());
        }

        Ok(result)
    }

    pub fn persist(&self) -> Result<()> {
        if self.mode == SearchMode::Live {
            return Ok(());
        }

        let Some(path) = self.cache_path.as_deref() else {
            return Ok(());
        };
        debug_path_state("persist_search_cache", path);

        if !*self.dirty.lock().unwrap() {
            return Ok(());
        }

        if let Some(parent) = Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }

        let queries = self.cache.lock().unwrap().clone();
        let payload = serde_json::json!({
            "version": 1,
            "provider": "duckduckgo",
            "queries": queries,
        });
        fs::write(path, serde_json::to_string_pretty(&payload)?)?;
        *self.dirty.lock().unwrap() = false;
        Ok(())
    }
}

fn load_search_cache(path: &str) -> Result<BTreeMap<String, String>> {
    debug_path_state("load_search_cache", path);
    let content = fs::read_to_string(path)?;
    let value: serde_json::Value = serde_json::from_str(&content)?;
    let query_value = value.get("queries").unwrap_or(&value);
    let object = query_value.as_object().ok_or_else(|| {
        inferlet::anyhow!("Search cache file '{}' must contain a JSON object", path)
    })?;

    let mut cache = BTreeMap::new();
    for (query, entry) in object {
        let result = if let Some(result) = entry.as_str() {
            result.to_string()
        } else if let Some(result) = entry.get("result").and_then(|result| result.as_str()) {
            result.to_string()
        } else {
            continue;
        };
        cache.insert(query.clone(), result);
    }

    Ok(cache)
}

async fn search_ddg_inner(client: &Client, query: &str, max_results: usize) -> Result<String> {
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
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
        let mut results = Vec::new();

        if let Some(abstract_text) = json.get("AbstractText").and_then(|value| value.as_str()) {
            if !abstract_text.is_empty() {
                results.push(format!("Summary: {}", abstract_text));
            }
        }

        if let Some(topics) = json.get("RelatedTopics").and_then(|value| value.as_array()) {
            for (idx, topic) in topics.iter().take(max_results).enumerate() {
                if let Some(text) = topic.get("Text").and_then(|value| value.as_str()) {
                    results.push(format!("{}. {}", idx + 1, text));
                }
            }
        }

        if !results.is_empty() {
            return Ok(results.join("\n"));
        }
    }

    Ok("No search results found.".to_string())
}

pub async fn search_ddg_live(
    client: &Client,
    query: &str,
    max_results: usize,
    timeout_secs: f32,
) -> Result<String> {
    if timeout_secs <= 0.0 {
        return search_ddg_inner(client, query, max_results).await;
    }

    let search = search_ddg_inner(client, query, max_results);
    let timeout_ms = (timeout_secs.max(0.0) * 1000.0).ceil() as u64;
    let timeout = inferlet::wstd::task::sleep(Duration::from_millis(timeout_ms));

    pin_mut!(search);
    pin_mut!(timeout);

    match select(search, timeout).await {
        Either::Left((result, _)) => result,
        Either::Right((_, _)) => Err(inferlet::anyhow!(
            "DuckDuckGo search timed out after {:.2} seconds",
            timeout_secs
        )),
    }
}

// ============================================================================
// Metrics
// ============================================================================

#[derive(Clone, Copy, Debug)]
enum MetricKind {
    FactscoreLite,
    ContainsMatch,
    EntityPrecision,
    QaF1,
    QaExactMatch,
    NumericExactMatch,
    ListPrecision,
    ListF1,
    RougeLF1,
}

#[derive(Clone, Copy, Debug)]
pub struct MetricSpec {
    pub name: &'static str,
    pub display_name: &'static str,
    pub threshold: Option<f64>,
    kind: MetricKind,
}

pub fn resolve_metric(metric_name: &str, dataset_name: &str) -> MetricSpec {
    let resolved = if metric_name.is_empty() || metric_name == "auto" {
        match dataset_name {
            "squad" | "nq" | "trivia_qa" | "halu_eval" | "bioasq" => "qa_f1",
            "svamp" => "numeric_exact_match",
            "multispanqa" | "cove_multispanqa" | "popqa" => "list_f1",
            "wikidata" | "cove_wikidata" | "quest" | "cove_quest" | "wiki_category"
            | "cove_wiki_category" => "list_precision",
            "eli5" | "longform" | "asqa" | "biography" | "cove_biography" => "rouge_l_f1",
            _ => "qa_f1",
        }
    } else {
        metric_name
    };

    match resolved {
        "factscore" | "factscore_lite" => MetricSpec {
            name: "factscore_lite",
            display_name: "FACTSCORE Lite",
            threshold: Some(0.5),
            kind: MetricKind::FactscoreLite,
        },
        "contains_match" => MetricSpec {
            name: "contains_match",
            display_name: "Contains Match",
            threshold: Some(1.0),
            kind: MetricKind::ContainsMatch,
        },
        "entity_precision" => MetricSpec {
            name: "entity_precision",
            display_name: "Entity Precision",
            threshold: Some(0.5),
            kind: MetricKind::EntityPrecision,
        },
        "list_precision" => MetricSpec {
            name: "list_precision",
            display_name: "List Precision",
            threshold: None,
            kind: MetricKind::ListPrecision,
        },
        "qa_f1" => MetricSpec {
            name: "qa_f1",
            display_name: "QA Token F1",
            threshold: Some(0.5),
            kind: MetricKind::QaF1,
        },
        "qa_exact_match" => MetricSpec {
            name: "qa_exact_match",
            display_name: "QA Exact Match",
            threshold: Some(1.0),
            kind: MetricKind::QaExactMatch,
        },
        "numeric_exact_match" => MetricSpec {
            name: "numeric_exact_match",
            display_name: "Numeric Exact Match",
            threshold: Some(1.0),
            kind: MetricKind::NumericExactMatch,
        },
        "list_f1" => MetricSpec {
            name: "list_f1",
            display_name: "List F1",
            threshold: Some(0.5),
            kind: MetricKind::ListF1,
        },
        "rouge_l_f1" => MetricSpec {
            name: "rouge_l_f1",
            display_name: "ROUGE-L F1",
            threshold: Some(0.3),
            kind: MetricKind::RougeLF1,
        },
        _ => panic!("Unsupported metric '{}'", metric_name),
    }
}

pub fn score_prediction(metric: MetricSpec, prediction: &str, references: &[String]) -> f64 {
    match metric.kind {
        MetricKind::FactscoreLite => factscore_lite(prediction, references),
        MetricKind::ContainsMatch => contains_match(prediction, references),
        MetricKind::EntityPrecision => entity_precision(prediction, references),
        MetricKind::ListPrecision => list_precision(prediction, references),
        MetricKind::QaF1 => best_reference_score(prediction, references, token_f1_score),
        MetricKind::QaExactMatch => best_reference_score(prediction, references, exact_match_score),
        MetricKind::NumericExactMatch => numeric_exact_match(prediction, references),
        MetricKind::ListF1 => list_f1(prediction, references),
        MetricKind::RougeLF1 => best_reference_score(prediction, references, rouge_l_score),
    }
}

fn best_reference_score(
    prediction: &str,
    references: &[String],
    scorer: fn(&str, &str) -> f64,
) -> f64 {
    references
        .iter()
        .filter(|reference| !reference.is_empty())
        .map(|reference| scorer(prediction, reference))
        .fold(0.0, f64::max)
}

fn factscore_lite(prediction: &str, references: &[String]) -> f64 {
    if references.is_empty() {
        return 0.0;
    }

    let prediction_norm = normalize_text(prediction);
    let matches = references
        .iter()
        .filter(|reference| {
            let reference_norm = normalize_text(reference);
            !reference_norm.is_empty() && prediction_norm.contains(&reference_norm)
        })
        .count();

    matches as f64 / references.len() as f64
}

fn contains_match(prediction: &str, references: &[String]) -> f64 {
    if references.is_empty() {
        return 0.0;
    }

    let prediction_norm = normalize_text(prediction);
    let matches = references
        .iter()
        .filter(|reference| {
            let reference_norm = normalize_text(reference);
            !reference_norm.is_empty() && prediction_norm.contains(&reference_norm)
        })
        .count();

    if matches == references.len() {
        1.0
    } else {
        0.0
    }
}

fn entity_precision(prediction: &str, references: &[String]) -> f64 {
    if references.is_empty() {
        return 0.0;
    }

    let reference_set = references
        .iter()
        .map(|reference| normalize_text(reference))
        .filter(|reference| !reference.is_empty())
        .collect::<BTreeSet<_>>();
    if reference_set.is_empty() {
        return 0.0;
    }

    let prediction_set = split_prediction_items(prediction)
        .into_iter()
        .collect::<BTreeSet<_>>();
    if prediction_set.is_empty() {
        return 0.0;
    }

    let matches = prediction_set.intersection(&reference_set).count();
    matches as f64 / reference_set.len() as f64
}

fn list_precision(prediction: &str, references: &[String]) -> f64 {
    let reference_items = references
        .iter()
        .map(|reference| normalize_text(reference))
        .filter(|reference| !reference.is_empty())
        .collect::<Vec<_>>();
    if reference_items.is_empty() {
        return 0.0;
    }

    let prediction_items = split_prediction_items(prediction);
    if prediction_items.is_empty() {
        return 0.0;
    }

    let matched_predictions = prediction_items
        .iter()
        .filter(|prediction_item| {
            reference_items
                .iter()
                .any(|reference| list_item_matches(prediction_item, reference))
        })
        .count();
    matched_predictions as f64 / prediction_items.len() as f64
}

fn exact_match_score(prediction: &str, reference: &str) -> f64 {
    if normalize_text(prediction) == normalize_text(reference) {
        1.0
    } else {
        0.0
    }
}

fn token_f1_score(prediction: &str, reference: &str) -> f64 {
    let prediction_tokens = normalized_tokens(prediction);
    let reference_tokens = normalized_tokens(reference);
    if prediction_tokens.is_empty() || reference_tokens.is_empty() {
        return 0.0;
    }

    let mut remaining = reference_tokens.clone();
    let mut overlap = 0usize;
    for token in &prediction_tokens {
        if let Some(position) = remaining.iter().position(|candidate| candidate == token) {
            remaining.remove(position);
            overlap += 1;
        }
    }

    let precision = overlap as f64 / prediction_tokens.len() as f64;
    let recall = overlap as f64 / reference_tokens.len() as f64;
    safe_f1(precision, recall)
}

fn rouge_l_score(prediction: &str, reference: &str) -> f64 {
    let prediction_tokens = normalized_tokens(prediction);
    let reference_tokens = normalized_tokens(reference);
    if prediction_tokens.is_empty() || reference_tokens.is_empty() {
        return 0.0;
    }

    let lcs = lcs_length(&prediction_tokens, &reference_tokens) as f64;
    let precision = lcs / prediction_tokens.len() as f64;
    let recall = lcs / reference_tokens.len() as f64;
    safe_f1(precision, recall)
}

fn numeric_exact_match(prediction: &str, references: &[String]) -> f64 {
    let Some(prediction_number) = extract_number(prediction) else {
        return 0.0;
    };

    for reference in references {
        if let Some(reference_number) = extract_number(reference) {
            if (prediction_number - reference_number).abs() <= 1e-9 {
                return 1.0;
            }
        }
    }
    0.0
}

fn list_f1(prediction: &str, references: &[String]) -> f64 {
    let reference_items = references
        .iter()
        .map(|reference| normalize_text(reference))
        .filter(|reference| !reference.is_empty())
        .collect::<Vec<_>>();
    if reference_items.is_empty() {
        return 0.0;
    }

    let prediction_items = split_prediction_items(prediction);
    if prediction_items.is_empty() {
        return 0.0;
    }

    let matched_references = reference_items
        .iter()
        .filter(|reference| {
            prediction_items
                .iter()
                .any(|prediction_item| list_item_matches(prediction_item, reference))
        })
        .count();
    let matched_predictions = prediction_items
        .iter()
        .filter(|prediction_item| {
            reference_items
                .iter()
                .any(|reference| list_item_matches(prediction_item, reference))
        })
        .count();

    let precision = matched_predictions as f64 / prediction_items.len() as f64;
    let recall = matched_references as f64 / reference_items.len() as f64;
    safe_f1(precision, recall)
}

fn safe_f1(precision: f64, recall: f64) -> f64 {
    if precision + recall == 0.0 {
        return 0.0;
    }
    2.0 * precision * recall / (precision + recall)
}

fn lcs_length(left: &[String], right: &[String]) -> usize {
    if left.is_empty() || right.is_empty() {
        return 0;
    }

    let mut prev = vec![0usize; right.len() + 1];
    let mut curr = vec![0usize; right.len() + 1];

    for left_token in left {
        for (idx, right_token) in right.iter().enumerate() {
            if left_token == right_token {
                curr[idx + 1] = prev[idx] + 1;
            } else {
                curr[idx + 1] = prev[idx + 1].max(curr[idx]);
            }
        }
        prev.clone_from(&curr);
        curr.fill(0);
    }

    prev[right.len()]
}

fn normalized_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_alphanumeric() {
            for lower in ch.to_lowercase() {
                current.push(lower);
            }
        } else if !current.is_empty() {
            push_token(&mut tokens, &mut current);
        }
    }

    if !current.is_empty() {
        push_token(&mut tokens, &mut current);
    }

    tokens
}

fn push_token(tokens: &mut Vec<String>, current: &mut String) {
    if current != "a" && current != "an" && current != "the" {
        tokens.push(std::mem::take(current));
    } else {
        current.clear();
    }
}

fn normalize_text(text: &str) -> String {
    normalized_tokens(text).join(" ")
}

fn extract_number(text: &str) -> Option<f64> {
    let cleaned = text.replace(',', "");
    let mut current = String::new();
    let mut started = false;
    let mut has_digit = false;
    let mut seen_dot = false;

    for ch in cleaned.chars() {
        if ch.is_ascii_digit() {
            current.push(ch);
            started = true;
            has_digit = true;
        } else if ch == '-' && !started {
            current.push(ch);
            started = true;
        } else if ch == '.' && started && !seen_dot {
            current.push(ch);
            seen_dot = true;
        } else if started {
            if has_digit {
                break;
            }
            current.clear();
            started = false;
            seen_dot = false;
        }
    }

    if !has_digit {
        return None;
    }

    current.parse::<f64>().ok()
}

fn split_prediction_items(prediction: &str) -> Vec<String> {
    let mut items = Vec::new();

    for raw_line in prediction.lines() {
        let stripped = strip_number_prefix_local(raw_line);
        if stripped.is_empty() {
            continue;
        }
        for part in stripped.split(';') {
            let cleaned = part.trim_matches(|ch: char| ch == '-' || ch == ' ' || ch == '\t');
            if !cleaned.is_empty() {
                items.push(cleaned.to_string());
            }
        }
    }

    if items.len() <= 1 && prediction.contains(',') {
        let comma_items = prediction
            .split(',')
            .map(strip_number_prefix_local)
            .map(|item| {
                item.trim_matches(|ch: char| ch == '-' || ch == ' ' || ch == '\t')
                    .to_string()
            })
            .filter(|item| !item.is_empty())
            .collect::<Vec<_>>();
        if comma_items.len() > 1 {
            items = comma_items;
        }
    }

    if items.is_empty() && !prediction.trim().is_empty() {
        items.push(prediction.trim().to_string());
    }

    let mut seen = BTreeSet::new();
    let mut normalized = Vec::new();
    for item in items {
        let item_norm = normalize_text(&item);
        if !item_norm.is_empty() && seen.insert(item_norm.clone()) {
            normalized.push(item_norm);
        }
    }

    normalized
}

fn list_item_matches(prediction_item: &str, reference_item: &str) -> bool {
    prediction_item == reference_item
        || prediction_item.contains(reference_item)
        || reference_item.contains(prediction_item)
        || token_f1_score(prediction_item, reference_item) >= 0.8
}

fn strip_number_prefix_local(line: &str) -> String {
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
        if ch == '.' || ch == ')' || ch == '\\' {
            rest = &rest[ch.len_utf8()..];
        } else {
            return trimmed.to_string();
        }
    }

    rest.trim_start().to_string()
}

pub fn percentile(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap());
    let rank = (percentile / 100.0) * (sorted.len().saturating_sub(1) as f64);
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        return sorted[lower];
    }

    let weight = rank - lower as f64;
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_verification_questions_splits_bundled_questions() {
        let text = "Where was Suthida born?, What is Suthida known for?, What is Suthida's title?";
        let parsed = parse_verification_questions(text, 10);
        assert_eq!(
            parsed,
            vec![
                "Where was Suthida born?",
                "What is Suthida known for?",
                "What is Suthida's title?",
            ]
        );
    }

    #[test]
    fn parse_verification_questions_prefers_labeled_question_lines() {
        let text = "Fact: Ada Lovelace was born in Paris in 1815.\nVerification Question: Where was Ada Lovelace born?";
        let parsed = parse_verification_questions(text, 10);
        assert_eq!(parsed, vec!["Where was Ada Lovelace born?"]);
    }

    #[test]
    fn splice_consistent_body_dedupes_sentences() {
        let spliced = splice_consistent_body(
            "Suthida is the Queen of Thailand. Suthida is the Queen of Thailand.",
        );
        assert_eq!(spliced, "Suthida is the Queen of Thailand.");
    }
}
