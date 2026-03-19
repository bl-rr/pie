use futures::future::join_all;
use inferlet::forward::Forward;
use inferlet::stop_condition::{self, StopCondition};
use inferlet::wstd::time::Duration;
use inferlet::{Args, Context, Result, Sampler, anyhow};
use serde::{Deserialize, Serialize};

const HELP: &str = "\
A benchmark script for optimized ReAct-style function-calling scenarios.

USAGE:
  agent-react-bench [OPTIONS]

OPTIONS:
  -t, --tokens-between-calls <N>  Max tokens for each Thought/Action step [default: 50]
  -d, --function-call-delay <MS>  Simulated function-call delay in milliseconds [default: 100]
      --use-prefix-cache          Enable caching of initial system prompt KV pages.
      --drop-tool-cache           Drop WebSearch documentation KV pages after use.
      --concurrent-calls          Simulate concurrent execution of tool calls.
  -h, --help                      Prints help information.";

const TEMPLATE_INTRO: &str = r#"
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

You are a helpful assistant that can use tools to answer questions. You have access to the following tools:
"#;

const WEB_SEARCH_DOCS: &str = r#"
- `WebSearch[query]`: Perform semantic search across web pages and return useful snippets.
"#;

const CODE_INTERPRETER_DOCS: &str = r#"
- `CodeInterpreter[code]`: Execute Python code in a sandboxed environment and return output.
"#;

const TEMPLATE_OUTRO: &str = r#"
To answer the user's question, break it into a sequence of steps:

Thought: reasoning about next action.
Action: ToolName[input]

After each action, you will receive an observation and continue until completion.
<|eot_id|>
"#;

const CACHE_FLAG_KEY: &str = "agent_react_bench_prefix_loaded_v1";
const CACHE_EXPORT_NAME: &str = "agent_react_bench_prefix_v1";
const CACHE_STATE_KEY: &str = "agent_react_bench_prefix_state_v1";

#[derive(Serialize, Deserialize)]
struct CachedPrefixState {
    token_ids: Vec<u32>,
    kv_page_last_len: usize,
}

#[derive(Clone)]
struct ToolTokenRange {
    start: usize,
    end: usize,
}

const PREDEFINED_SEQUENCES: [(&str, &str); 6] = [
    (
        "\nThought: I should gather a geography fact first.\nAction: WebSearch[highest mountain in North America]",
        "\nObservation: Denali is the highest mountain in North America.",
    ),
    (
        "\nThought: I should gather a literature fact.\nAction: WebSearch[author of Pride and Prejudice]",
        "\nObservation: Pride and Prejudice was written by Jane Austen.",
    ),
    (
        "\nThought: I should gather a chemistry fact.\nAction: WebSearch[chemical formula for caffeine]",
        "\nObservation: The chemical formula for caffeine is C8H10N4O2.",
    ),
    (
        "\nThought: Now run a quick computation.\nAction: CodeInterpreter[print(sum([i**2 for i in range(1, 51)]))]",
        "\nObservation: 42925",
    ),
    (
        "\nThought: Compute Fibonacci.\nAction: CodeInterpreter[a,b=0,1\nfor _ in range(19):\n a,b=b,a+b\nprint(a)]",
        "\nObservation: 6765",
    ),
    (
        "\nThought: Approximate pi.\nAction: CodeInterpreter[pi=3.0\nsign=1\nfor i in range(2,2001,2):\n pi += sign*4/(i*(i+1)*(i+2))\n sign *= -1\nprint(pi)]",
        "\nObservation: 3.1415921535897914",
    ),
];

fn get_tool_token_range(model: &inferlet::Model) -> ToolTokenRange {
    let tokenizer = model.get_tokenizer();
    let intro = tokenizer.tokenize(TEMPLATE_INTRO).len();
    let search = tokenizer.tokenize(WEB_SEARCH_DOCS).len();

    ToolTokenRange {
        start: intro,
        end: intro + search,
    }
}

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let tokens_between_calls: usize = args
        .value_from_str(["-t", "--tokens-between-calls"])
        .unwrap_or(50);
    let function_call_delay_ms: u64 = args
        .value_from_str(["-d", "--function-call-delay"])
        .unwrap_or(100);

    let use_prefix_cache = args.contains("--use-prefix-cache");
    let drop_tool_cache = args.contains("--drop-tool-cache");
    let concurrent_calls = args.contains("--concurrent-calls");

    let full_system_prompt = format!(
        "{}{}{}{}",
        TEMPLATE_INTRO, WEB_SEARCH_DOCS, CODE_INTERPRETER_DOCS, TEMPLATE_OUTRO
    );

    let model = inferlet::get_auto_model();
    let eos_tokens = model.eos_tokens();
    let web_search_range = get_tool_token_range(&model);
    let mut ctx = model.create_context();

    if use_prefix_cache {
        let queue = ctx.queue();
        if inferlet::store_get(CACHE_FLAG_KEY) == Some("true".to_string()) {
            let imported_pages = queue.import_kv_pages(CACHE_EXPORT_NAME);
            let state_json = inferlet::store_get(CACHE_STATE_KEY)
                .ok_or_else(|| anyhow!("Cache state missing for prefix cache"))?;
            let state: CachedPrefixState = serde_json::from_str(&state_json)?;
            ctx = Context::from_imported_state(
                &model,
                imported_pages,
                state.token_ids,
                state.kv_page_last_len,
            );
        } else {
            let mut prefill_ctx = model.create_context();
            prefill_ctx.fill(&full_system_prompt);
            prefill_ctx.flush().await;

            let state = CachedPrefixState {
                token_ids: prefill_ctx.get_token_ids().to_vec(),
                kv_page_last_len: prefill_ctx.get_kv_page_last_len(),
            };

            prefill_ctx
                .queue()
                .export_kv_pages(&prefill_ctx.kv_pages, CACHE_EXPORT_NAME);

            inferlet::store_set(CACHE_STATE_KEY, &serde_json::to_string(&state)?);
            inferlet::store_set(CACHE_FLAG_KEY, "true");

            ctx = prefill_ctx;
        }
    } else {
        ctx.fill(&full_system_prompt);
    }

    ctx.fill("<|start_header_id|>user<|end_header_id|>\n\nPerform a series of research and calculation tasks.");
    ctx.fill("<|eot_id|><|start_header_id|>assistant<|end_header_id|>");

    let stop_condition =
        stop_condition::max_len(tokens_between_calls).or(stop_condition::ends_with_any(eos_tokens));

    if concurrent_calls {
        let mut first_wave = Vec::new();
        for _ in 0..3 {
            let _ = ctx.generate(Sampler::greedy(), stop_condition.clone()).await;
            first_wave.push(inferlet::wstd::task::sleep(Duration::from_millis(
                function_call_delay_ms,
            )));
        }
        join_all(first_wave).await;

        for (_, observation) in PREDEFINED_SEQUENCES.iter().take(3) {
            ctx.fill("<|eot_id|><|start_header_id|>tool<|end_header_id|>");
            ctx.fill(observation);
            ctx.fill("<|eot_id|><|start_header_id|>assistant<|end_header_id|>");
        }

        if drop_tool_cache {
            let mut start_idx = web_search_range.start;
            if start_idx % 16 != 0 {
                start_idx += 16 - (start_idx % 16);
            }
            let mut end_idx = web_search_range.end;
            if end_idx % 16 != 0 {
                end_idx -= end_idx % 16;
            }

            if end_idx > start_idx {
                ctx.mask_token_range(start_idx, end_idx, true);
                ctx.drop_masked_kv_pages();
            }
        }

        let mut second_wave = Vec::new();
        for _ in 3..6 {
            let _ = ctx.generate(Sampler::greedy(), stop_condition.clone()).await;
            second_wave.push(inferlet::wstd::task::sleep(Duration::from_millis(
                function_call_delay_ms,
            )));
        }
        join_all(second_wave).await;

        for (_, observation) in PREDEFINED_SEQUENCES.iter().skip(3) {
            ctx.fill("<|eot_id|><|start_header_id|>tool<|end_header_id|>");
            ctx.fill(observation);
            ctx.fill("<|eot_id|><|start_header_id|>assistant<|end_header_id|>");
        }
    } else {
        for (i, (_, observation)) in PREDEFINED_SEQUENCES.iter().enumerate() {
            let _ = ctx.generate(Sampler::greedy(), stop_condition.clone()).await;
            inferlet::wstd::task::sleep(Duration::from_millis(function_call_delay_ms)).await;

            ctx.fill("<|eot_id|><|start_header_id|>tool<|end_header_id|>");
            ctx.fill(observation);
            ctx.fill("<|eot_id|><|start_header_id|>assistant<|end_header_id|>");

            if i == 2 && drop_tool_cache {
                let mut start_idx = web_search_range.start;
                if start_idx % 16 != 0 {
                    start_idx += 16 - (start_idx % 16);
                }
                let mut end_idx = web_search_range.end;
                if end_idx % 16 != 0 {
                    end_idx -= end_idx % 16;
                }

                if end_idx > start_idx {
                    ctx.mask_token_range(start_idx, end_idx, true);
                    ctx.drop_masked_kv_pages();
                }
            }
        }
    }

    println!("Benchmark complete.");
    Ok(())
}
