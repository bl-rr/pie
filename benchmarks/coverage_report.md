# Benchmark Coverage Report

Freeze date: `2026-03-02`

## Backend versions

| Backend | Pinned | Latest |
| --- | --- | --- |
| vLLM | `0.6.0` | `0.16.0` |
| SGLang | `0.4.4` | `0.5.9` |

## Matrix

| Workload | Type | PIE | vLLM pinned | vLLM latest | SGLang pinned | SGLang latest |
| --- | --- | --- | --- | --- | --- | --- |
| `test_1_agent_react` | legacy | `test_1_agent_react_pie.py` | `test_1_agent_react_vllm_0_6_0.py` | `test_1_agent_react_vllm_0_16_0.py` | `test_1_agent_react_sglang_0_4_4.py` | `test_1_agent_react_sglang_0_5_9.py` |
| `test_2_agent_codeact` | legacy | `test_2_agent_codeact_pie.py` | `test_2_agent_codeact_vllm_0_6_0.py` | `test_2_agent_codeact_vllm_0_16_0.py` | `test_2_agent_codeact_sglang_0_4_4.py` | `test_2_agent_codeact_sglang_0_5_9.py` |
| `test_3_agent_swarm` | legacy | `test_3_agent_swarm_pie.py` | `test_3_agent_swarm_vllm_0_6_0.py` | `test_3_agent_swarm_vllm_0_16_0.py` | `test_3_agent_swarm_sglang_0_4_4.py` | `test_3_agent_swarm_sglang_0_5_9.py` |
| `test_4_agent_case_study` | legacy | `test_4_agent_case_study_pie.py` | `test_4_agent_case_study_vllm_0_6_0.py` | `test_4_agent_case_study_vllm_0_16_0.py` | `test_4_agent_case_study_sglang_0_4_4.py` | `test_4_agent_case_study_sglang_0_5_9.py` |
| `test_5_text_completion` | legacy | `test_5_text_completion_pie.py` | `test_5_text_completion_vllm_0_6_0.py` | `test_5_text_completion_vllm_0_16_0.py` | `test_5_text_completion_sglang_0_4_4.py` | `test_5_text_completion_sglang_0_5_9.py` |
| `test_6_prefix_tree` | legacy | `test_6_prefix_tree_pie.py` | `test_6_prefix_tree_vllm_0_6_0.py` | `test_6_prefix_tree_vllm_0_16_0.py` | `test_6_prefix_tree_sglang_0_4_4.py` | `test_6_prefix_tree_sglang_0_5_9.py` |
| `test_7_tot` | legacy | `test_7_tot_pie.py` | `test_7_tot_vllm_0_6_0.py` | `test_7_tot_vllm_0_16_0.py` | `test_7_tot_sglang_0_4_4.py` | `test_7_tot_sglang_0_5_9.py` |
| `test_8_rot` | legacy | `test_8_rot_pie.py` | `test_8_rot_vllm_0_6_0.py` | `test_8_rot_vllm_0_16_0.py` | `test_8_rot_sglang_0_4_4.py` | `test_8_rot_sglang_0_5_9.py` |
| `test_9_got` | legacy | `test_9_got_pie.py` | `test_9_got_vllm_0_6_0.py` | `test_9_got_vllm_0_16_0.py` | `test_9_got_sglang_0_4_4.py` | `test_9_got_sglang_0_5_9.py` |
| `test_10_skot` | legacy | `test_10_skot_pie.py` | `test_10_skot_vllm_0_6_0.py` | `test_10_skot_vllm_0_16_0.py` | `test_10_skot_sglang_0_4_4.py` | `test_10_skot_sglang_0_5_9.py` |
| `test_11_cache` | legacy | `test_11_cache_pie.py` | `test_11_cache_vllm_0_6_0.py` | `test_11_cache_vllm_0_16_0.py` | `test_11_cache_sglang_0_4_4.py` | `test_11_cache_sglang_0_5_9.py` |
| `test_12_ebnf` | legacy | `test_12_ebnf_pie.py` | `test_12_ebnf_vllm_0_6_0.py` | `test_12_ebnf_vllm_0_16_0.py` | `test_12_ebnf_sglang_0_4_4.py` | `test_12_ebnf_sglang_0_5_9.py` |
| `test_13_specdec` | legacy | `test_13_specdec_pie.py` | `test_13_specdec_vllm_0_6_0.py` | `test_13_specdec_vllm_0_16_0.py` | omitted: No equivalent n-gram speculative decoding API in SGLang OpenAI-compatible server mode. | omitted: No equivalent n-gram speculative decoding API in SGLang OpenAI-compatible server mode. |
| `test_14_beamsearch` | legacy | `test_14_beamsearch_pie.py` | `test_14_beamsearch_vllm_0_6_0.py` | `test_14_beamsearch_vllm_0_16_0.py` | omitted: No tested SGLang endpoint parity for vLLM `use_beam_search` request contract. | omitted: No tested SGLang endpoint parity for vLLM `use_beam_search` request contract. |
| `test_15_attnsink` | legacy | `test_15_attnsink_pie.py` | `test_15_attnsink_vllm_0_6_0.py` | `test_15_attnsink_vllm_0_16_0.py` | `test_15_attnsink_sglang_0_4_4.py` | `test_15_attnsink_sglang_0_5_9.py` |
| `test_16_parallel_generation` | new-example | `test_16_parallel_generation_pie.py` | `test_16_parallel_generation_vllm_0_6_0.py` | `test_16_parallel_generation_vllm_0_16_0.py` | `test_16_parallel_generation_sglang_0_4_4.py` | `test_16_parallel_generation_sglang_0_5_9.py` |

## Omitted Cells

- `test_13_specdec` sglang pinned (0.4.4): No equivalent n-gram speculative decoding API in SGLang OpenAI-compatible server mode.
- `test_13_specdec` sglang latest (0.5.9): No equivalent n-gram speculative decoding API in SGLang OpenAI-compatible server mode.
- `test_14_beamsearch` sglang pinned (0.4.4): No tested SGLang endpoint parity for vLLM `use_beam_search` request contract.
- `test_14_beamsearch` sglang latest (0.5.9): No tested SGLang endpoint parity for vLLM `use_beam_search` request contract.

## Summary

- Workloads: `16`
- Matrix cells: `64`
- Implemented: `60`
- Omitted: `4`
