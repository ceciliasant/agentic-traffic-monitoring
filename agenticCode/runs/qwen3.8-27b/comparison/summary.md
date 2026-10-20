# Agentic vs classic pipeline

Reference for fidelity = classic run (same models, same data). Ground truth for accuracy = observed flows. Verdict rules are a reference, not ground truth.

## sessions

| metric | value |
|---|---|
| agent_sessions | 169 |
| hours_compared | 169 |
| repeated_sessions | 0 |

## coverage

| metric | value |
|---|---|
| mean_coverage_vs_classic | 0.8649 |
| sessions_with_full_coverage | 96 |
| sessions_with_missing_nodes | 73 |
| sessions_with_extra_nodes | 0 |

## input fidelity

| metric | value |
|---|---|
| baseline_predict_calls | 1891 |
| calls_with_wrong_payload | 16 |
| wrong_payload_rate | 0.0085 |
| get_flows_wrong_hour | 2 |
| calls_to_nodes_that_should_not_be_touched | 0 |

## prediction fidelity

| metric | value |
|---|---|
| node_hours_compared | 1751 |
| exact_match_rate | 1.0 |
| mean_abs_diff | 0.0 |
| max_abs_diff | 0.0 |

## forecast accuracy vs actual

| metric | value |
|---|---|
| node_hours_with_actual | 1739 |
| mean_smape_classic | 0.0635 |
| mean_smape_agent | 0.0635 |
| mean_paired_diff_agent_minus_classic | 0.0 |
| share_agent_worse | 0.0 |

## evaluation fidelity

| metric | value |
|---|---|
| sessions_with_evaluation | 168 |
| mean_eval_coverage_vs_classic | 0.7907 |
| check_deviation_calls | 1630 |
| check_deviation_calls_with_wrong_args | 8 |

## verdicts

| metric | value |
|---|---|
| node_level_agreement | 0.7821 |
| node_level_agreement_when_inputs_correct | 0.8933 |
| node_level_cohen_kappa | 0.2121 |
| overall_verdict_match_rate | 0.9053 |
| summary_counts_match_rate | 0.4793 |
| sessions_without_log_run_summary | 5 |
| sessions_without_log_final_ditto | 5 |

## failure modes and cost

| metric | value |
|---|---|
| tool_errors | 14 |
| repeated_calls | 202 |
| unanswered_calls | 0 |
| mean_tool_calls_agent | 59.9822 |
| mean_tool_calls_classic | 63.787 |
| mean_duration_s_agent_trace | 265.3018 |
| mean_duration_s_classic | 17.6499 |
| rate_limit_errors | 0 |
| delegate_parameter_errors | 72 |
| sessions_where_coordinator_called_pipeline_tools_directly | 0 |
| subagents_using_load_instead_of_tools | 783 |
| delegate_calls | 4122 |
