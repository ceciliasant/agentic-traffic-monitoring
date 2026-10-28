# Agentic vs classic pipeline

Reference for fidelity = classic run (same models, same data). Ground truth for accuracy = observed flows. Verdict rules are a reference, not ground truth.

## sessions

| metric | value |
|---|---|
| agent_sessions | 169 |
| hours_compared | 168 |
| repeated_sessions | 1 |

## coverage

| metric | value |
|---|---|
| mean_coverage_vs_classic | 0.7811 |
| sessions_with_full_coverage | 13 |
| sessions_with_missing_nodes | 156 |
| sessions_with_extra_nodes | 0 |

## input fidelity

| metric | value |
|---|---|
| baseline_predict_calls | 4742 |
| calls_with_wrong_payload | 390 |
| wrong_payload_rate | 0.0822 |
| get_flows_wrong_hour | 0 |
| calls_to_nodes_that_should_not_be_touched | 0 |

## prediction fidelity

| metric | value |
|---|---|
| node_hours_compared | 1570 |
| exact_match_rate | 0.9981 |
| mean_abs_diff | 0.9193 |
| max_abs_diff | 595.15 |

## forecast accuracy vs actual

| metric | value |
|---|---|
| node_hours_with_actual | 1560 |
| mean_smape_classic | 0.0626 |
| mean_smape_agent | 0.0628 |
| mean_paired_diff_agent_minus_classic | 0.0003 |
| share_agent_worse | 0.0019 |

## evaluation fidelity

| metric | value |
|---|---|
| sessions_with_evaluation | 168 |
| mean_eval_coverage_vs_classic | 0.7024 |
| check_deviation_calls | 4003 |
| check_deviation_calls_with_wrong_args | 0 |

## verdicts

| metric | value |
|---|---|
| node_level_agreement | 0.0149 |
| node_level_agreement_when_inputs_correct | 0.0191 |
| node_level_cohen_kappa | 0.0072 |
| overall_verdict_match_rate | 0.0059 |
| summary_counts_match_rate | 0.0059 |
| sessions_without_log_run_summary | 168 |
| sessions_without_log_final_ditto | 168 |

## failure modes and cost

| metric | value |
|---|---|
| tool_errors | 313 |
| repeated_calls | 13776 |
| unanswered_calls | 0 |
| mean_tool_calls_agent | 144.787 |
| mean_tool_calls_classic | 63.787 |
| mean_duration_s_agent_trace | 357.574 |
| mean_duration_s_classic | 17.6436 |
| rate_limit_errors | 0 |
| delegate_parameter_errors | 0 |
| sessions_where_coordinator_called_pipeline_tools_directly | 169 |
| subagents_using_load_instead_of_tools | 0 |
| delegate_calls | 0 |
