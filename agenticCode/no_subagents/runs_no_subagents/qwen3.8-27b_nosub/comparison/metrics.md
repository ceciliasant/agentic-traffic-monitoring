# Run metrics

- **sessions**: 169
## wall_time_s

- n: 169.0
- mean: 427.84
- std: 208.0
- min: 91.0
- p50: 399.0
- p90: 710.8
- p95: 942.6
- p99: 1125.52
- max: 1178.0

- **wall_time_total_h**: 20.08
## llm_calls_per_session (turns)

- n: 169.0
- mean: 45.96
- std: 15.61
- min: 30.0
- p50: 43.0
- p90: 69.0
- p95: 79.0
- p99: 93.64
- max: 98.0

## turns_vs_max_150

- mean_share_of_max: 0.306
- sessions_at_or_over_max: 0

## input_tokens_per_session

- n: 169.0
- mean: 582667.68
- std: 212641.8
- min: 316824.0
- p50: 538930.0
- p90: 874173.0
- p95: 1025838.2
- p99: 1245061.68
- max: 1309275.0

## output_tokens_per_session

- n: 169.0
- mean: 51263.05
- std: 26411.69
- min: 8466.0
- p50: 48253.0
- p90: 85485.0
- p95: 115796.4
- p99: 140407.88
- max: 150990.0

- **total_input_tokens**: 98470838
- **total_output_tokens**: 8663455
## max_prompt_vs_context_24000

- pct_of_limit_per_session: {'n': 169.0, 'mean': 79.31, 'std': 1.06, 'min': 69.35, 'p50': 79.49, 'p90': 80.11, 'p95': 80.18, 'p99': 80.37, 'max': 80.48}
- sessions_over_80pct: 25
- sessions_over_100pct: 0

## summarize

- calls_flagged_as_summarize (heuristic): 338
- sessions_with_summarize: 168

## failures

- sessions_goose_nonzero_exit: 0
- sessions_at_turn_limit: 0
- llm_requests_failed_or_timed_out: 1
- llm_requests_retried_after_429_503: 0
- llm_retries_total: 0
- max_queue_wait_s: 0.0
- total_queue_wait_s: 0.0
- outputs_truncated_finish_reason_length: 0
- sessions_with_truncated_output: 0
- output_tokens_in_truncated_requests: 0
- share_of_all_output_tokens_in_truncated: 0.0
- longest_single_request_s: 908.5
- requests_over_300s: 1
- prompt_tokens_over_context_limit_requests: 0
- sessions_with_tool_errors: 7

- **usage_reported_by_server_share**: 1.0