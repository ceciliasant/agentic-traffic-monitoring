# Run metrics

- **sessions**: 169
## wall_time_s

- n: 169.0
- mean: 280.36
- std: 21.23
- min: 146.0
- p50: 281.0
- p90: 300.0
- p95: 308.6
- p99: 334.44
- max: 354.0

- **wall_time_total_h**: 13.16
## llm_calls_per_session (turns)

- n: 169.0
- mean: 91.55
- std: 10.6
- min: 47.0
- p50: 96.0
- p90: 100.2
- p95: 101.6
- p99: 103.32
- max: 106.0

## turns_vs_max_150

- mean_share_of_max: 0.61
- sessions_at_or_over_max: 0

## input_tokens_per_session

- n: 169.0
- mean: 470361.67
- std: 73272.37
- min: 264538.0
- p50: 499383.0
- p90: 531513.2
- p95: 541668.8
- p99: 574882.84
- max: 633934.0

## output_tokens_per_session

- n: 169.0
- mean: 25588.31
- std: 3746.95
- min: 12927.0
- p50: 24648.0
- p90: 30376.4
- p95: 32413.0
- p99: 40688.8
- max: 42338.0

- **total_input_tokens**: 79491123
- **total_output_tokens**: 4324425
## max_prompt_vs_context_24000

- pct_of_limit_per_session: {'n': 169.0, 'mean': 94.58, 'std': 8.42, 'min': 63.71, 'p50': 94.21, 'p90': 104.48, 'p95': 106.84, 'p99': 114.31, 'max': 120.1}
- sessions_over_80pct: 159
- sessions_over_100pct: 34

## summarize

- calls_flagged_as_summarize (heuristic): 0
- sessions_with_summarize: 0

## failures

- sessions_goose_nonzero_exit: 0
- sessions_at_turn_limit: 0
- llm_requests_failed_or_timed_out: 493
- llm_requests_retried_after_429_503: 4648
- llm_retries_total: 25848
- max_queue_wait_s: 0.0
- total_queue_wait_s: 0.0
- outputs_truncated_finish_reason_length: 0
- sessions_with_truncated_output: 0
- output_tokens_in_truncated_requests: 0
- share_of_all_output_tokens_in_truncated: 0.0
- longest_single_request_s: 223.0
- requests_over_300s: 0
- prompt_tokens_over_context_limit_requests: 55
- sessions_with_tool_errors: 12

- **usage_reported_by_server_share**: 0.968