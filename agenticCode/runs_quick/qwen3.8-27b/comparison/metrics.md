# Run metrics

- **sessions**: 4
## wall_time_s

- n: 4.0
- mean: 258.25
- std: 81.7
- min: 137.0
- p50: 291.5
- p90: 308.8
- p95: 310.9
- p99: 312.58
- max: 313.0

- **wall_time_total_h**: 0.29
## llm_calls_per_session (turns)

- n: 4.0
- mean: 84.0
- std: 26.67
- min: 44.0
- p50: 97.0
- p90: 97.7
- p95: 97.85
- p99: 97.97
- max: 98.0

## turns_vs_max_150

- mean_share_of_max: 0.56
- sessions_at_or_over_max: 0

## input_tokens_per_session

- n: 4.0
- mean: 443567.25
- std: 138655.28
- min: 237880.0
- p50: 497744.5
- p90: 529005.6
- p95: 534952.8
- p99: 539710.56
- max: 540900.0

## output_tokens_per_session

- n: 4.0
- mean: 20965.0
- std: 7348.1
- min: 10428.0
- p50: 22980.0
- p90: 26273.5
- p95: 26872.75
- p99: 27352.15
- max: 27472.0

- **total_input_tokens**: 1774269
- **total_output_tokens**: 83860
## max_prompt_vs_context_24000

- pct_of_limit_per_session: {'n': 4.0, 'mean': 85.89, 'std': 22.53, 'min': 58.99, 'p50': 85.69, 'p90': 106.49, 'p95': 109.83, 'p99': 112.51, 'max': 113.18}
- sessions_over_80pct: 3
- sessions_over_100pct: 1

## summarize

- calls_flagged_as_summarize (heuristic): 0
- sessions_with_summarize: 0

## failures

- sessions_goose_nonzero_exit: 0
- sessions_at_turn_limit: 0
- llm_requests_failed_or_timed_out: 9
- llm_requests_retried_after_429_503: 79
- llm_retries_total: 492
- max_queue_wait_s: 0.0
- total_queue_wait_s: 0.0
- outputs_truncated_finish_reason_length: 0
- sessions_with_truncated_output: 0
- sessions_with_tool_errors: 0

- **usage_reported_by_server_share**: 0.973