# Run metrics

- **sessions**: 169
## wall_time_s

- n: 169.0
- mean: 648.9
- std: 2714.87
- min: 299.0
- p50: 347.0
- p90: 384.4
- p95: 437.6
- p99: 5217.6
- max: 34657.0

- **wall_time_total_h**: 30.46
## llm_calls_per_session (turns)

- n: 169.0
- mean: 147.86
- std: 13.56
- min: 48.0
- p50: 150.0
- p90: 150.0
- p95: 150.0
- p99: 150.32
- max: 151.0

## turns_vs_max_150

- mean_share_of_max: 0.986
- sessions_at_or_over_max: 164

## input_tokens_per_session

- n: 169.0
- mean: 1820240.89
- std: 175244.81
- min: 553814.0
- p50: 1844833.0
- p90: 1888430.2
- p95: 1896651.8
- p99: 1909936.48
- max: 1927619.0

## output_tokens_per_session

- n: 169.0
- mean: 39880.33
- std: 11646.77
- min: 7261.0
- p50: 38692.0
- p90: 42574.6
- p95: 46974.2
- p99: 77550.48
- max: 150949.0

- **total_input_tokens**: 307620711
- **total_output_tokens**: 6739775
## max_prompt_vs_context_24000

- pct_of_limit_per_session: {'n': 169.0, 'mean': 79.75, 'std': 0.72, 'min': 71.77, 'p50': 79.86, 'p90': 80.07, 'p95': 80.12, 'p99': 80.19, 'max': 80.22}
- sessions_over_80pct: 42
- sessions_over_100pct: 0

## summarize

- calls_flagged_as_summarize (heuristic): 498
- sessions_with_summarize: 167

## failures

- sessions_goose_nonzero_exit: 2
- sessions_at_turn_limit: 164
- llm_requests_failed_or_timed_out: 4
- llm_requests_retried_after_429_503: 0
- llm_retries_total: 0
- max_queue_wait_s: 0.0
- total_queue_wait_s: 0.0
- outputs_truncated_finish_reason_length: 2
- sessions_with_truncated_output: 2
- output_tokens_in_truncated_requests: 235872
- share_of_all_output_tokens_in_truncated: 0.035
- longest_single_request_s: 41472.6
- requests_over_300s: 6
- prompt_tokens_over_context_limit_requests: 0
- sessions_with_tool_errors: 163

- **usage_reported_by_server_share**: 1.0