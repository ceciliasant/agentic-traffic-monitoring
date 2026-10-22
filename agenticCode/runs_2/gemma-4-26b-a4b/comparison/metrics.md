# Run metrics

- **sessions**: 169
## wall_time_s

- n: 169.0
- mean: 370.53
- std: 312.35
- min: 202.0
- p50: 290.0
- p90: 361.2
- p95: 1158.2
- p99: 1705.2
- max: 2508.0

- **wall_time_total_h**: 17.39
## llm_calls_per_session (turns)

- n: 169.0
- mean: 124.15
- std: 20.38
- min: 59.0
- p50: 123.0
- p90: 149.2
- p95: 152.0
- p99: 158.32
- max: 160.0

## turns_vs_max_150

- mean_share_of_max: 0.828
- sessions_at_or_over_max: 17

## input_tokens_per_session

- n: 169.0
- mean: 1021276.6
- std: 320888.77
- min: 335071.0
- p50: 1000812.0
- p90: 1444805.2
- p95: 1498420.0
- p99: 1623231.2
- max: 1676950.0

## output_tokens_per_session

- n: 169.0
- mean: 51162.98
- std: 26964.06
- min: 26755.0
- p50: 44574.0
- p90: 50823.2
- p95: 141268.8
- p99: 152278.08
- max: 205814.0

- **total_input_tokens**: 172595745
- **total_output_tokens**: 8646543
## max_prompt_vs_context_24000

- pct_of_limit_per_session: {'n': 169.0, 'mean': 131.83, 'std': 15.19, 'min': 76.92, 'p50': 130.57, 'p90': 149.17, 'p95': 155.82, 'p99': 167.78, 'max': 186.69}
- sessions_over_80pct: 168
- sessions_over_100pct: 166

## summarize

- calls_flagged_as_summarize (heuristic): 0
- sessions_with_summarize: 0

## failures

- sessions_goose_nonzero_exit: 0
- sessions_at_turn_limit: 17
- llm_requests_failed_or_timed_out: 14
- llm_requests_retried_after_429_503: 886
- llm_retries_total: 3781
- max_queue_wait_s: 0.0
- total_queue_wait_s: 0.0
- outputs_truncated_finish_reason_length: 10
- sessions_with_truncated_output: 10
- sessions_with_tool_errors: 102

- **usage_reported_by_server_share**: 0.999