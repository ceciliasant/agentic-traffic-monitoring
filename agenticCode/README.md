# Agentic Traffic Monitoring

This directory contains the implementation and evaluation infrastructure for the agentic traffic monitoring system developed in the dissertation.

## Objective

The experiments investigate whether LLM-based orchestration can reproduce the behaviour of a deterministic traffic monitoring workflow while maintaining correctness and traceability.

Beyond forecasting accuracy, the evaluation examines whether the agentic system:

- Invokes the appropriate tools.
- Provides the correct inputs, timestamps, and sensor-direction identifiers.
- Generates and evaluates the expected predictions.
- Produces classifications consistent with the deterministic reference.
- Handles tool interactions without unnecessary or incorrect operations.
- Completes monitoring sessions within the available execution time.

## Main Components

- **`config.yaml`**: Goose configuration for connecting the agent to the MCP tools and runtime environment.
- **`mcp_server.py`**: MCP server exposing traffic data retrieval, forecasting, evaluation, and monitoring operations.
- **`classic_runner.py`**: Deterministic reference implementation using the same MCP tools without LLM-based orchestration.
- **`run_models.sh`**: Executes the hierarchical agentic configurations and their comparison and metrics procedures.
- **`compare.py`**: Compares agentic and deterministic outputs, including prediction coverage, input correctness, classification agreement, and execution errors.
- **`metrics.py`**: Aggregates operational measurements such as execution time, LLM requests, token consumption, and failures.
- **`llm_proxy.py`**: Records LLM request information and applies automatic retries with backoff for HTTP 429/503 responses.
- **`preflight.sh`**: Checks configuration, dependencies, service connectivity, and other prerequisites before experimental execution.
- **`pick_window.py`**: Retrieves contextual information, including weather, and holidays, for a specified period.
- **`analysis.ipynb`**: Processes and visualizes the collected experimental results.
- **`Dockerfile` and `docker-compose.yaml`**: Define the containerized execution environment.

## Experimental Workflow

1. The deterministic reference pipeline is executed using `classic_runner.py`.
2. Agentic experiments are executed using the selected LLM and delegation configuration.
3. The agent interacts with the MCP tools to retrieve observations, generate predictions, evaluate deviations, and classify traffic conditions.
4. Predictions, tool execution traces, LLM requests, and session timings are recorded.
5. `compare.py` compares the agentic outputs against the deterministic reference.
6. `metrics.py` and `analysis.ipynb` process the results for correctness, agreement, and operational performance analysis.

## Environment Configuration

The experimental environment is configured through a `.env` file in the `agenticCode/` directory. This file contains the evaluation period, external service endpoints, and authentication credentials required by the traffic monitoring workflow.

Create a `.env` file containing the following variables:

```dotenv
# Historical replay interval
TEST_START=2024-09-23T00:00:00
TEST_END=2024-09-29T23:00:00

# LLM inference service
OPENWEBUI_API_KEY=<your_api_key>

# Digital twin authentication
USER_DITTO=<your_username>
PASS_DITTO=<your_password>
TOKEN_URL=<authentication_endpoint>
CLIENT_ID=<client_identifier>

# Historical traffic data
VCI_HISTORIC_DITTO_URL=<historical_ditto_api_url>

# Object storage
AWS_ENDPOINT_URL=<object_storage_endpoint>
```

The following variables are also used to configure individual experiments:

- `GOOSE_MODEL`: Selects the language model used by the agentic system.
- `RUN_MODE`: Selects the execution mode.
- `PREDICTION_THING_PREFIX`: Defines the storage prefix for predictions, allowing experiment results to remain separate.

Before running the experiments, execute `preflight.sh` to check the configuration, required files, Docker environment, and connectivity to the external services.