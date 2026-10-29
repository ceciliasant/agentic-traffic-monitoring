# Agentic Artificial Intelligence for Traffic Monitoring in a Mobility Digital Twin

This repository contains the implementation, experiments, and analysis developed for the dissertation *Agentic Artificial Intelligence for Traffic Monitoring in a Mobility Digital Twin*.

The project investigates the use of large language models (LLMs) to orchestrate an hourly traffic monitoring workflow within a mobility digital twin, comparing agentic execution against a deterministic reference implementation.

## Repository Structure

- **`agenticCode/`**  
  Agentic monitoring implementation, MCP tools, deterministic reference runner, experiment execution scripts, and evaluation analysis.

- **`dataAnalysis/`**  
  Data preparation, cleaning, and exploratory analysis scripts and notebooks for historical traffic observations.

- **`mlTrainPipeline/`**  
  Machine learning training pipeline, including model development, evaluation, and supporting infrastructure.

## Project Workflow

1. Historical traffic data are prepared and explored in `dataAnalysis/`.
2. Traffic forecasting models are developed and evaluated in `mlTrainPipeline/`.
3. The agentic and deterministic monitoring workflows are executed using `agenticCode/`.
4. Experimental outputs are compared to assess workflow correctness, classification agreement, and operational performance.

Further implementation and execution details are provided in the README files within each directory.