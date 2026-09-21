# Serverless GenAI Fine-Tuning & Deployment Pipeline on AWS

An end-to-end MLOps/LLMOps project demonstrating fine-tuning, serverless inference, and monitoring for a large language model on AWS — built as a portfolio piece for GenAI/LLMOps roles.

## Overview

This project fine-tunes **microsoft/Phi-3-mini-4k-instruct** using **LoRA/QLoRA** on Amazon SageMaker, then deploys the resulting model behind a fully serverless, production-style inference stack (API Gateway → Lambda → SageMaker endpoint), with request/response logging to DynamoDB for observability.

It covers the full lifecycle a GenAI/LLMOps engineer is expected to own:

- Data preparation and fine-tuning
- Custom model packaging for an unsupported architecture
- Cloud deployment and infrastructure wiring
- Logging and monitoring
- Cost-aware validation and cleanup

## Architecture

```
Client
  │
  ▼
API Gateway  ──►  AWS Lambda  ──►  SageMaker Real-Time Endpoint (ml.g5.xlarge)
                       │
                       ▼
                  DynamoDB (request/response logs)
```

- **Fine-tuning**: SageMaker Training Job (LoRA/QLoRA) on the Dolly-15k dataset
- **Inference**: SageMaker real-time GPU endpoint with a custom `inference.py`
- **API layer**: API Gateway + Lambda as a lightweight invocation/orchestration layer
- **Observability**: DynamoDB table logging every request and response for auditability



## Tech Stack

| Layer | Tools |
|---|---|
| Model | `microsoft/Phi-3-mini-4k-instruct` |
| Fine-tuning | Hugging Face `transformers`, `peft`, `trl`, LoRA/QLoRA |
| Dataset | `databricks/databricks-dolly-15k` |
| Training/Hosting | Amazon SageMaker (`ml.g5.xlarge`) |
| API | AWS Lambda, API Gateway |
| Logging | Amazon DynamoDB |
| IAM/Infra | AWS IAM, S3 |
| Dev environment | Windows, Miniconda, VS Code, Jupyter |
| Region | `ap-south-1` (Mumbai) |

## Key Engineering Challenges & Solutions

Fine-tuning and deploying Phi-3 on SageMaker surfaced several non-obvious issues, each of which required a targeted fix:

1. **LoRA target modules differ by architecture**
   Phi-3 uses a fused QKV attention layout, so LoRA must target `["qkv_proj", "o_proj"]` instead of the Llama-style `["q_proj", "v_proj"]`.

2. **Gradient checkpointing + PEFT compatibility**
   Enabling gradient checkpointing with a PEFT-wrapped model requires explicitly calling `model.enable_input_require_grads()` after `get_peft_model()`, or gradients silently fail to flow to the LoRA adapters.

3. **TRL API versioning**
   Newer versions of `trl` moved `max_seq_length` and related training params from `TrainingArguments` into `SFTConfig`.

4. **IAM permissions boundary gotcha**
   A permissions boundary scoped to `AmazonS3FullAccess` silently blocked ECR image pulls during endpoint deployment, even though broader policies were attached to the role. Fixed via `iam.delete_role_permissions_boundary()`.

5. **Custom inference handler required**
   SageMaker's default Hugging Face Inference Toolkit doesn't support Phi-3's architecture out of the box. A custom `inference.py` with explicit `model_fn` / `predict_fn` was written and bundled under a `code/` subfolder, referenced via the `SAGEMAKER_PROGRAM` and `SAGEMAKER_SUBMIT_DIRECTORY` environment variables.

6. **Transformers API drift**
   `DynamicCache.get_max_length` was renamed in newer `transformers` releases. Resolved with a monkey-patch: `DynamicCache.get_max_length = DynamicCache.get_max_cache_shape`.

7. **SageMaker SDK v3 breaking change**
   The `HuggingFace` estimator import broke under SageMaker SDK v3.21.0+. Pinned to `sagemaker<3.0.0` for stability.

## Cost-Conscious Validation Pattern

Given the cost of GPU endpoints, this project follows a **validate-first** approach: before deploying any live endpoint, inference logic is first run and verified as a cheap SageMaker training/processing job. This fails fast on packaging or code errors without incurring sustained GPU endpoint costs.

## Project Lifecycle

1. **Data prep** — Load and format the Dolly-15k dataset for instruction fine-tuning
2. **Fine-tuning** — LoRA/QLoRA training job on SageMaker
3. **Validation** — Cheap dry-run of inference logic as a training/processing job
4. **Deployment** — Real-time SageMaker endpoint (`ml.g5.xlarge`) with custom inference code
5. **API layer** — Lambda function wired to API Gateway for external invocation
6. **Logging** — DynamoDB table capturing every inference request/response
7. **Cleanup** — Manual teardown of billable resources via the AWS Console to stop ongoing charges


## Status

✅ Fine-tuning complete
✅ Endpoint deployed and tested end-to-end
✅ Logging pipeline verified
✅ AWS resources torn down to stop billing (endpoint, GPU instances, and associated resources removed post-demo)

> Note: The live endpoint is not currently running to avoid ongoing GPU costs.


---

**Author**: Shubham
**Purpose**: Portfolio project demonstrating end-to-end GenAI/LLMOps competency on AWS
