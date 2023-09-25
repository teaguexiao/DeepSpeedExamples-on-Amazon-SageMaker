# DeepSpeedChat on Amazon SageMaker
This repository contains demo code and Jupyter Notebook for using Sagemaker training job for finetuning LLM (lora and full parameter tunning), Sagemaker endpoints for inferencing LLM with vLLM, PagedAttention and continusing batching (rolling batch).

Use [DeepSpeedChat-training-on-SageMaker.ipynb](./DeepSpeedChat-training-on-SageMaker.ipynb) as the starting point for preparing the docker images, base model, training dataset and the whole training process.

The best way to run this notebook is through **SageMaker Notebook instance** (No GPU is needed, as GPU will be used through SageMaker training job), otherwise you will need to configure the access to Amazon S3, ECR (Elastic Container Registry) and Amazon SageMaker training job/endpoints.

# DeepSpeed Examples
This repository contains various examples including training, inference, compression, benchmarks, and applications that use [DeepSpeed](https://github.com/microsoft/DeepSpeed).

## 1. Applications
This folder contains end-to-end applications that use DeepSpeed to train and use cutting-edge models.

## 2. Training
There are several training and finetuning examples so please see the individual folders for specific instructions.

## 3. Inference
The DeepSpeed Huggingface inference [README](./inference/huggingface/README.md) explains how to get started with running DeepSpeed Huggingface inference examples.

## 4. Compression
Model compression examples.

## 5. Benchmarks
All benchmarks that use the DeepSpeed library are maintained in this folder.

# Build Pipeline Status
| Description | Status |
| ----------- | ------ |
| Integrations | [![nv-ds-chat](https://github.com/microsoft/DeepSpeed/actions/workflows/nv-ds-chat.yml/badge.svg?branch=master)](https://github.com/microsoft/DeepSpeed/actions/workflows/nv-ds-chat.yml) |

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
