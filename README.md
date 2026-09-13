# Improving-STR-in-Rainy-Weather-Conditions-with-Controlled-Rain-Realism-and-Text-Readability
# Improving Scene Text Recognition in Rainy Weather Conditions with Controlled Rain Realism and Text Readability

This repository contains the datasets, rain-generation code, evaluation scripts,
experimental results, and fine-tuned checkpoints associated with our paper:

**"Improving Scene Text Recognition in Rainy Weather Conditions with Controlled
Rain Realism and Text Readability"**

Published in *Array*.

---
## Contents
- `rain_generation/` — Code for the proposed controlled rain-generation pipeline
- `data/` — Links to generated and real rainy datasets
- `benchmarks/` — STR benchmarking results
- `finetuning/` — Scripts used to fine-tune pretrained STR models
- `evaluation/` — Scripts/notebooks for computing evaluation metrics
- `checkpoints/` — Fine-tuned model checkpoints
---
## Datasets

### Generated Rainy Dataset

Rainy images generated using the proposed rain-generation framework.

- Train: [Google Drive link]
- Validation: [Google Drive link]
- Test: [Google Drive link]

### Real Rainy Dataset

Real rainy scene-text images used for evaluation.

- Train: [Google Drive link]
- Validation: [Google Drive link]
- Test: [Google Drive link]


We do not redistribute the original pretrained model weights in this repository.

---

## Fine-Tuning

For the fine-tuning experiments, pretrained STR models were initialized using
their publicly available pretrained weights and subsequently fine-tuned using
the rainy datasets described in the paper.

Fine-tuning scripts/configurations used in our experiments are provided in:

`finetuning/`


---

## Evaluation

The evaluation scripts are provided in:

`evaluation/`

The following Scene Text Recognition metrics are supported:

- Word Error Rate (WER)
- Character Error Rate (CER)
- Accuracy
- Precision
- Recall
- F1-score
