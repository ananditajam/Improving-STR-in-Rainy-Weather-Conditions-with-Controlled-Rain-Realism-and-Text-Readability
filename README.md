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

- Train: [[Google Drive link](https://drive.google.com/drive/folders/1v0aytj3HjEwAPz5Hr9Hnp7Ghv2dYFz7r?usp=drive_link)]
- Validation: [Google Drive link]
- Test: [Google Drive link]

## Fine-Tuning

## STR Models

The experiments in this work use publicly available pretrained Scene Text
Recognition (STR) models. The original implementations and pretrained weights
can be obtained from their respective official repositories:

- PARSeq — [[Official GitHub repository](https://github.com/baudm/parseq)]
- MAERec — [[Official GitHub repository](https://github.com/Mountchicken/Union14M)]

The pretrained models were fine-tuned on the rainy-weather datasets described
in the paper. 

## Evaluation
Evaluation metrics include WER, CER, Precision, Recall, and F1-Score.
- Word Error Rate (WER)
- Character Error Rate (CER)
- Accuracy
- Precision
- Recall
- F1-score
