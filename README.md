# Explainable Movie Recommendation System — Qwen Zero-shot vs QLoRA Fine-tuned

A comparison of **zero-shot prompting** vs **QLoRA fine-tuning** for explainable movie recommendations, using Qwen models on the Amazon Movies & TV dataset. Each variant asks the model to output a structured JSON recommendation with a natural-language explanation.

---

## Models

| Folder | Model |
|---|---|
| `qwen2.5-1.5B-reccsys/` | `Qwen/Qwen2.5-1.5B` |
| `qwen2.5-3B-reccsys/` | `Qwen/Qwen2.5-3B-Instruct` |
| `qwen3-4B-reccsys/` | `Qwen/Qwen3-4B-Instruct-2507` |

Each folder contains a `.py` script and a matching `.ipynb` notebook (generated from Google Colab).

---

## How It Works

### Dataset
- **Amazon Movies & TV 5-core** — only users/items with at least 5 reviews
- Loads 200k reviews + 500k metadata rows, merges titles, filters to `rating >= 4`
- Keeps 500 sampled users who each have at least 6 liked items

### User Profile Construction
For each user:
- **History** — last 5 liked movies (shown to the model as context)
- **Target item** — the user's next liked movie (ground truth)
- **Candidates** — target + 4 random distractor titles (model must pick the right one)
- **Target review** — user's actual review text (ground truth explanation)

### Prompt Format
The model is asked to respond in strict JSON:
```json
{"recommendation": "<movie title>", "explanation": "<reason based on user history>"}
```

### Pipeline (per model)
1. Load & preprocess dataset
2. Build user profiles and train/test split (80/20)
3. **Zero-shot inference** — base model, no training
4. Fine-tune with **QLoRA** on the training split via `SFTTrainer`
5. **Fine-tuned inference** on the same test set
6. Evaluate and print a comparison table

---

## QLoRA Fine-tuning Setup

| Hyperparameter | Value |
|---|---|
| Quantization | 4-bit NF4 (double quant) |
| LoRA rank (`r`) | 16 |
| LoRA alpha | 32 |
| Target modules | `q/k/v/o_proj`, `gate/up/down_proj` |
| LoRA dropout | 0.05 |
| Epochs | 2 |
| Batch size | 1 (+ gradient accumulation ×8) |
| Learning rate | 2e-4 |
| Optimizer | `paged_adamw_8bit` |
| Max sequence length | 512 tokens |

Designed to run on a **free Google Colab T4 GPU (15 GB)**. Only ~1–2% of parameters are trained via the LoRA adapters.

---

## Evaluation Metrics

**Recommendation**
- `Hit@1` / `Precision@1` — did the model pick the correct item from 5 candidates?
- `NDCG@1` — normalized discounted cumulative gain (equals Precision@1 for single-item output)

**Explanation quality** (generated explanation vs actual user review)
- `BLEU` — n-gram overlap
- `ROUGE-1` — unigram overlap
- `ROUGE-L` — longest common subsequence

**Parse Fail Rate** — fraction of outputs that were not valid JSON

---

## Results Summary

Fine-tuning with QLoRA on ~400 samples teaches the model:
1. The exact JSON output format (reduces parse failures)
2. How to connect user watch history to item features in the explanation
3. Domain vocabulary — movie genres, styles, themes

| | Zero-shot | Fine-tuned (QLoRA) |
|---|---|---|
| Training required | No | Yes (~few minutes on T4) |
| Parameters updated | 0 | ~1–2% (LoRA adapters only) |
| Output format | Often inconsistent | Consistently valid JSON |
| Recommendation accuracy | Moderate | Higher |
| Explanation quality | Generic | Grounded in user history |

---

## Requirements

```bash
pip install datasets trl peft bitsandbytes nltk rouge_score transformers torch
```

Run the notebooks in **Google Colab** (GPU runtime recommended) or execute the `.py` scripts locally with a CUDA-capable GPU.

---

## References

- [Qwen models — Hugging Face](https://huggingface.co/Qwen)
- [Amazon Reviews 2023 Dataset — McAuley Lab](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [PEFT library](https://github.com/huggingface/peft)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
