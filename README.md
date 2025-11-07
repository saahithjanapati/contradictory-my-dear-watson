# Contradictory, My Dear Watson Experiments

Experiments and notes for the Kaggle competition  
[`Contradictory, My Dear Watson`](https://www.kaggle.com/competitions/contradictory-my-dear-watson), which frames multi-lingual natural language inference as a three-class classification problem (entailment vs. contradiction vs. neutral).

> **Heads-up:** this is a TPU-enabled competition. The runs captured here were evaluated on a 1.2k-example validation split carved out of the provided training data.

## Repository Layout
- `bert.ipynb` – fine-tune `bert-base-uncased` for NLI.
- `qwen_icl_0_5B.ipynb`, `qwen_icl_1.7B.ipynb`, `qwen_icl_4B.ipynb` – in-context learning experiments with the Qwen 3 series.
- `qwen_sft_watson.ipynb` – supervised fine-tuning (LoRA) of Qwen 1.7B on the competition data.

## Getting Started
1. Install dependencies for the chosen notebook (PyTorch/Transformers for BERT, `transformers` + `peft` for Qwen runs).
2. Open the notebook in your environment of choice (Colab, Kaggle TPU, or a local GPU box).
3. Update data paths so the competition dataset is accessible, then run the cells top-to-bottom.

## Approaches & Key Takeaways
- **BERT fine-tune:** serves as a baseline; struggles with cross-lingual examples, but is quick to iterate.
- **ICL with Qwen:** prompt-engineering only; larger checkpoints bring consistent accuracy gains without gradient updates.
- **LoRA SFT:** adapting Qwen 1.7B with lightweight LoRA layers gives the best accuracy observed so far.

## Validation Scores
| Approach | Model / Notes | Accuracy |
| --- | --- | --- |
| Fine-tuning | `bert-base-uncased`, full-train | **0.6675** |
| ICL | Qwen 3 0.5B, curated prompt | 0.5520 |
| ICL | Qwen 3 1.7B, curated prompt | 0.7599 |
| ICL | Qwen 3 4B, curated prompt | 0.7855 |
| LoRA SFT | Qwen 3 1.7B, rank-8 adapters | **0.8498** |

## Next Ideas
- Try multilingual checkpoints (e.g., XLM-R/LLaMA variants) for better zero-shot coverage.
- Blend SFT + ICL by using SFT models as teachers for distilled prompts.
- Swap the validation strategy to cross-validation for a more reliable leaderboard estimate before TPU submission.



