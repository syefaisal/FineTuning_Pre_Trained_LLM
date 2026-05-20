# Fine-Tuning GPT-2 to Follow Instructions

Supervised fine-tuning (SFT) of GPT-2 Medium (355M) on an instruction-following dataset, built on top of a transformer implementation written from scratch in PyTorch.

---

## Overview

This project takes OpenAI's pretrained GPT-2 Medium weights and fine-tunes them using the Alpaca prompt format so the model can respond to structured instructions. The full pipeline covers weight loading, dataset preparation, training, inference, and automated LLM-based evaluation.

---

## Architecture

The transformer is implemented from scratch — no `transformers` library.

| Component | Implementation |
|---|---|
| Attention | Multi-head causal self-attention with scaled dot-product and upper-triangular mask |
| Normalization | Custom `LayerNorm` with learnable scale/shift parameters |
| Activation | Custom `GELU` approximation via tanh |
| Feed-forward | 2-layer MLP with 4× embedding expansion |
| Block | Pre-norm `TransformerBlock` with residual connections |
| Positional encoding | Learned absolute position embeddings |

**GPT-2 Medium config used for fine-tuning:**

```python
{
    "vocab_size":      50257,
    "context_length":  1024,
    "emb_dim":         1024,
    "n_heads":         16,
    "n_layers":        24,
    "drop_rate":       0.0,
    "qkv_bias":        True
}
```

---

## Dataset

- **1,100 instruction samples** in Alpaca format (`instruction`, `input`, `output`)
- Formatted using the standard Alpaca prompt template:

```
Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

- Custom `InstructionDataset` with pre-tokenization via `tiktoken` (BPE, GPT-2 vocabulary)
- Custom `collate_fn` handles variable-length sequences with padding and masked loss targets (padding tokens excluded from loss computation)

---

## Weight Loading

GPT-2 weights are downloaded from OpenAI's public checkpoint and loaded from TensorFlow format:

- TF checkpoint tensors are extracted and reshaped to match the PyTorch model layout
- QKV projection weights (stored as a single fused matrix in the TF ckpt) are split across `W_query`, `W_key`, `W_value`
- All weights are wrapped as `nn.Parameter` after shape verification

---

## Training

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 2
```

- Cross-entropy loss computed only on response tokens (instruction tokens masked out)
- Periodic evaluation on a held-out validation set during training
- Training and validation loss curves saved to `loss-plot.pdf`
- Final model saved as `gpt2-medium355M-sft.pth` (~1.6 GB)

---

## Inference

Generation uses temperature scaling and top-k filtering:

```python
generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=256,
    context_size=1024,
    eos_id=50256       # <|endoftext|> as stop token
)
```

---

## Evaluation

Model responses are scored automatically using **LLM-as-a-judge** via a locally running Ollama instance (Llama 3):

```python
prompt = (
    f"Given the input `{format_input(entry)}` "
    f"and correct output `{entry['output']}`, "
    f"score the model response `{entry['model_response']}`"
    f" on a scale from 0 to 100, where 100 is the best score. "
    f"Respond with the integer number only."
)
```

- Evaluated on 110 held-out test samples
- Deterministic scoring: `temperature=0`, `seed=123`, `num_ctx=2048`
- Results stored in `instruction-data-with-response.json`

---

## Stack

| Library | Role |
|---|---|
| `torch` | Model, training loop, optimizer |
| `tiktoken` | BPE tokenization (GPT-2 vocab) |
| `tensorflow` | Loading pretrained GPT-2 TF checkpoints |
| `ollama` | Local LLM judge for response scoring |
| `numpy` | Weight tensor manipulation during loading |

---

## File Structure

```
.
├── Finetune_to_Follow_Instructions.ipynb   # Main notebook
├── previous_chapters.py                    # GPT model, training utilities (Ch. 2–6)
├── gpt_download.py                         # GPT-2 weight downloader/loader
├── instruction-data.json                   # 1,100 instruction samples
├── instruction-data-with-response.json     # Test set with model responses
├── gpt2-medium355M-sft.pth                 # Fine-tuned model weights (1.6 GB)
└── loss-plot.pdf                           # Training/validation loss curves
```

---

## References

Based on Chapter 7 of [*Build a Large Language Model From Scratch*](https://www.manning.com/books/build-a-large-language-model-from-scratch) by Sebastian Raschka.
