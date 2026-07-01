# Fine-Tuning GPT-2 to Follow Instructions

Supervised fine-tuning (SFT) of **GPT-2 Medium (355M)** on an instruction-following dataset, built on top of a transformer implemented **from scratch in PyTorch** — no `transformers` library.

The full pipeline covers pretrained-weight loading, dataset preparation, masked-loss training, generation, and automated LLM-as-a-judge evaluation.

> **Status — what this repo actually is:** a reproducible, end-to-end *training + evaluation* notebook. It produces a fine-tuned checkpoint and an automated quality score. It does **not** yet include serving, a demo UI, quantization, or deployment — those are tracked honestly in the [Roadmap / TODO](#roadmap--todo) section rather than claimed as features.

---

## Results at a glance

| Metric | Value |
|---|---|
| Base model | GPT-2 Medium, 355M params (OpenAI pretrained weights) |
| Training data | 1,100 Alpaca-format instructions (935 train / 110 test / 55 val) |
| Training hardware | **CPU** (MPS unavailable on the torch build used) |
| Training time | **13.05 minutes**, 2 epochs |
| Train loss | 2.637 → ~0.30 |
| Validation loss | 2.626 → 0.632 |
| LLM-judge avg score | **49.45 / 100** (Llama 3 via Ollama, 110 test samples) |
| Output checkpoint | `gpt2-medium355M-sft.pth` (~1.6 GB) |

The loss curves are saved in [`loss-plot.pdf`](loss-plot.pdf).

---

## Architecture

The transformer is written from scratch in [`previous_chapters.py`](previous_chapters.py).

```
                        Instruction (Alpaca prompt)
                                  │
                                  ▼
                       tiktoken BPE tokenizer
                          (GPT-2 vocab, 50257)
                                  │
                                  ▼
        ┌─────────────────────────────────────────────┐
        │  Token embedding  +  learned position embed   │
        └─────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────┐
        │   TransformerBlock  × 24  (pre-norm)        │
        │   ┌───────────────────────────────────────┐ │
        │   │ LayerNorm                             │ │
        │   │ Multi-head causal self-attention (16) │ │
        │   │   scaled dot-product + triangular mask│ │
        │   │ + residual                            │ │
        │   │ LayerNorm                             │ │
        │   │ Feed-forward (4× expansion) + GELU    │ │
        │   │ + residual                            │ │
        │   └───────────────────────────────────────┘ │
        └─────────────────────────────────────────────┘
                                  │
                                  ▼
                  Final LayerNorm → output head
                                  │
                                  ▼
                      Logits over 50257 tokens
```

| Component | Implementation |
|---|---|
| Attention | Multi-head causal self-attention, scaled dot-product, upper-triangular mask |
| Normalization | Custom `LayerNorm` with learnable scale/shift |
| Activation | Custom `GELU` (tanh approximation) |
| Feed-forward | 2-layer MLP with 4× embedding expansion |
| Block | Pre-norm `TransformerBlock` with residual connections |
| Positional encoding | Learned absolute position embeddings |

**GPT-2 Medium config used:**

```python
{
    "vocab_size":      50257,
    "context_length":  1024,
    "emb_dim":         1024,
    "n_heads":         16,
    "n_layers":        24,
    "drop_rate":       0.0,
    "qkv_bias":        True,
}
```

---

## Training pipeline

### 1. Data preparation
- **1,100 instruction samples** in Alpaca format (`instruction`, `input`, `output`) in [`instruction-data.json`](instruction-data.json). The `input` field may be empty.
- Split: **85% train (935) / 10% test (110) / 5% validation (55)**.
- Formatted with the standard Alpaca prompt template:

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

### 2. Tokenization & batching
- `tiktoken` BPE tokenizer (GPT-2 vocabulary).
- Custom `InstructionDataset` pre-tokenizes every example.
- Custom `custom_collate_fn` handles variable-length sequences within a batch:
  - pads to the longest sequence in the batch (not a global max),
  - shifts inputs/targets by one position for next-token prediction,
  - replaces padding token IDs in the targets with `-100` so they are **ignored by the cross-entropy loss**,
  - optionally caps sequences at `allowed_max_length=1024`.
- `batch_size = 8`, `shuffle=True`, `drop_last=True`.

### 3. Weight loading
GPT-2 weights are downloaded from OpenAI's public checkpoint and loaded from TensorFlow format ([`gpt_download.py`](gpt_download.py)):
- TF checkpoint tensors are extracted and reshaped to the PyTorch layout,
- the fused QKV matrix from the TF checkpoint is split into `W_query`, `W_key`, `W_value`,
- all weights are wrapped as `nn.Parameter` after shape verification.

### 4. Fine-tuning
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 2
```
- Cross-entropy loss, with padding tokens masked out via the `-100` ignore index.
- Periodic train/val loss logging every 5 steps (`eval_freq=5`, `eval_iter=5`).
- Ran on **CPU** in **13.05 minutes**.
- Final weights saved as `gpt2-medium355M-sft.pth`.

---

## Inference

Generation uses the `generate` function in [`previous_chapters.py`](previous_chapters.py), which **supports** temperature scaling and top-k filtering:

```python
generate(model, idx, max_new_tokens, context_size,
         temperature=0.0, top_k=None, eos_id=None)
```

In this project's test-set generation, the defaults are used (`temperature=0.0`, `top_k=None`) → **greedy decoding**, with `eos_id=50256` (`<|endoftext|>`) as the stop token and `max_new_tokens=256`.

Example (test set):

```
### Instruction: Rewrite the sentence using a simile.
### Input:       The car is very fast.
Correct:         The car is as fast as lightning.
Model:           The car is as fast as a bullet.
```

---

## Evaluation

Model responses are scored automatically using **LLM-as-a-judge** via a locally running **Ollama** instance (**Llama 3**):

```python
prompt = (
    f"Given the input `{format_input(entry)}` "
    f"and correct output `{entry['output']}`, "
    f"score the model response `{entry['model_response']}`"
    f" on a scale from 0 to 100, where 100 is the best score. "
    f"Respond with the integer number only."
)
```

- Evaluated on **110 held-out test samples**.
- Deterministic scoring: `temperature=0`, `seed=123`, `num_ctx=2048`.
- **Average score: 49.45 / 100.**
- Per-sample responses are stored in [`instruction-data-with-response.json`](instruction-data-with-response.json).

---

## How to run

The entire pipeline lives in the notebook:

```bash
pip install numpy matplotlib tiktoken torch tqdm tensorflow
jupyter lab Finetune_to_Follow_Instructions.ipynb
```

For the evaluation section, install [Ollama](https://ollama.com) and pull Llama 3:

```bash
ollama pull llama3
ollama serve   # or launch the Ollama app
```

To reload the fine-tuned checkpoint:

```python
model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))
```

---

## Stack

| Library | Role |
|---|---|
| `torch` | Model, training loop, optimizer |
| `tiktoken` | BPE tokenization (GPT-2 vocab) |
| `tensorflow` | Loading pretrained GPT-2 TF checkpoints |
| `ollama` (Llama 3) | Local LLM judge for response scoring |
| `numpy` | Weight tensor manipulation during loading |

---

## File structure

```
.
├── Finetune_to_Follow_Instructions.ipynb   # Main notebook (full pipeline)
├── previous_chapters.py                    # From-scratch GPT model + training utils
├── gpt_download.py                         # GPT-2 weight downloader/loader (TF ckpt)
├── instruction-data.json                   # 1,100 instruction samples
├── instruction-data-with-response.json     # Test set with model responses + scores
├── gpt2-medium355M-sft.pth                 # Fine-tuned checkpoint (~1.6 GB, git-ignored)
└── loss-plot.pdf                           # Training/validation loss curves
```

> Note: model weight files (`*.pth`) are excluded from version control via `.gitignore`.

---

## Roadmap / TODO

The following are **not implemented yet**. They are listed here as future work — *not* as existing features of this repo.

**Serving & inference optimization**
- [ ] Quantization (e.g. 8-bit / 4-bit) to shrink the 1.6 GB checkpoint and speed up inference
- [ ] Inference optimization (KV-cache reuse, batching, ONNX/TorchScript export)
- [ ] Wrap the model in a serving layer (e.g. FastAPI endpoint)

**Demo & deployment**
- [ ] Live demo (Gradio or Streamlit interface)
- [ ] Deploy to Hugging Face Spaces / a VPS / Railway
- [ ] Personal portfolio site with a live chat interface

**Modeling / quality**
- [ ] Improve the LLM-judge score above the current 49.45 baseline (more data, more epochs, LR schedule, larger base model)
- [ ] Train on GPU/MPS to cut training time
- [ ] Custom/automated eval suite beyond the single Llama-3 judge (e.g. multiple judges, task-specific metrics)
- [ ] Decoding experiments (temperature / top-k / top-p sweeps — the `generate` function already supports these but they are not currently used in eval)

**MLOps / production**
- [ ] Model & data versioning (DVC or MLflow)
- [ ] Experiment tracking for runs and hyperparameters
- [ ] Monitoring and logging for a deployed endpoint
- [ ] A/B testing across model versions
- [ ] RAG integration for external knowledge

---

## References

Based on Chapter 7 of [*Build a Large Language Model From Scratch*](https://www.manning.com/books/build-a-large-language-model-from-scratch) by Sebastian Raschka — [code](https://github.com/rasbt/LLMs-from-scratch).
