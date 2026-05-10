# GPT Development Notebook - Explained

A character-level GPT (Generative Pre-trained Transformer) trained on the Tiny Shakespeare dataset. Based on the decoder-only transformer architecture from "Attention Is All You Need."

---

## 1. Data Loading & Tokenization

```python
with open("tiny-shakespeare.txt") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
```

Builds a **character-level tokenizer**. Every unique character in the text gets a unique integer ID. `encode` converts a string into a list of integers, `decode` reverses it. This is the simplest possible tokenizer - production models use subword tokenizers (BPE/SentencePiece) for better compression, but character-level works fine for learning.

`vocab_size` = 65 (26 lowercase + 26 uppercase + digits + punctuation + newline + space).

---

## 2. Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `block_size` | 256 | Context window - max number of tokens the model can look back at |
| `batch_size` | 64 | Number of independent sequences processed in parallel per step |
| `n_embd` | 384 | Embedding dimension - the size of the internal representation vectors |
| `n_head` | 6 | Number of parallel attention heads |
| `n_layer` | 6 | Number of transformer blocks stacked sequentially |
| `learning_rate` | 3e-4 | Step size for the optimizer |
| `max_iters` | 5000 | Total training steps |
| `eval_iters` | 200 | Number of batches averaged over when estimating loss |
| `eval_interval` | 20 | How often (in steps) to print evaluation loss |
| `dropout` | 0.2 | Fraction of neurons randomly zeroed during training (regularization) |

The `device` variable auto-selects CUDA (GPU) if available, otherwise falls back to CPU.

---

## 3. Train/Val Split

```python
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
```

90% of the data is used for training, 10% for validation. The validation set measures how well the model generalizes to text it hasn't trained on - if train loss drops but val loss doesn't, the model is overfitting.

---

## 4. Batch Sampling (`get_batch`)

```python
def get_batch(split):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,))
    xb = torch.stack([data[i:i+block_size] for i in idx])
    yb = torch.stack([data[i+1:i+block_size+1] for i in idx])
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb
```

Randomly samples `batch_size` chunks of `block_size` tokens from the dataset. For each input chunk `xb`, the target `yb` is the same chunk shifted one position to the right - the model learns to predict the next character at every position.

Output shapes: `xb` = `[B, T]`, `yb` = `[B, T]` where B=batch_size, T=block_size.

The `.to(device)` call moves tensors to the GPU if available. All tensors involved in computation must be on the same device.

---

## 5. Self-Attention Head (`Head`)

```python
class Head(nn.Module):
    def __init__(self, head_size):
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        v = self.value(x)
        out = weights @ v
        return out
```

The core mechanism of the transformer. Each token produces three vectors:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I provide?"

The attention score between two tokens is the dot product of one's query with the other's key. High score = high relevance. The steps:

1. **Project** input through K, Q, V linear layers: `[B, T, C]` -> `[B, T, head_size]`
2. **Compute affinities**: `Q @ K^T` gives a `[B, T, T]` matrix where entry `[i, j]` is how much token `i` attends to token `j`
3. **Scale** by `1/sqrt(head_size)` to prevent softmax saturation (keeps gradients healthy)
4. **Mask** future positions with `-inf` (causal/autoregressive - a token can only attend to itself and earlier tokens, not future ones)
5. **Softmax** normalizes weights to sum to 1 per row
6. **Dropout** randomly zeros some attention weights during training
7. **Weighted sum** of values: `weights @ V` produces the output

### The Causal Mask (`tril`) in Detail

`torch.tril(torch.ones(block_size, block_size))` builds a **lower triangular matrix** of 1s. For `block_size=5` it looks like:

```
1 0 0 0 0
1 1 0 0 0
1 1 1 0 0
1 1 1 1 0
1 1 1 1 1
```

`register_buffer('tril', ...)` saves it as part of the module's state (it moves with `.to(device)` and is included in `state_dict`) but it is **not a learnable parameter** - it's a fixed mask.

**Why it's needed.** After computing raw attention scores `weights = q @ k.T` (shape `[B, T, T]`), every position `i` has a score against every other position `j`. Without masking, position 2 could attend to position 4 - i.e., a token could "see the future." The line:

```python
weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
```

sets all entries **above the diagonal** to `-inf`. After `softmax`, those `-inf` entries become `0`, so:

- Token at position 0 only attends to position 0
- Token at position 1 attends to positions 0 and 1
- Token at position `i` attends to positions `0..i`

**Why this matters for a GPT.** The model is trained to predict the next token. During training the whole sequence is fed in parallel and a loss is computed at every position simultaneously - but each position must only use information from earlier positions, otherwise the model would trivially "cheat" by reading the answer. The triangular mask enforces that constraint inside the parallel matmul.

**Why slice `[:T, :T]`.** `tril` is pre-allocated at the maximum `block_size`, but the actual sequence length `T` in a forward pass may be shorter (e.g., during generation when the context is still being built up). `self.tril[:T, :T]` grabs the top-left `T x T` corner so the mask matches the current attention matrix's shape.

**Encoder vs. decoder.** A BERT-style encoder would *not* use this mask - it wants bidirectional context. The triangular mask is the single line that turns a generic self-attention block into a *decoder/causal* block, which is what makes this a GPT.

---

## 6. Multi-Head Attention (`MultiHeadAttention`)

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
```

Runs multiple attention heads in parallel and concatenates their outputs. With `n_head=6` and `head_size=64`, each head captures different types of relationships (e.g., one head might learn syntactic patterns, another semantic ones). The concatenated output (`6 * 64 = 384 = n_embd`) is projected back through a linear layer.

The **projection layer** (`self.proj`) allows the model to mix information across heads before passing it to the next layer. This is critical - without it, the heads remain completely independent.

---

## 7. Feed-Forward Network (`FeedForward`)

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
```

A simple two-layer MLP applied independently to each token position. The inner dimension is **4x wider** than the embedding dimension (384 -> 1536 -> 384). This expansion/compression pattern gives the network capacity to learn complex per-token transformations.

- **GELU** (Gaussian Error Linear Unit): A smooth activation function used in GPT-2 and later. Unlike ReLU which has a hard cutoff at 0, GELU smoothly gates values near zero, which empirically works better for transformers.
- **Dropout** on the output for regularization.

If attention is "communication" between tokens (gathering information), feed-forward is "computation" on each token (processing that information individually).

---

## 8. Transformer Block (`Block`)

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

One complete transformer layer combining attention + feed-forward with two critical design choices:

### Pre-Norm Architecture
LayerNorm is applied **before** each sub-layer (`self.ln1(x)` then `self.sa(...)`) rather than after. This is the GPT-2 style. It produces more stable gradients and makes training easier, especially for deeper models.

### Residual Connections
The `x + ...` pattern is a **skip connection**. The output is the input plus the transformation. This solves the vanishing gradient problem in deep networks - gradients can flow directly through the addition, keeping the signal strong even through 6 stacked blocks. Without residuals, deep transformers are essentially untrainable.

---

## 9. The GPT Model (`BigramLanguageModel`)

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
```

### Forward Pass
1. **Token embedding**: Maps each integer token to a learned 384-dim vector. Shape: `[B, T]` -> `[B, T, 384]`
2. **Position embedding**: Adds a learned vector for each position (0 to T-1). This is how the model knows token order - without it, attention is permutation-invariant and "hello" = "olleh"
3. **Transformer blocks**: 6 layers of attention + feed-forward
4. **Final LayerNorm** (`ln_f`): Normalizes the output of the last block before the projection
5. **Language model head** (`lm_head`): Projects from embedding space (384) back to vocabulary space (65), producing a score (logit) for each possible next character

### Loss Computation
Uses **cross-entropy loss**: compares the model's predicted probability distribution over the 65 characters against the actual next character. Lower = better. Random guessing would give `-ln(1/65) = 4.17`, so initial loss should be near that.

### Generation
```python
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

Autoregressive generation loop:
1. **Crop** to last `block_size` tokens (the model can't see beyond its context window)
2. **Forward pass** to get logits for all positions
3. **Take only the last position's logits** (`[:, -1, :]`) - that's the prediction for the *next* token
4. **Softmax** converts logits to probabilities
5. **Sample** from the distribution (multinomial sampling adds randomness/creativity vs. always picking the most likely token)
6. **Append** the sampled token and repeat

---

## 10. Loss Estimation (`estimate_loss`)

```python
@torch.no_grad()
def estimate_loss():
    ...
```

Averages loss over `eval_iters` (200) random batches for both train and val splits. This gives a much more stable loss estimate than looking at a single batch.

Key details:
- `@torch.no_grad()`: Disables gradient tracking during evaluation, saving memory and compute
- `model.eval()`: Disables dropout (uses all neurons). Switched back with `model.train()` after

---

## 11. Training Loop

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(...)
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

Standard PyTorch training loop:
1. **Sample** a random batch
2. **Forward pass**: compute predictions and loss
3. **Zero gradients**: clear accumulated gradients from the previous step (`set_to_none=True` is slightly more efficient than setting to zero)
4. **Backward pass**: compute gradients of loss w.r.t. all parameters
5. **Optimizer step**: update parameters using AdamW (Adam with weight decay regularization)

**AdamW** is the standard optimizer for transformers. It maintains per-parameter learning rates that adapt based on gradient history, making it much more effective than plain SGD for this architecture.

---

## Architecture Summary

```
Input tokens [B, T]
    |
Token Embedding [B, T, 384]  +  Position Embedding [B, T, 384]
    |
    v
Transformer Block x6:
    |-- LayerNorm -> Multi-Head Attention (6 heads x 64 dim) -> Residual Add
    |-- LayerNorm -> Feed-Forward (384 -> 1536 -> 384)       -> Residual Add
    |
Final LayerNorm
    |
Linear Head [B, T, 65]  (logits over vocabulary)
```

Total parameter count: ~10.7M parameters.
