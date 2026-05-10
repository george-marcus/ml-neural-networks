# ml-neural-networks

Progressive neural network implementations for language modeling, following Andrej Karpathy's "Zero to Hero" / "makemore" series. Each notebook builds on the previous one, working from raw character statistics up to a working decoder-only transformer.

## Notebooks

- **Bigram MakeMore.ipynb** - Simple bigram language model that learns next-character probabilities directly from a count matrix, then re-derives the same model as a single-layer neural network.
- **MLP MakeMore.ipynb** - Multi-layer perceptron with character embeddings, BatchNorm, and a manual training loop. Uses running statistics at inference time so dropout/BN behave correctly.
- **GPT Development.ipynb** - Character-level GPT with multi-head self-attention, causal masking, residual connections, and pre-norm transformer blocks. Trained on Tiny Shakespeare. See [GPT Development.md](GPT%20Development.md) for a full walkthrough of the architecture.

## Companion docs

- [GPT Development.md](GPT%20Development.md) - Section-by-section explanation of the GPT notebook: tokenization, batching, attention heads, the causal `tril` mask, multi-head attention, feed-forward blocks, residuals, and the training/generation loops.

## Data

- `names.txt` - Names dataset used by the bigram and MLP notebooks.
- `tiny-shakespeare.txt` - Concatenated works of Shakespeare used to train the GPT notebook.

## Requirements

- Python 3.x
- PyTorch
- Jupyter (for running the notebooks)
- A CUDA-capable GPU is recommended for the GPT notebook; the bigram and MLP notebooks run comfortably on CPU.
