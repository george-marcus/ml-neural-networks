# ml-neural-networks

Progressive neural network implementations for language modeling, following Andrej Karpathy's "Zero to Hero" / "makemore" series. Each notebook builds on the previous one, working from raw character statistics up to a working decoder-only transformer.

## Notebooks

- **Bigram MakeMore.ipynb** - Simple bigram language model that learns next-character probabilities directly from a count matrix, then re-derives the same model as a single-layer neural network.
- **MLP MakeMore.ipynb** - Multi-layer perceptron with character embeddings, BatchNorm, and a manual training loop. Uses running statistics at inference time so dropout/BN behave correctly.
- **GPT Development.ipynb** - Character-level GPT with multi-head self-attention, causal masking, residual connections, and pre-norm transformer blocks. Trained on Tiny Shakespeare. See [GPT Development.md](GPT%20Development.md) for a full walkthrough of the architecture.
- **Tokenization.ipynb** - Byte Pair Encoding (BPE) tokenizer built from scratch: byte-level start, iterative pair merging, encode/decode, and a GPT-2-style regex pre-split variant. See [Tokenization.md](Tokenization.md) for the walkthrough.

## Companion docs

- [GPT Development.md](GPT%20Development.md) - Section-by-section explanation of the GPT notebook: tokenization, batching, attention heads, the causal `tril` mask, multi-head attention, feed-forward blocks, residuals, and the training/generation loops.
- [Tokenization.md](Tokenization.md) - Walkthrough of the BPE notebook: UTF-8 byte vocabulary, pair counting, the merge loop, encode/decode roundtrip, and why GPT-2 pre-splits with a regex before merging.

## Data

- `names.txt` - Names dataset used by the bigram and MLP notebooks.
- `tiny-shakespeare.txt` - Concatenated works of Shakespeare used to train the GPT notebook.

## Requirements

- Python 3.x
- PyTorch
- Jupyter (for running the notebooks)
- `regex` package (for the GPT-2-style pre-split in the Tokenization notebook - `pip install regex`)
- A CUDA-capable GPU is recommended for the GPT notebook; the bigram, MLP, and tokenization notebooks run comfortably on CPU.
