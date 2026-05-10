# Tokenization Notebook - Explained

A from-scratch implementation of **Byte Pair Encoding (BPE)**, the subword tokenization algorithm used by GPT-2, GPT-3, GPT-4, LLaMA, and most modern LLMs. Based on Andrej Karpathy's "Let's build the GPT Tokenizer" video and the [`minbpe`](https://github.com/karpathy/minbpe) reference.

The earlier notebooks in this repo (`Bigram`, `MLP`, `GPT Development`) all use a trivial **character-level** tokenizer - one ID per unique character. That's fine for learning but wasteful in practice: every single character costs a token, the model has to relearn that "the" is a common word from raw character co-occurrences, and there's no way to share representations across morphologically related words. BPE solves this by learning a vocabulary of **byte sequences** that compress frequent patterns into single tokens.

---

## 1. Loading Text

```python
with open("tiny-shakespeare.txt") as f:
    text = f.read()
```

We reuse the Tiny Shakespeare corpus from the GPT notebook so the comparison is direct: same text, different tokenizer.

---

## 2. UTF-8 Bytes as the Starting Vocabulary

```python
tokens = text.encode("utf-8")
tokens = list(map(int, tokens))
```

Every BPE tokenizer starts from a **fixed alphabet of 256 byte values** (0-255). This matters because:

- **Universal coverage.** Any Unicode string can be expressed as UTF-8 bytes, so we never hit an "unknown character" - even emoji, Chinese, or Cyrillic. ASCII characters take 1 byte; higher Unicode code points take 2-4 bytes.
- **No special tokens.** No `<UNK>`, no `<PAD>`. Just bytes.
- **The starting point is bigger than the source text.** Non-ASCII characters expand: `"é"` is 1 character but 2 bytes (`b'\xc3\xa9'`). For pure ASCII text like Shakespeare this doesn't matter; for multilingual text it's the price of universal coverage.

The vocabulary will grow from 256 (the bytes) up to a target `vocab_size` by **learning** new entries through merging.

---

## 3. Counting Pair Frequencies (`get_stats`)

```python
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts
```

Walks the sequence and counts every adjacent pair. `zip(ids, ids[1:])` is the idiomatic way to get consecutive pairs without an explicit index. The most common pair on Shakespeare turns out to be `(101, 32)` = `b'e '` - "e followed by space" - which makes sense for English text.

This is the **statistic that drives the algorithm**: BPE always picks the most frequent adjacent pair to merge next.

---

## 4. Merging a Pair (`merge`)

```python
def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids
```

Walks the sequence and replaces every occurrence of `pair` with the new token id `idx`. The `i += 2` after a match is what prevents overlapping merges - if `pair = (1, 1)` and the input is `[1, 1, 1, 1]`, we get `[idx, idx]`, not `[idx, 1, idx]` or `[idx, idx, 1]`.

The sanity check `merge([5, 6, 6, 7, 9, 1], (6, 7), 99)` returns `[5, 6, 99, 9, 1]` - the `6, 7` in the middle becomes `99`; the leading `6` is left alone because it's not paired with a `7`.

---

## 5. Training the BPE (the core loop)

```python
vocab_size = 512
num_merges = vocab_size - 256

ids = list(tokens)
merges = {}

for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    ids = merge(ids, pair, idx)
    merges[pair] = idx
```

This is the entire BPE training algorithm. Each iteration:

1. **Count** all adjacent pairs in the current sequence.
2. **Pick** the most frequent pair.
3. **Mint** a new token id (sequentially: 256, 257, 258, ...).
4. **Replace** every occurrence of the pair with the new id - shrinking the sequence.
5. **Record** the merge in the `merges` dict so we can replay it during encoding.

After 256 merges, we have a vocabulary of 512 ids: 256 raw bytes + 256 learned merges. The sequence shrinks each iteration - merging `b'e '` once eliminates ~30,000 token positions in Shakespeare. The `compression ratio` printed at the end is `len(original_bytes) / len(final_ids)` and typically lands around 1.5-2.5x for English at this vocab size.

The order of merges matters at encode time, which is why `merges` is a dict keyed by pair with value = new id (the id implicitly encodes "when" the merge was learned, since ids are assigned sequentially).

---

## 6. Building the Vocab Lookup

```python
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]
```

`vocab` maps each token id back to its **raw bytes**. The first 256 entries are trivial - id `i` maps to the single byte `i`. Each merged token's bytes are the concatenation of its two parents' bytes, which are already in `vocab` because dicts preserve insertion order and we built up sequentially.

This is what makes decode work: to turn an id sequence back into text, look up the bytes for each id, concatenate, then UTF-8 decode.

---

## 7. Decoding (`decode`)

```python
def decode(ids):
    raw = b"".join(vocab[idx] for idx in ids)
    return raw.decode("utf-8", errors="replace")
```

Two steps: ids -> bytes (via `vocab` lookup) -> string (via `bytes.decode`).

`errors="replace"` is important. The model can in principle generate any sequence of token ids - including one whose concatenated bytes are **not valid UTF-8**. For example, a multi-byte character split across two tokens, where the model emits only the first one. Without `errors="replace"`, `bytes.decode` would raise `UnicodeDecodeError`. With it, invalid byte sequences become the replacement character `�`. This makes decoding lossy in a tiny number of cases but never crashes - critical when the tokenizer sits in the inference loop of a generative model.

---

## 8. Encoding (`encode`)

```python
def encode(text):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids
```

Encoding **replays the merges in the order they were learned**. Start with raw UTF-8 bytes, then repeatedly:

1. Find every pair currently in the sequence.
2. Pick the one with the **lowest merge id** - i.e. the merge learned earliest. Pairs not in `merges` get `float("inf")` so they're never chosen.
3. Apply that merge, shrinking the sequence.
4. Stop when no pair in the sequence is in `merges`.

The `min` with `float("inf")` fallback is a clever trick: it lets us share `get_stats` between training (which uses `max` on counts) and inference (which uses `min` on merge order) without writing a separate "find next mergeable pair" function.

**Why earliest-first?** The training loop merged the most frequent pair first, then re-counted on the new sequence. So merge 256 was applied before merge 257 ever saw the data. To reproduce the exact same token sequence at encode time, we have to apply merges in the same order. Picking the lowest merge id encodes that ordering.

This is O(n²) per call in the worst case (each iteration scans the whole sequence), which is fine for a learning notebook but is one of the things production tokenizers like `tiktoken` optimize aggressively (with priority queues over pair locations).

---

## 9. The `BPETokenizer` Class

```python
class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

    def train(self, text, vocab_size, verbose=False):
        ...

    def encode(self, text):
        ...

    def decode(self, ids):
        ...
```

Same algorithm, packaged so the trained state (`merges`, `vocab`) lives on the instance instead of in module-level globals. This is a step toward something you could pickle, ship with a model, or train multiple of for different corpora.

The roundtrip assertion `tok.decode(tok.encode(sample)) == sample` is the key correctness check - it catches almost every bug in encode/decode/merge ordering.

---

## 10. The GPT-2 Regex Pre-Split Pattern

```python
GPT2_SPLIT_PATTERN = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
```

A subtle problem with naive BPE: nothing stops it from learning merges that **cross word boundaries** or **cross category boundaries**. If `". "` (period + space) is frequent enough, BPE will merge them - then learn `". The"` as another token, then `". The "`, and so on. You end up with hundreds of variants of "common phrase ending followed by next word's start" instead of clean reusable subwords.

GPT-2's solution: **split the text into chunks first**, and never merge across chunk boundaries. The regex above produces chunks that are, in order of alternatives:

1. `'(?:[sdmt]|ll|ve|re)` - English contractions (`'s`, `'d`, `'m`, `'t`, `'ll`, `'ve`, `'re`) so `"don't"` becomes `["don", "'t"]`.
2. ` ?\p{L}+` - an optional space followed by one or more letters (any Unicode letter) - i.e. "word with leading space if present".
3. ` ?\p{N}+` - same for digits.
4. ` ?[^\s\p{L}\p{N}]+` - same for runs of punctuation/symbols.
5. `\s+(?!\S)` - trailing whitespace at end of line/string.
6. `\s+` - any other whitespace run.

The leading `?` on the space alternatives is what lets the tokenizer learn `" the"` (space-the) as a single token while keeping it separate from `"the"` at the start of a string - which is why GPT-2 token lists are full of entries like ` the`, ` and`, ` of`. Word-with-leading-space is the natural English unit.

Note this requires the third-party `regex` package, not Python's stdlib `re` - because stdlib `re` doesn't support the Unicode property escapes `\p{L}` and `\p{N}`.

---

## 11. The `RegexBPETokenizer` Class

```python
class RegexBPETokenizer(BPETokenizer):
    def __init__(self, pattern=GPT2_SPLIT_PATTERN):
        super().__init__()
        self.pattern = re.compile(pattern)

    def train(self, text, vocab_size, verbose=False):
        chunks = self.pattern.findall(text)
        ids = [list(ch.encode("utf-8")) for ch in chunks]
        ...
```

Same BPE algorithm with one structural change: the input is no longer a single flat list of ids but a **list of lists** - one inner list per regex chunk. During training:

- `get_stats` is computed across all chunks but only over **within-chunk** pairs - we never count `(last_byte_of_chunk_A, first_byte_of_chunk_B)` because the `zip(ids, ids[1:])` runs inside each chunk independently.
- `merge` is applied to each chunk independently.

This means BPE can never learn a merge that crosses a regex boundary - exactly what we wanted. At encode time, we pre-split the input the same way and merge each chunk independently, then concatenate the per-chunk id sequences.

The first few learned merges with the regex pre-split look noticeably cleaner than without - they're fragments of real English words and word-with-leading-space patterns rather than weird cross-word junk like `e\nT` or `.\nO`.

---

## How this connects to the rest of the repo

The character-level encode/decode in `GPT Development.ipynb` could in principle be swapped for `BPETokenizer.encode` / `.decode`. The model architecture is unchanged - only `vocab_size` (passed to `nn.Embedding`) needs to update. Practical caveats:

- **Smaller `vocab_size` = less memory but worse compression**, larger = the opposite. For Tiny Shakespeare, 512-1024 is plenty; production GPT-2 uses 50,257.
- **Block size now counts BPE tokens, not characters.** A `block_size` of 256 BPE tokens covers maybe 600-1000 characters of context - more than the equivalent character-level model, which is most of the win.
- **The `encode` step becomes a non-trivial cost** at training time when you tokenize the whole corpus once up front, and at inference time on every prompt. The pure-Python implementation here is fine for learning; production code uses `tiktoken` or similar.

The simplifications vs. real production tokenizers:
- No special tokens (`<|endoftext|>`, `<|im_start|>`, etc.) - production tokenizers reserve a few ids that should never be produced by training, only injected at the boundaries of documents/turns.
- No save/load to disk - production tokenizers serialize `merges` + `vocab` + the regex pattern as a single artifact that ships with model weights.
- No tiktoken-style token-byte mapping for printable display - this is what makes GPT's tokenizer playground show ` the` as a visible token rather than as raw bytes.
