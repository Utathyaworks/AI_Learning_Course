# Byte-pair encoding: learn merges

Implement `learn_bpe(word_counts, num_merges)`.

- `word_counts`: dict word → count. Each word starts as a tuple of its characters plus the end marker `"</w>"`.
- Repeat `num_merges` times: count adjacent symbol pairs weighted by word counts; merge the most frequent pair
  everywhere. **Break ties by the lexicographically smallest pair** (compare the tuple `(a, b)`). Stop early if no
  pair remains.
- Return the list of merges as tuples `(a, b)` in order.
