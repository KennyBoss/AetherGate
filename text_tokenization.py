#!/usr/bin/env python3
"""Tokenization helpers for the TextPy/SoA text runtime.

The project started with a tiny word-level tokenizer. This module keeps that
path as the default while adding a lightweight BPE-style subword tokenizer that
can be learned from a real corpus and stored inside checkpoints.
"""

from __future__ import annotations

import re
from copy import deepcopy
from collections import Counter
from typing import Any

import numpy as np


UNK = "<unk>"
BOS = "<bos>"
END_WORD = "</w>"

WORD_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
WORD_ONLY_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
PUNCT_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)


def word_tokenize(text: str) -> list[str]:
    return WORD_PATTERN.findall(text.lower())


def word_detokenize(tokens: list[str]) -> str:
    out: list[str] = []
    no_space_before = set(".,;:!?)]}")
    no_space_after = set("([{")
    for token in tokens:
        if token in {UNK, BOS}:
            continue
        if not out:
            out.append(token)
        elif token in no_space_before:
            out[-1] += token
        elif out[-1][-1:] in no_space_after:
            out[-1] += token
        else:
            out.append(" " + token)
    return "".join(out)


def default_word_tokenizer_config() -> dict[str, Any]:
    return {
        "type": "word",
        "lowercase": True,
    }


def initial_bpe_pieces(token: str) -> tuple[str, ...]:
    if WORD_ONLY_PATTERN.fullmatch(token):
        return tuple([*token, END_WORD])
    return (token,)


def merge_pair_once(pieces: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    merged: list[str] = []
    i = 0
    while i < len(pieces):
        if i + 1 < len(pieces) and pieces[i] == pair[0] and pieces[i + 1] == pair[1]:
            merged.append(pieces[i] + pieces[i + 1])
            i += 2
        else:
            merged.append(pieces[i])
            i += 1
    return tuple(merged)


def bpe_merge_ranks(tokenizer_config: dict[str, Any]) -> dict[tuple[str, str], int]:
    if any(not str(left) or not str(right) for left, right in tokenizer_config.get("merges", [])):
        raise ValueError("BPE merge operands must be nonempty")
    return {
        (str(left), str(right)): rank
        for rank, (left, right) in enumerate(tokenizer_config.get("merges", []))
    }


def merge_best_ranked_pair(
    pieces: tuple[str, ...],
    merge_ranks: dict[tuple[str, str], int],
) -> tuple[str, ...] | None:
    best_pair: tuple[str, str] | None = None
    best_rank: int | None = None
    for i in range(len(pieces) - 1):
        pair = (pieces[i], pieces[i + 1])
        rank = merge_ranks.get(pair)
        if rank is not None and (best_rank is None or rank < best_rank):
            best_pair = pair
            best_rank = rank
    if best_pair is None:
        return None
    return merge_pair_once(pieces, best_pair)


def count_pairs(words: dict[tuple[str, ...], int]) -> Counter[tuple[str, str]]:
    pairs: Counter[tuple[str, str]] = Counter()
    for pieces, count in words.items():
        for i in range(len(pieces) - 1):
            pairs[(pieces[i], pieces[i + 1])] += count
    return pairs


def learn_bpe_tokenizer_config(
    text: str,
    *,
    target_vocab_size: int,
    min_pair_frequency: int,
    train_chars: int | None,
) -> dict[str, Any]:
    if target_vocab_size < 8:
        raise ValueError("--bpe-vocab-size must be at least 8.")
    if min_pair_frequency < 1:
        raise ValueError("--bpe-min-frequency must be at least 1.")

    sample = text[:train_chars] if train_chars and train_chars > 0 else text
    raw_tokens = word_tokenize(sample)
    if not raw_tokens:
        raise ValueError("Cannot learn BPE tokenizer from an empty corpus.")

    words: dict[tuple[str, ...], int] = Counter(initial_bpe_pieces(token) for token in raw_tokens)
    units = {piece for pieces in words for piece in pieces}
    merges: list[list[str]] = []

    while len(units) + 2 < target_vocab_size:
        pairs = count_pairs(words)
        if not pairs:
            break
        (left, right), frequency = max(pairs.items(), key=lambda item: (item[1], item[0]))
        if frequency < min_pair_frequency:
            break
        pair = (left, right)
        merged_unit = left + right
        next_words: Counter[tuple[str, ...]] = Counter()
        for pieces, count in words.items():
            next_words[merge_pair_once(pieces, pair)] += count
        words = next_words
        merges.append([left, right])
        units.add(merged_unit)

    return {
        "type": "bpe",
        "lowercase": True,
        "end_word": END_WORD,
        "target_vocab_size": int(target_vocab_size),
        "min_pair_frequency": int(min_pair_frequency),
        "train_chars": int(train_chars) if train_chars else None,
        "trained_on_chars": len(sample),
        "trained_on_word_tokens": len(raw_tokens),
        "merges": merges,
    }


def encode_bpe_raw_token(
    token: str,
    tokenizer_config: dict[str, Any],
    merge_ranks: dict[tuple[str, str], int] | None = None,
    *,
    allowed_units: frozenset[str] | None = None,
    reverse_merges: dict[str, tuple[str, str]] | None = None,
) -> list[str]:
    pieces = initial_bpe_pieces(token)
    ranks = merge_ranks if merge_ranks is not None else bpe_merge_ranks(tokenizer_config)
    while len(pieces) > 1:
        next_pieces = merge_best_ranked_pair(pieces, ranks)
        if next_pieces is None:
            break
        pieces = next_pieces
    if allowed_units is None and "training_vocab" in tokenizer_config:
        allowed_units = frozenset(tokenizer_config["training_vocab"])
    if allowed_units is None:
        return list(pieces)
    reverse_merges = reverse_merges if reverse_merges is not None else {
        str(a) + str(b): (str(a), str(b)) for a, b in tokenizer_config.get("merges", [])
    }
    return [unit for piece in pieces for unit in _fallback_piece(piece, allowed_units, reverse_merges)]


def _fallback_piece(piece, allowed, reverse):
    if piece in allowed:
        return [piece]
    if piece in reverse:
        left, right = reverse[piece]
        return _fallback_piece(left, allowed, reverse) + _fallback_piece(right, allowed, reverse)
    return [UNK]


def tokenize_with_config(text: str, tokenizer_config: dict[str, Any] | None) -> list[str]:
    tokenizer_config = tokenizer_config or default_word_tokenizer_config()
    tokenizer_type = tokenizer_config.get("type", "word")
    raw_tokens = word_tokenize(text)
    if tokenizer_type == "word":
        return raw_tokens
    if tokenizer_type == "bpe":
        tokens: list[str] = []
        merge_ranks = bpe_merge_ranks(tokenizer_config)
        allowed = (frozenset(tokenizer_config["training_vocab"])
                   if "training_vocab" in tokenizer_config else None)
        reverse = {a + b: (a, b) for a, b in merge_ranks}
        cache: dict[str, tuple[str, ...]] = {}
        for raw_token in raw_tokens:
            encoded = cache.get(raw_token)
            if encoded is None:
                encoded = tuple(encode_bpe_raw_token(raw_token, tokenizer_config, merge_ranks,
                                                    allowed_units=allowed, reverse_merges=reverse))
                cache[raw_token] = encoded
            tokens.extend(encoded)
        return tokens
    raise ValueError(f"Unknown tokenizer type: {tokenizer_type!r}")


def detokenize_subwords(tokens: list[str], tokenizer_config: dict[str, Any] | None = None) -> str:
    end_word = str((tokenizer_config or {}).get("end_word", END_WORD))
    word_tokens: list[str] = []
    buffer: list[str] = []
    for token in tokens:
        if token == UNK and "training_vocab" in (tokenizer_config or {}):
            if buffer:
                word_tokens.append("".join(buffer))
                buffer = []
            word_tokens.append("\ufffd")
            continue
        if token in {UNK, BOS}:
            continue
        if token.endswith(end_word):
            buffer.append(token[: -len(end_word)])
            word_tokens.append("".join(buffer))
            buffer = []
        elif PUNCT_PATTERN.fullmatch(token) and not buffer:
            word_tokens.append(token)
        else:
            buffer.append(token)
    if buffer:
        word_tokens.append("".join(buffer))
    return word_detokenize(word_tokens)


def detokenize_with_config(tokens: list[str], tokenizer_config: dict[str, Any] | None = None) -> str:
    tokenizer_type = (tokenizer_config or {}).get("type")
    if tokenizer_type == "bpe" or any(token.endswith(END_WORD) for token in tokens):
        return detokenize_subwords(tokens, tokenizer_config)
    return word_detokenize(tokens)


def build_tokenizer_config(
    text: str,
    *,
    tokenizer: str,
    bpe_vocab_size: int,
    bpe_min_frequency: int,
    bpe_train_chars: int | None,
    tokenizer_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if tokenizer_config is not None:
        return tokenizer_config
    if tokenizer == "word":
        return default_word_tokenizer_config()
    if tokenizer == "bpe":
        return learn_bpe_tokenizer_config(
            text,
            target_vocab_size=bpe_vocab_size,
            min_pair_frequency=bpe_min_frequency,
            train_chars=bpe_train_chars,
        )
    raise ValueError(f"Unknown tokenizer: {tokenizer!r}")


def build_vocab(tokens: list[str]) -> tuple[dict[str, int], list[str]]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    vocab = [UNK, BOS]
    vocab.extend(sorted((token for token in counts if token not in {UNK, BOS}),
                        key=lambda item: (-counts[item], item)))
    token_to_id = {token: i for i, token in enumerate(vocab)}
    return token_to_id, vocab


def tokenizer_vocab_units(tokenizer_config: dict[str, Any]) -> list[str]:
    if tokenizer_config.get("type") != "bpe":
        return []
    units: set[str] = {END_WORD}
    for left, right in tokenizer_config.get("merges", []):
        left = str(left)
        right = str(right)
        units.add(left)
        units.add(right)
        units.add(left + right)
    return sorted(units)


def build_vocab_for_tokenizer(
    tokens: list[str],
    tokenizer_config: dict[str, Any],
) -> tuple[dict[str, int], list[str]]:
    # Merge operands are tokenizer instructions, not necessarily emitted targets.
    return build_vocab(tokens)


def make_token_streams(token_ids, streams, tokens_per_stream):
    if streams < 1 or tokens_per_stream < 1 or len(token_ids) < 1:
        raise ValueError("Need nonempty token IDs and positive stream dimensions")
    needed = streams * tokens_per_stream + 1
    repeats = int(np.ceil(needed / len(token_ids)))
    return np.tile(token_ids, repeats)[:streams * tokens_per_stream].reshape(streams, tokens_per_stream)


def sample_training_streams(token_ids, streams, tokens_per_stream, seq_len):
    """Rarest-first TRAIN coverage, then evenly spaced windows; fixed budget.

    Covering a class means putting it in a consumed TARGET, not merely in the
    first input or the unused tail. Evaluation never enters this selection.
    """
    if streams < 1 or not 0 < seq_len < tokens_per_stream or len(token_ids) < 1:
        raise ValueError("Need nonempty training IDs and complete target windows")
    ids = np.asarray(token_ids)
    used = ((tokens_per_stream - 1) // seq_len) * seq_len
    counts = np.bincount(ids)
    first = np.full(len(counts), len(ids), dtype=np.int64)
    np.minimum.at(first, ids, np.arange(len(ids), dtype=np.int64))
    required = sorted((i for i in range(2, len(counts)) if counts[i]), key=lambda i: (counts[i], i))
    covered = set()
    starts = []
    for token in required:
        if token in covered:
            continue
        if len(starts) == streams:
            break
        start = max(0, min(int(first[token]) - used // 2, max(0, len(ids) - used - 1)))
        starts.append(start)
        target_positions = (start + np.arange(1, used + 1)) % len(ids)
        covered.update(np.asarray(ids[target_positions]).tolist())
    coverage_windows = len(starts)
    remaining = streams - len(starts)
    for start in np.linspace(0, max(0, len(ids) - tokens_per_stream), remaining, dtype=np.int64):
        starts.append(int(start))
    positions = (np.asarray(starts)[:, None] + np.arange(tokens_per_stream)) % len(ids)
    return ids[positions], {"method": "train_coverage_then_uniform_v1", "stream_starts": starts,
                            "coverage_windows": coverage_windows, "source_units": len(required),
                            "covered_source_units": len(set(required) & covered)}


def fit_bpe_training_data(text, tokenizer_config, *, streams, tokens_per_stream, seq_len):
    """Fit output units to actual TRAIN targets; expose fallback atoms naturally.

    No evaluation data or synthetic alphabet examples are used. A merge that
    swallows an otherwise untrained fallback atom is disabled. Unsupported
    combinations decompose through the merge tree; unseen characters use UNK.
    """
    if tokenizer_config.get("type") != "bpe" or not 0 < seq_len < tokens_per_stream:
        raise ValueError("Need BPE and at least one complete training chunk")
    config = deepcopy(tokenizer_config)
    config.pop("training_vocab", None)
    passes = []
    used = ((tokens_per_stream - 1) // seq_len) * seq_len
    for _ in range(64):
        tokens = tokenize_with_config(text, config)
        if not tokens:
            raise ValueError("Cannot fit a training vocabulary to empty text")
        mapping, vocab = build_vocab(tokens)
        packed, selection = sample_training_streams(encode_tokens(tokens, mapping), streams,
                                                   tokens_per_stream, seq_len)
        counts = np.bincount(packed[:, 1:used + 1].reshape(-1), minlength=len(vocab))
        active = {vocab[i] for i in range(2, len(vocab)) if counts[i] > 0}
        reverse = {str(a) + str(b): (str(a), str(b)) for a, b in config.get("merges", [])}

        def atoms(piece):
            if piece not in reverse:
                return {piece}
            a, b = reverse[piece]
            return atoms(a) | atoms(b)

        missing = set().union(*(atoms(t) for t in active)) - active if active else set()
        passes.append({"vocab_size": len(vocab), "ordinary_zero_targets": int(np.sum(counts[2:] == 0)),
                       "missing_fallback_atoms": sorted(missing), "merges": len(config.get("merges", []))})
        if missing:
            merges = [pair for pair in config.get("merges", [])
                      if pair[0] not in missing and pair[1] not in missing]
            if len(merges) == len(config.get("merges", [])):
                raise ValueError("Cannot expose the required BPE fallback atoms")
            config["merges"] = merges
            config.pop("training_vocab", None)
            continue
        if "training_vocab" in config and set(config["training_vocab"]) == active:
            config["vocab_policy"] = "actual_train_targets_v1"
            return {"tokenizer_config": config, "token_to_id": mapping, "vocab": vocab,
                    "train_tokens": tokens, "train_streams": packed,
                    "coverage": {"ordinary_zero_targets": 0, "ordinary_units": len(active),
                                 "reserved_target_counts": {UNK: int(counts[0]), BOS: int(counts[1])},
                                 "trained_targets": int(counts.sum()), "passes": passes,
                                 "sampling": selection}}
        config["training_vocab"] = sorted(active)
    raise ValueError("BPE training vocabulary did not converge within 64 fitting passes")


def encode_tokens(tokens: list[str], token_to_id: dict[str, int]) -> np.ndarray:
    unk_id = token_to_id[UNK]
    ids = [token_to_id[BOS]]
    ids.extend(token_to_id.get(token, unk_id) for token in tokens)
    return np.asarray(ids, dtype=np.int32)


def tokenizer_summary(
    text: str,
    tokenizer_config: dict[str, Any],
    tokens: list[str] | None = None,
) -> dict[str, Any]:
    raw_tokens = word_tokenize(text)
    encoded_tokens = tokens if tokens is not None else tokenize_with_config(text, tokenizer_config)
    return {
        "type": tokenizer_config.get("type", "word"),
        "word_tokens": len(raw_tokens),
        "tokens": len(encoded_tokens),
        "tokens_per_word": len(encoded_tokens) / max(1, len(raw_tokens)),
        "unique_tokens": len(set(encoded_tokens)),
        "merges": len(tokenizer_config.get("merges", [])),
        "target_vocab_size": tokenizer_config.get("target_vocab_size"),
    }
