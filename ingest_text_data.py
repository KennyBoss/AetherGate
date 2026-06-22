#!/usr/bin/env python3
"""Ingest real plain-text corpora for TextPy/SoA experiments.

The first production-grade fuel source is a small Project Gutenberg preset:
public-domain books downloaded as UTF-8 text, stripped of boilerplate, joined
into one corpus, and recorded with metadata for repeatable sweeps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from train_ssm_text import load_text
from text_tokenization import tokenize_with_config, default_word_tokenizer_config


ROOT = Path(__file__).resolve().parent
DEFAULT_BOOKS = [
    {
        "id": "1342",
        "title": "Pride and Prejudice",
        "author": "Jane Austen",
        "url": "https://www.gutenberg.org/files/1342/1342-0.txt",
    },
    {
        "id": "11",
        "title": "Alice's Adventures in Wonderland",
        "author": "Lewis Carroll",
        "url": "https://www.gutenberg.org/files/11/11-0.txt",
    },
    {
        "id": "84",
        "title": "Frankenstein",
        "author": "Mary Wollstonecraft Shelley",
        "url": "https://www.gutenberg.org/files/84/84-0.txt",
    },
    {
        "id": "1661",
        "title": "The Adventures of Sherlock Holmes",
        "author": "Arthur Conan Doyle",
        "url": "https://www.gutenberg.org/files/1661/1661-0.txt",
    },
    {
        "id": "2701",
        "title": "Moby-Dick",
        "author": "Herman Melville",
        "url": "https://www.gutenberg.org/files/2701/2701-0.txt",
    },
    {
        "id": "98",
        "title": "A Tale of Two Cities",
        "author": "Charles Dickens",
        "url": "https://www.gutenberg.org/files/98/98-0.txt",
    },
    {
        "id": "345",
        "title": "Dracula",
        "author": "Bram Stoker",
        "url": "https://www.gutenberg.org/files/345/345-0.txt",
    },
    {
        "id": "5200",
        "title": "Metamorphosis",
        "author": "Franz Kafka",
        "url": "https://www.gutenberg.org/files/5200/5200-0.txt",
    },
    {
        "id": "4300",
        "title": "Ulysses",
        "author": "James Joyce",
        "url": "https://www.gutenberg.org/files/4300/4300-0.txt",
    },
    {
        "id": "2600",
        "title": "War and Peace",
        "author": "Leo Tolstoy",
        "url": "https://www.gutenberg.org/files/2600/2600-0.txt",
    },
    {
        "id": "74",
        "title": "The Adventures of Tom Sawyer",
        "author": "Mark Twain",
        "url": "https://www.gutenberg.org/files/74/74-0.txt",
    },
    {
        "id": "76",
        "title": "Adventures of Huckleberry Finn",
        "author": "Mark Twain",
        "url": "https://www.gutenberg.org/files/76/76-0.txt",
    },
]


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def fetch_url(url: str, timeout: int, retries: int) -> str:
    headers = {"User-Agent": "TextPy-SoA research corpus builder (contact: local)"}
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
            return raw.decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise SystemExit(f"Failed to fetch {url}: {last_error}")


def fetch_first_url(urls: list[str], timeout: int, retries: int) -> tuple[str, str]:
    errors: list[str] = []
    for url in urls:
        try:
            return fetch_url(url, timeout, retries), url
        except SystemExit as exc:
            errors.append(str(exc))
    raise SystemExit("Failed to fetch all URL candidates:\n" + "\n".join(errors))


def strip_gutenberg_boilerplate(text: str) -> str:
    start_match = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.I | re.S)
    end_match = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.I | re.S)
    if start_match:
        text = text[start_match.end() :]
    if end_match:
        text = text[: end_match.start()]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def resolve_books(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.gutenberg_ids or args.id_range:
        selected_ids: list[str] = []
        if args.gutenberg_ids:
            selected_ids.extend(item.strip() for item in args.gutenberg_ids.split(",") if item.strip())
        if args.id_range:
            try:
                start_raw, end_raw = args.id_range.split(":", 1)
                start_id = int(start_raw)
                end_id = int(end_raw)
            except ValueError as exc:
                raise SystemExit("--id-range must look like START:END") from exc
            if end_id < start_id:
                raise SystemExit("--id-range END must be >= START")
            selected_ids.extend(str(book_id) for book_id in range(start_id, end_id + 1))
        seen: set[str] = set()
        books: list[dict[str, str]] = []
        for book_id in selected_ids:
            if book_id in seen:
                continue
            seen.add(book_id)
            books.append(
                {
                    "id": book_id,
                    "title": f"Gutenberg {book_id}",
                    "author": "",
                    "url": f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
                }
            )
        return books
    if args.book_ids:
        selected_ids = {item.strip() for item in args.book_ids.split(",") if item.strip()}
        known = {book["id"]: book for book in DEFAULT_BOOKS}
        missing = sorted(selected_ids - set(known))
        if missing:
            raise SystemExit(f"Unknown --book-ids: {', '.join(missing)}")
        return [known[book["id"]] for book in DEFAULT_BOOKS if book["id"] in selected_ids]
    if args.manifest:
        payload = load_json(Path(args.manifest))
        books = payload.get("books", payload)
        if not isinstance(books, list):
            raise SystemExit("--manifest must be a list or an object with a books list.")
        return [
            {
                "id": str(item.get("id", index)),
                "title": str(item.get("title", item.get("url", index))),
                "author": str(item.get("author", "")),
                "url": str(item["url"]),
            }
            for index, item in enumerate(books)
        ]
    return list(DEFAULT_BOOKS)


def gutenberg_url_candidates(book: dict[str, str]) -> list[str]:
    book_id = book["id"]
    urls = [book["url"]]
    for suffix in ("-0", ""):
        candidate = f"https://www.gutenberg.org/files/{book_id}/{book_id}{suffix}.txt"
        if candidate not in urls:
            urls.append(candidate)
    return urls


def corpus_stats(text: str) -> dict[str, Any]:
    tokens = tokenize_with_config(text, default_word_tokenizer_config())
    lines = text.splitlines()
    return {
        "characters": len(text),
        "bytes_utf8": len(text.encode("utf-8")),
        "lines": len(lines),
        "word_tokens": len(tokens),
        "unique_word_tokens": len(set(tokens)),
        "sha256": sha256_text(text),
    }


def write_shards(
    sections: list[str],
    *,
    output_dir: Path,
    shard_bytes: int | None,
) -> list[dict[str, Any]]:
    if not shard_bytes or shard_bytes <= 0:
        return []
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, Any]] = []
    current: list[str] = []
    current_bytes = 0

    def flush() -> None:
        nonlocal current, current_bytes
        if not current:
            return
        text = "\n\n\n".join(current).strip() + "\n"
        path = shard_dir / f"shard_{len(shards) + 1:06d}.txt"
        path.write_text(text, encoding="utf-8")
        stats = corpus_stats(text)
        shards.append({"index": len(shards) + 1, "path": str(path), **stats})
        current = []
        current_bytes = 0

    for section in sections:
        section_bytes = len(section.encode("utf-8"))
        if current and current_bytes + section_bytes > shard_bytes:
            flush()
        current.append(section)
        current_bytes += section_bytes
    flush()
    return shards


def ingest_gutenberg(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    books = resolve_books(args)
    if args.max_books is not None:
        books = books[: args.max_books]
    if not books:
        raise SystemExit("No books selected for ingestion.")

    sections: list[str] = []
    book_reports: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    total_bytes = 0
    for book in books:
        cache_path = cache_dir / f"{book['id']}.clean.txt"
        source_url = book["url"]
        cache_hit = False
        if args.use_cache and cache_path.exists():
            print(f"cache {book['id']}: {book['title']}")
            clean_text = cache_path.read_text(encoding="utf-8")
            cache_hit = True
        else:
            print(f"fetch {book['id']}: {book['title']}")
            try:
                raw_text, source_url = fetch_first_url(gutenberg_url_candidates(book), args.timeout, args.retries)
            except SystemExit as exc:
                if not args.continue_on_error:
                    raise
                error = {"id": book["id"], "title": book["title"], "error": str(exc)}
                errors.append(error)
                print(f"skip {book['id']}: {exc}")
                if args.delay_s > 0:
                    time.sleep(args.delay_s)
                continue
            clean_text = strip_gutenberg_boilerplate(raw_text)
            cache_path.write_text(clean_text, encoding="utf-8")
        if args.max_chars_per_book and args.max_chars_per_book > 0:
            clean_text = clean_text[: args.max_chars_per_book].strip()
        if not clean_text:
            raise SystemExit(f"Book became empty after cleaning: {book}")

        candidate_bytes = len(clean_text.encode("utf-8"))
        if args.target_bytes and total_bytes > 0 and total_bytes + candidate_bytes > args.target_bytes:
            print(
                f"stop before {book['id']}: target bytes would be exceeded "
                f"({total_bytes + candidate_bytes:,} > {args.target_bytes:,})"
            )
            break

        raw_path = raw_dir / f"{book['id']}.txt"
        raw_path.write_text(clean_text, encoding="utf-8")
        stats = corpus_stats(clean_text)
        total_bytes += stats["bytes_utf8"]
        book_reports.append(
            {
                **book,
                "source_url": source_url,
                "path": str(raw_path),
                "cache_path": str(cache_path),
                "cache_hit": cache_hit,
                **stats,
            }
        )
        sections.append(
            "\n".join(
                [
                    f"Title: {book['title']}",
                    f"Author: {book['author']}",
                    "",
                    clean_text,
                ]
            )
        )
        if args.target_bytes and total_bytes >= args.target_bytes:
            break
        if args.delay_s > 0:
            time.sleep(args.delay_s)

    if not sections:
        raise SystemExit("No books were ingested. Adjust --target-bytes, --max-books, or book selection.")

    corpus = "\n\n\n".join(sections).strip() + "\n"
    output_text = Path(args.output_text)
    if not args.no_joined_corpus:
        output_text.parent.mkdir(parents=True, exist_ok=True)
        output_text.write_text(corpus, encoding="utf-8")
    shards = write_shards(sections, output_dir=output_dir, shard_bytes=args.shard_bytes)

    report = {
        "created_at": timestamp(),
        "source": "project_gutenberg",
        "book_count": len(book_reports),
        "books": book_reports,
        "error_count": len(errors),
        "errors": errors,
        "output_text": str(output_text) if not args.no_joined_corpus else None,
        "output_json": str(Path(args.output_json)),
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "target_bytes": args.target_bytes,
        "stats": corpus_stats(corpus),
        "shard_bytes": args.shard_bytes,
        "shards": shards,
        "notes": [
            "Project Gutenberg texts are public-domain in the United States, but reuse outside the US may vary.",
            "The script strips Gutenberg boilerplate and keeps cleaned per-book text under output_dir/raw.",
        ],
    }
    write_json(Path(args.output_json), report)
    return report


def ingest_local(args: argparse.Namespace) -> dict[str, Any]:
    if not args.input_text:
        raise SystemExit("--input-text is required for --source local.")
    text = load_text(args.input_text)
    output_text = Path(args.output_text)
    if not args.no_joined_corpus:
        output_text.parent.mkdir(parents=True, exist_ok=True)
        output_text.write_text(text, encoding="utf-8")
    shards = write_shards([text], output_dir=Path(args.output_dir), shard_bytes=args.shard_bytes)
    report = {
        "created_at": timestamp(),
        "source": "local",
        "input_text": args.input_text,
        "output_text": str(output_text) if not args.no_joined_corpus else None,
        "output_json": str(Path(args.output_json)),
        "shard_bytes": args.shard_bytes,
        "shards": shards,
        "stats": corpus_stats(text),
    }
    write_json(Path(args.output_json), report)
    return report


def print_report(report: dict[str, Any]) -> None:
    stats = report["stats"]
    print("")
    print("TextPy/SoA data ingestion")
    print(f"source: {report['source']}")
    print(f"output_text: {report['output_text']}")
    if report.get("shards"):
        print(f"shards: {len(report['shards'])}")
    print(f"characters: {stats['characters']:,}")
    print(f"bytes_utf8: {stats['bytes_utf8']:,}")
    print(f"word_tokens: {stats['word_tokens']:,}")
    print(f"unique_word_tokens: {stats['unique_word_tokens']:,}")
    print(f"sha256: {stats['sha256']}")
    if "book_count" in report:
        print(f"books: {report['book_count']}")
    if report.get("error_count"):
        print(f"errors: {report['error_count']}")
    print(f"metadata: {report['output_json']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest real text data for TextPy/SoA experiments.")
    parser.add_argument("--source", choices=("gutenberg", "local"), default="gutenberg")
    parser.add_argument("--manifest", default=None, help="Optional JSON list of {id,title,author,url} books.")
    parser.add_argument("--book-ids", default=None, help="Comma-separated Gutenberg IDs from the built-in preset.")
    parser.add_argument("--gutenberg-ids", default=None, help="Comma-separated arbitrary Gutenberg IDs.")
    parser.add_argument("--id-range", default=None, help="Inclusive arbitrary Gutenberg ID range START:END.")
    parser.add_argument("--input-text", default=None, help="Local input text for --source local.")
    parser.add_argument("--output-dir", default="artifacts/corpora/gutenberg_demo")
    parser.add_argument("--output-text", default="artifacts/corpora/gutenberg_demo/corpus.txt")
    parser.add_argument("--output-json", default="artifacts/corpora/gutenberg_demo/corpus_manifest.json")
    parser.add_argument("--shard-bytes", type=int, default=None, help="Write corpus shards near this byte size.")
    parser.add_argument("--no-joined-corpus", action="store_true", help="Only write shards plus manifest.")
    parser.add_argument("--max-books", type=int, default=None)
    parser.add_argument("--max-chars-per-book", type=int, default=None)
    parser.add_argument("--target-bytes", type=int, default=None, help="Stop after roughly this many cleaned UTF-8 bytes.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--no-cache", dest="use_cache", action="store_false")
    parser.set_defaults(use_cache=True)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--delay-s", type=float, default=0.0, help="Delay between fetches for polite bulk downloads.")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if args.source == "gutenberg":
        report = ingest_gutenberg(args)
    else:
        report = ingest_local(args)
    print_report(report)


if __name__ == "__main__":
    run(parse_args())
