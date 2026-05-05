"""Preprocess SciDER trajectory JSONL files for SFT training.

Reads one or more JSONL files (output of ``prepare_data.py``) and applies
configurable cleanup steps:

1. ``--merge-consecutive``  Merge consecutive same-role messages into one
   (e.g. two user / two assistant turns in a row → joined with a newline).
2. ``--max-tool-tokens N``  Truncate any ``role:"tool"`` message whose content
   exceeds N tokens, keeping the head and tail with an ellipsis marker.
3. ``--max-message-tokens N``  Split a row whose ``messages`` array sums to
   more than N tokens into multiple new rows. Splits happen at user-turn
   boundaries so tool_call / tool_result pairs stay together.
4. ``--minimal``  Keep only ``id`` and ``messages`` in each output row, plus
   a new ``datasource`` field set to the input filename (without ``.jsonl``).

Tokenization uses ``tiktoken`` with the ``cl100k_base`` encoding by default
(GPT-4 tokenizer). It is an approximation — Gemini / Claude tokens may
differ by ±20% — but is consistent and dependency-light.

Usage::

    python train/preprocess_data.py \\
        --input  train/raw_datafiles/ \\
        --out    train/dataset/datafiles/ \\
        --merge-consecutive \\
        --max-tool-tokens 512 \\
        --max-message-tokens 8192 \\
        --minimal

Each input ``foo.jsonl`` is written to ``<out>/foo.jsonl`` so HF
``load_dataset`` glob pattern still works.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

try:
    import tiktoken
except ImportError:  # pragma: no cover
    print(
        "preprocess_data.py requires tiktoken — `uv add tiktoken` or `pip install tiktoken`",
        file=sys.stderr,
    )
    raise


# --- Token counting -------------------------------------------------------


def _make_encoder(name: str):
    return tiktoken.get_encoding(name)


def _count_tokens(text: str | None, enc) -> int:
    if not text:
        return 0
    return len(enc.encode(text, disallowed_special=()))


def _count_message_tokens(msg: dict, enc) -> int:
    """Approximate token count for a single message: role + content + tool_calls."""
    n = 4  # rough overhead for role + delimiters
    n += _count_tokens(msg.get("content"), enc)
    for tc in msg.get("tool_calls") or []:
        # Each tool call contributes its serialized function block.
        n += _count_tokens(json.dumps(tc, ensure_ascii=False), enc)
    return n


# --- Step 1: merge consecutive same-role messages -------------------------


def merge_consecutive(messages: list[dict]) -> list[dict]:
    """Merge consecutive messages with the same role into one.

    Tool messages are NEVER merged (each tool_result keeps its own
    tool_call_id; LF's converter joins them itself with a
    ``</tool_response><tool_response>`` separator).

    User and assistant turns are merged aggressively:
      - text contents are concatenated with ``\\n\\n``
      - ``tool_calls`` lists are accumulated, so an assistant text turn
        followed by an assistant tool-call turn collapses into one
        assistant message that has both — which preserves SFT alternation.
    """
    if not messages:
        return messages

    out: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        last = out[-1] if out else None
        if last is not None and role == last.get("role") and role != "tool":
            a = last.get("content") or ""
            b = msg.get("content") or ""
            sep = "\n\n" if a and b else ""
            last["content"] = a + sep + b
            extra_tc = msg.get("tool_calls") or []
            if extra_tc:
                last["tool_calls"] = (last.get("tool_calls") or []) + list(extra_tc)
        else:
            out.append(dict(msg))
    return out


# --- Step 2: truncate large tool messages ---------------------------------


def _truncate_with_ellipsis(text: str, max_tokens: int, enc) -> str:
    """Keep head + tail of a long string, joined by an ellipsis marker."""
    if max_tokens <= 0:
        return text
    toks = enc.encode(text, disallowed_special=())
    if len(toks) <= max_tokens:
        return text
    # Reserve some budget for the marker text itself.
    marker_template = "\n... [truncated {n} tokens] ...\n"
    marker_budget = 24
    head_n = max(1, int((max_tokens - marker_budget) * 0.7))
    tail_n = max(1, max_tokens - marker_budget - head_n)
    head = enc.decode(toks[:head_n])
    tail = enc.decode(toks[-tail_n:])
    dropped = len(toks) - head_n - tail_n
    return head + marker_template.format(n=dropped) + tail


def truncate_tool_messages(messages: list[dict], max_tokens: int, enc) -> list[dict]:
    """Truncate every ``role:"tool"`` message whose content exceeds ``max_tokens``."""
    out = []
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("content"):
            new_content = _truncate_with_ellipsis(msg["content"], max_tokens, enc)
            if new_content != msg["content"]:
                msg = {**msg, "content": new_content}
        out.append(msg)
    return out


# --- Step 3: chunk long message lists -------------------------------------


def _split_into_groups(messages: list[dict]) -> list[list[dict]]:
    """Group messages such that each group is an atomic unit safe to split on.

    A new group starts at every ``role:"user"`` message; everything that
    follows (assistant / tool_calls / tool results) stays glued to it.
    Anything before the first user message becomes a leading group on its own.
    """
    groups: list[list[dict]] = []
    current: list[dict] = []
    for msg in messages:
        if msg.get("role") == "user" and current:
            groups.append(current)
            current = []
        current.append(msg)
    if current:
        groups.append(current)
    return groups


def chunk_messages(
    messages: list[dict],
    max_tokens: int,
    enc,
    *,
    on_oversize_warning=None,
) -> list[list[dict]]:
    """Split ``messages`` into chunks, each ≤ ``max_tokens``.

    Splits at user-turn boundaries; assistant + tool_call/result groups
    stay together. A single group exceeding ``max_tokens`` is emitted as
    one chunk anyway (with a warning), since further splitting would break
    tool linkage.
    """
    if max_tokens <= 0:
        return [messages]

    groups = _split_into_groups(messages)

    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0
    for group in groups:
        g_tokens = sum(_count_message_tokens(m, enc) for m in group)
        if g_tokens > max_tokens and on_oversize_warning is not None:
            on_oversize_warning(g_tokens)
        if current and current_tokens + g_tokens > max_tokens:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.extend(group)
        current_tokens += g_tokens
    if current:
        chunks.append(current)
    return chunks


# --- Field normalization -------------------------------------------------

# Keys kept on each message in the final SFT output. Anything else (debug
# fields like is_compact_boundary, compact_metadata, is_meta, agent_sender,
# tool_name) is dropped so HF Datasets can infer a stable schema across all
# rows. If even one row carries an extra field, ``cast_array_to_feature``
# fails with a struct mismatch during dataset preparation.
_CANONICAL_MSG_KEYS = ("role", "content", "reasoning_content", "tool_calls", "tool_call_id")


def _drop_orphan_tool_calls(messages: list[dict]) -> list[dict]:
    """Strip ``tool_calls`` from any assistant turn whose tool calls are
    never answered.

    LF treats an assistant with ``tool_calls`` as the FUNCTION role and
    requires the very next slot to be an OBSERVATION (tool). When SciDER's
    critic retry / user_review flow forces a USER message immediately
    after — without giving the tool call a chance to execute — we get
    ``assistant_with_tc → user`` which breaks alternation.

    We drop the orphan ``tool_calls`` (the call never ran in the captured
    trajectory anyway) and, if that leaves the assistant message with
    empty content, drop the message entirely.
    """
    out: list[dict] = []
    for i, m in enumerate(messages):
        if (
            m.get("role") == "assistant"
            and m.get("tool_calls")
            and (i + 1 >= len(messages) or messages[i + 1].get("role") != "tool")
        ):
            content = m.get("content") or ""
            if content.strip():
                cleaned = dict(m)
                cleaned["tool_calls"] = []
                out.append(cleaned)
            # else: drop the empty orphan assistant outright
        else:
            out.append(m)
    return out


def _trim_to_valid_window(messages: list[dict]) -> list[dict]:
    """Drop leading non-user and trailing non-assistant messages.

    LF accepts trajectories that end either with a plain assistant turn
    or with an assistant turn carrying tool_calls (it gets mapped to
    FUNCTION). Trim only the obvious junk: lead-in non-user fragments
    and trailing tool / user messages that have no follow-up.
    """
    start = 0
    while start < len(messages) and messages[start].get("role") != "user":
        start += 1
    end = len(messages)
    while end > start and messages[end - 1].get("role") != "assistant":
        end -= 1
    return messages[start:end]


def _is_valid_alternation(messages: list[dict]) -> bool:
    """Mirror LF's OpenAIDatasetConverter alternation rule.

    Reject rows that the converter would mark ``broken_data`` and emit as
    null _prompt/_response — those break HF Datasets cross-shard alignment.

    LF's converter coalesces consecutive ``tool`` messages into a single
    "observation", then enforces strict alternation:
      - even slots: user / tool-block
      - odd slots:  assistant (with or without tool_calls)
      - even total slot count, last slot must be assistant.

    We replicate the same coalescing here before checking.
    """
    if not messages:
        return False
    aligned_roles: list[str] = []
    prev_was_tool = False
    for m in messages:
        role = m.get("role")
        if role == "tool":
            if not prev_was_tool:
                aligned_roles.append("tool")
            prev_was_tool = True
        else:
            aligned_roles.append(role or "")
            prev_was_tool = False

    if len(aligned_roles) < 2 or len(aligned_roles) % 2 != 0:
        return False
    for i, role in enumerate(aligned_roles):
        if i % 2 == 0:
            if role not in ("user", "tool"):
                return False
        else:
            if role != "assistant":
                return False
    return aligned_roles[-1] == "assistant"


def _normalize_tool_call(tc: dict) -> dict:
    """Return a canonical tool_call dict.

    ``arguments`` is kept as a JSON-encoded *string* (the OpenAI wire format)
    on purpose — turning it into a dict makes HF Arrow infer a giant union
    struct over every tool's argument keys, which collapses with type
    conflicts across rows (e.g. ``offset`` int vs float).

    LF's ``_parse_functions`` is patched to accept already-string arguments
    so the string flows through to Qwen's tool formatter unchanged.
    """
    fn = tc.get("function") or {}
    args = fn.get("arguments")
    if not isinstance(args, str):
        args = json.dumps(args if args is not None else {}, ensure_ascii=False)
    return {
        "id": tc.get("id") or "",
        "function": {
            "name": fn.get("name") or "",
            "arguments": args,
        },
    }


def _normalize_messages(messages: list[dict], *, drop_reasoning: bool = False) -> list[dict]:
    out = []
    for m in messages:
        role = m.get("role")
        if not role:
            continue
        # Emit every canonical key with a non-null default so HF Arrow
        # produces a uniform schema and downstream converters can call
        # ``len(msg["tool_calls"])`` without hitting None. ``drop_reasoning``
        # forces ``reasoning_content`` to "" for every message — used when
        # the trainer's template would otherwise consume the reasoning
        # string. Stays as empty string (not null) for schema uniformity
        # with the default code path.
        if drop_reasoning:
            reasoning = ""
        else:
            reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
        cleaned = {
            "role": role,
            "content": m.get("content") if m.get("content") is not None else "",
            "reasoning_content": reasoning,
            "tool_calls": [_normalize_tool_call(tc) for tc in (m.get("tool_calls") or [])],
            "tool_call_id": m.get("tool_call_id") or "",
        }
        out.append(cleaned)
    return out


# --- Top-level row processing --------------------------------------------


def process_row(
    row: dict,
    *,
    datasource: str,
    do_merge: bool,
    max_tool_tokens: int | None,
    max_message_tokens: int | None,
    minimal: bool,
    drop_reasoning: bool,
    enc,
) -> Iterator[dict]:
    """Run the configured pipeline on a single row, yielding one or more rows."""
    messages = list(row.get("messages") or [])
    if not messages:
        return

    if do_merge:
        messages = merge_consecutive(messages)
    if max_tool_tokens is not None:
        messages = truncate_tool_messages(messages, max_tool_tokens, enc)

    if max_message_tokens is not None:
        oversized = [False]

        def _warn(g_tokens: int) -> None:
            oversized[0] = True
            print(
                f"[warn] row {row.get('id')!r}: an atomic group is "
                f"{g_tokens} tokens (> {max_message_tokens}); kept whole",
                file=sys.stderr,
            )

        chunks = chunk_messages(messages, max_message_tokens, enc, on_oversize_warning=_warn)
    else:
        chunks = [messages]

    base_id = row.get("id", "<no-id>")
    for i, chunk in enumerate(chunks):
        new_id = base_id if len(chunks) == 1 else f"{base_id}#part{i + 1:02d}"
        # Always normalize message fields — keeps the JSONL schema flat and
        # uniform so HF Datasets can infer a single Features struct.
        norm_chunk = _normalize_messages(chunk, drop_reasoning=drop_reasoning)
        # Drop / cleanse orphaned tool_calls (assistant_with_tc → user without
        # an intervening tool result) — common after critic-retry interrupts.
        norm_chunk = _drop_orphan_tool_calls(norm_chunk)
        # Repair leading/trailing junk so partial trajectories still train.
        norm_chunk = _trim_to_valid_window(norm_chunk)
        # Skip chunks LF would refuse anyway. Otherwise the OpenAI converter
        # emits null prompt/response and breaks HF cross-shard alignment.
        if not _is_valid_alternation(norm_chunk):
            print(
                f"[skip] row {new_id!r}: roles do not alternate user/tool → assistant",
                file=sys.stderr,
            )
            continue
        if minimal:
            yield {"id": new_id, "messages": norm_chunk, "datasource": datasource}
        else:
            out = {**row, "id": new_id, "messages": norm_chunk, "datasource": datasource}
            yield out


# --- File / directory iteration -------------------------------------------


def _iter_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.glob("*.jsonl") if p.is_file())
    raise FileNotFoundError(f"--input not found: {input_path}")


def process_file(
    in_path: Path,
    out_path: Path,
    *,
    do_merge: bool,
    max_tool_tokens: int | None,
    max_message_tokens: int | None,
    minimal: bool,
    drop_reasoning: bool,
    enc,
) -> tuple[int, int]:
    """Process one JSONL file. Returns (rows_in, rows_out)."""
    datasource = in_path.stem
    rows_in = 0
    rows_out = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[skip] {in_path.name}: invalid JSON line — {e}", file=sys.stderr)
                continue
            rows_in += 1
            for new_row in process_row(
                row,
                datasource=datasource,
                do_merge=do_merge,
                max_tool_tokens=max_tool_tokens,
                max_message_tokens=max_message_tokens,
                minimal=minimal,
                drop_reasoning=drop_reasoning,
                enc=enc,
            ):
                fout.write(json.dumps(new_row, ensure_ascii=False))
                fout.write("\n")
                rows_out += 1
    return rows_in, rows_out


# --- CLI ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file or directory of JSONL files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory (mirrors input filenames). Created if missing.",
    )
    parser.add_argument(
        "--merge-consecutive",
        action="store_true",
        help="Merge consecutive same-role messages into one (skips tool / tool_calls).",
    )
    parser.add_argument(
        "--max-tool-tokens",
        type=int,
        default=None,
        help="Truncate tool messages exceeding N tokens (recommend 512). " "Off when omitted.",
    )
    parser.add_argument(
        "--max-message-tokens",
        type=int,
        default=None,
        help="Split a row's messages so each chunk ≤ N tokens (recommend 8192). "
        "Off when omitted.",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Output rows keep only id, messages, datasource.",
    )
    parser.add_argument(
        "--drop-reasoning",
        action="store_true",
        help="Force every output message's reasoning_content to null. Use "
        "when training a non-thinking template that would otherwise consume "
        "the reasoning string. Schema field is still emitted for uniformity.",
    )
    parser.add_argument(
        "--encoding",
        default="cl100k_base",
        help="tiktoken encoding name. Default: cl100k_base (GPT-4 family).",
    )
    args = parser.parse_args()

    files = _iter_input_files(args.input)
    if not files:
        print(f"No .jsonl files found under {args.input}", file=sys.stderr)
        return 2

    enc = _make_encoder(args.encoding)
    args.out.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0
    for f in files:
        out_path = args.out / f.name
        n_in, n_out = process_file(
            f,
            out_path,
            do_merge=args.merge_consecutive,
            max_tool_tokens=args.max_tool_tokens,
            max_message_tokens=args.max_message_tokens,
            minimal=args.minimal,
            drop_reasoning=args.drop_reasoning,
            enc=enc,
        )
        total_in += n_in
        total_out += n_out
        print(f"  {f.name}: {n_in} → {n_out} rows")

    print(f"Done. {total_in} input rows → {total_out} output rows across {len(files)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
