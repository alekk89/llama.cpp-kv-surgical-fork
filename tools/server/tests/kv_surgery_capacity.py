#!/usr/bin/env python3
"""Run the Qwen managed-slot attention-KV capacity-reuse integration test.

Start llama-server with a fresh slot and --ctx-size matching --ctx-size below.
This is not a CI unit test because it needs a Qwen hybrid model and a live server.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def request(server_url, path, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        server_url + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{path} returned HTTP {exc.code}: {detail}") from exc


def get_json(server_url, path, timeout=60):
    try:
        with urllib.request.urlopen(server_url + path, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{path} returned HTTP {exc.code}: {detail}") from exc


def tokenize(server_url, text):
    return request(server_url, "/tokenize", {"content": text}, 60)["tokens"]


def make_segment(server_url, index, target_tokens):
    segment_id = f"2_seg{index:04d}"
    prefix = f"[segment {segment_id} keep]\nRouter memory object {index}: "
    low = 0
    high = target_tokens * 3
    best = None

    while low <= high:
        middle = (low + high) // 2
        text = prefix + " operational-detail" * middle + "\n"
        count = len(tokenize(server_url, text))
        if count <= target_tokens:
            best = text
            low = middle + 1
        else:
            high = middle - 1

    if best is None:
        raise RuntimeError(f"could not create segment {segment_id}")
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--ctx-size", type=int, default=12288)
    parser.add_argument("--segments", type=int, default=50)
    parser.add_argument("--segment-tokens", type=int, default=200)
    parser.add_argument("--append-batch", type=int, default=256)
    args = parser.parse_args()

    if args.segments < 2 or args.segments % 2:
        parser.error("--segments must be an even number of at least 2")

    server_url = args.server_url.rstrip("/")
    system = (
        "<|im_start|>system\nYou are a router-aware assistant.\n"
        "<|im_end|>\n<|im_start|>user\nSession memory objects:\n"
    )
    parts = [
        make_segment(server_url, index, args.segment_tokens)
        for index in range(1, args.segments + 1)
    ]
    prompt = system + "".join(parts)
    prompt_tokens = tokenize(server_url, prompt)
    if len(prompt_tokens) >= args.ctx_size:
        raise RuntimeError("fixture is larger than --ctx-size")

    starts = []
    prefix = system
    for part in parts:
        starts.append(len(tokenize(server_url, prefix)))
        prefix += part

    slot_state = next(
        (slot for slot in get_json(server_url, "/slots") if slot["id"] == args.slot_id),
        None,
    )
    if slot_state is None:
        raise RuntimeError(f"slot {args.slot_id} is not available")
    if slot_state["is_processing"] or slot_state["managed_append_only"]:
        raise RuntimeError("capacity test requires an erased, idle slot")

    prefill = request(
        server_url,
        f"/slots/{args.slot_id}?action=managed_native_prefill",
        {
            "expected_revision": slot_state["managed_revision"],
            "tokens": prompt_tokens,
        },
        900,
    )
    if prefill["managed_revision"] != slot_state["managed_revision"] + 1 or prefill["managed_requires_rebuild"]:
        raise RuntimeError("managed prefill did not establish a healthy revision")
    revision = prefill["managed_revision"]

    edits = []
    for index in range(2, args.segments + 1, 2):
        marker = f"[section removed: 2_seg{index:04d}]\n"
        part = parts[index - 1]
        edits.append(
            {
                "start_pos": starts[index - 1],
                "end_pos": starts[index - 1] + len(tokenize(server_url, part)),
                "tokens": tokenize(server_url, marker),
            }
        )

    started = time.monotonic()
    edit = request(
        server_url,
        f"/slots/{args.slot_id}?action=kv_edit",
        {
            "expected_revision": revision,
            "experimental_attention_only": True,
            "compact_positions": False,
            "edits": edits,
        },
        900,
    )
    revision = edit["managed_revision"]
    if edit["managed_requires_rebuild"]:
        raise RuntimeError("managed edit unexpectedly requires rebuild")
    active_after_edit = (
        len(prompt_tokens) - edit["n_removed"] + edit["n_inserted"]
    )
    if active_after_edit >= args.ctx_size:
        raise RuntimeError("edit did not leave capacity for new tail tokens")

    tail_tokens = tokenize(server_url, " new-tail-router-event" * (args.ctx_size * 2))
    required = args.ctx_size - active_after_edit
    if len(tail_tokens) < required:
        raise RuntimeError("tail token source is too short")

    appended = 0
    last_append = None
    while appended < required:
        size = min(args.append_batch, required - appended)
        last_append = request(
            server_url,
            f"/slots/{args.slot_id}?action=kv_append",
            {
                "expected_revision": revision,
                "tokens": tail_tokens[appended:appended + size],
            },
            900,
        )
        revision = last_append["managed_revision"]
        if last_append["managed_requires_rebuild"]:
            raise RuntimeError("managed append unexpectedly requires rebuild")
        appended += size

    result = {
        "ctx_capacity": args.ctx_size,
        "source_tokens": len(prompt_tokens),
        "n_removed": edit["n_removed"],
        "n_inserted": edit["n_inserted"],
        "active_after_edit": active_after_edit,
        "physical_cells_freed": len(prompt_tokens) - active_after_edit,
        "new_tail_tokens": appended,
        "estimated_active_entries": active_after_edit + appended,
        "logical_pos_end": last_append["pos_end"],
        "edit_elapsed_s": round(time.monotonic() - started, 3),
    }
    print(json.dumps(result, indent=2))

    if result["estimated_active_entries"] != args.ctx_size:
        raise RuntimeError("managed append did not refill the configured capacity")
    if result["logical_pos_end"] <= len(prompt_tokens):
        raise RuntimeError("tail positions did not advance beyond the original prompt")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"kv surgery capacity test failed: {exc}", file=sys.stderr)
        sys.exit(1)
