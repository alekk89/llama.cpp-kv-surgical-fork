# KV surgery technical guide

## Contract

This experimental fork gives a router explicit ownership of one text-only `llama-server` slot. The router supplies exact target-model token IDs, absolute half-open ranges, and an optimistic concurrency revision. The server can:

- prefill an empty managed slot without generation;
- delete an interior attention-KV range without decoding the retained suffix;
- replace a range with an equal or shorter marker or summary, decoding only the replacement;
- append exact tokens at the logical tail;
- continue generation through the normal llama.cpp scheduler; and
- mirror supported edits into a DFlash companion cache.

This is not exact context rewriting. Qwen 3.6 is a hybrid attention/recurrent model. Attention cells are released, but retained recurrent state is restored after an interior replacement and can still contain influence from removed content. Use an authoritative full rebuild when exact deletion semantics matter.

The API remains version 1 and is intentionally separate from OpenAI-compatible completion routes. It is currently tested only with Qwen 3.6 27B. DFlash is supported and tested. MTP is not supported after an interior edit. DSpark is present in upstream llama.cpp but is not enabled or validated for this managed dual-edit path.

Managed operations reject servers configured with an `mmproj`, including text-only slots on such a server. Managed position holes use `LLAMA_TOKEN_NULL`; rejecting `mmproj` avoids confusing those holes with multimodal placeholders.

## Slot state machine

Every slot exposes:

| Field | Meaning |
| --- | --- |
| `managed_revision` | Monotonic concurrency guard |
| `managed_append_only` | The router owns the slot; ordinary completion is rejected |
| `managed_draft_coherent` | Every acknowledged managed mutation also reached the supported model-backed draft cache |
| `managed_requires_rebuild` | A mutation failed after cache state may have changed |

A fresh slot is unmanaged at revision 0. Every successful managed prefill, edit, append, or managed generation request increments the revision once. Every such request must include the last acknowledged revision as `expected_revision`.

Validation failures, missing revisions, and stale revisions do not mutate state or increment the revision. If target or draft mutation has started and the operation then fails, the server increments the revision, sets `managed_requires_rebuild: true`, disables draft coherence, and rejects all later managed operations. `erase` or a successful slot restore clears this state. A router should normally erase and rebuild from its authoritative token ledger.

The server restores the exact prior flags and revision if a managed-native request cannot be scheduled, so a launch failure is not reported as a committed mutation.

## Endpoint contract

The managed capability object is returned by `GET /props`. Important discovery fields include:

```json
{
  "api_version": 1,
  "requires_expected_revision": true,
  "reports_rebuild_required": true,
  "supports_mmproj": false,
  "supports_context_shift": false,
  "supports_slot_save": false,
  "native_completion": true,
  "qwen_attention_only_dflash_dual_edit": true,
  "qwen_compact_positions": false
}
```

### Native prefill

```json
POST /slots/0?action=managed_native_prefill
{
  "expected_revision": 0,
  "tokens": [151644, 8948, 198, 2610]
}
```

The slot must be empty and unmanaged. Token IDs must be valid for the target vocabulary. The response contains the appended range and revision 1.

### Delete or summarize

```json
POST /slots/0?action=kv_edit
{
  "expected_revision": 1,
  "experimental_attention_only": true,
  "compact_positions": false,
  "edits": [
    {"start_pos": 1176, "end_pos": 1243, "tokens": []},
    {"start_pos": 1650, "end_pos": 1706, "tokens": [101, 202, 303]}
  ]
}
```

`start_pos` is inclusive and `end_pos` is exclusive. Edits must be ordered, non-overlapping, inside the current attention range, and use valid IDs. A replacement may not exceed the removed range. For Qwen, `experimental_attention_only` must be true and `compact_positions` must remain false.

Empty replacement arrays delete ranges without prefill. Non-empty replacement arrays decode only the supplied marker or summary at the old range start. Later cached positions remain unchanged and are not re-prefilled. Unused positions become logical holes and their attention cells are available for reuse.

### Append without generation

```json
POST /slots/0?action=kv_append
{
  "expected_revision": 2,
  "tokens": [606, 707, 808]
}
```

The response includes `pos_start`, `pos_end`, `n_appended`, and `token_probe`. Large appends are decoded in server-batch-sized chunks. A partial decode failure poisons the slot instead of leaving its usability ambiguous.

### Append and generate

```json
POST /slots/0?action=managed_native_completion
{
  "expected_revision": 3,
  "tokens": [606, 707, 808],
  "n_predict": 128,
  "temperature": 0.2,
  "stream": false
}
```

This is the preferred generation route. It passes a position-aligned virtual prompt to the normal scheduler. Longest-prefix matching keeps the physical edited cache and evaluates only the new tail. Omitted sampling fields and `n_predict` use server defaults.

Streaming responses use `managed_native_begin`, `managed_native_delta`, and `managed_native_final`, followed by `[DONE]`. Deltas include exact generated token IDs. The final response includes the append count, generated IDs and text, timings, managed positions, revision, draft coherence, and rebuild state.

`managed_generate` is retained as a deprecated compatibility endpoint. It uses a separate explicit generation loop and should not be selected for new router integrations.

Both generation actions accept an empty token array only when `continue_generation: true` is set.

## Internal design

The managed prompt ledger preserves absolute positions. Non-compacting edits store `LLAMA_TOKEN_NULL` at released positions. Admission checks, context limits, context shifting, and speculative headroom count only live non-null text tokens, while the ledger high-water mark determines the next absolute position.

Ordinary context shifting is disabled for managed slots even when the server-wide option is enabled. Generation stops at the live context limit instead of compacting router-owned positions. The router must delete, summarize, or rebuild to create more capacity.

Unified-KV idle-slot pressure does not automatically purge a router-owned managed slot. Use a fixed slot ID and explicit erase so ownership changes remain revisioned and observable.

The upstream slot-save format does not preserve managed position holes and router metadata. Saving a managed slot is therefore rejected. Restore may be used only with an ordinary pre-managed snapshot and returns the slot to an unmanaged state; authoritative router rebuild remains the normal recovery path.

Fork-only low-level operations are kept out of the stable public `llama.h` ABI:

- `llama_decode_ext(..., LLAMA_DECODE_FLAG_ALLOW_NONSEQUENTIAL)` lives in the existing `src/llama-ext.h` staging API;
- attention-only removal and range queries also live in `src/llama-ext.h`;
- `llama_batch` is unchanged from upstream;
- non-sequential permission is passed as an internal allocator/decode argument; and
- memory support is queried through default-unsupported virtual capabilities rather than server-side casts to a concrete hybrid type.

`llama_memory_hybrid` and `llama_memory_hybrid_iswa` expose attention-only capabilities. Unsupported memory layouts return false or -1 and the server returns a not-supported error before mutation.

For attention-only Qwen replacement, the server snapshots the partial recurrent state, decodes replacement tokens into the released attention positions, then restores recurrent state. This deliberately avoids recomputing the retained suffix, but it is the source of the approximation boundary.

DFlash removes the same draft-cache range and injects replacement K/V from target features at the original positions. M-RoPE positions are expanded locally for the DFlash injection batch. Draft coherence is acknowledged only after both caches succeed. Target-only edits clear stale draft state and prevent model-backed speculation.

## Build and test

Build the server:

```powershell
cmake -S . -B build -DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build --config Release --target llama-server -j 8
```

Run the focused regression:

```powershell
Set-Location tools\server\tests
python -m pytest unit\test_slot_kv_edit.py -v
```

The regression covers capability discovery, legacy compatibility, native prefill/edit/append/generation, streaming records, required and stale revisions, rejection atomicity, ordinary-completion isolation, managed save rejection, induced partial-decode poisoning, erase recovery, and stopping at capacity without context shift.

Run the real Qwen capacity harness against an erased idle slot:

```powershell
python tools\server\tests\kv_surgery_capacity.py `
  --server-url http://127.0.0.1:8080 `
  --slot-id 0 `
  --ctx-size 4096 `
  --segments 20 `
  --segment-tokens 100
```

The harness discovers the current slot revision, performs managed native prefill, removes alternating router chunks, appends until live entries exactly equal the configured context, and fails on any rebuild-required response.

## Validation record

On 2026-08-02, the hardened working tree passed the focused six-test CUDA server regression, the existing slot save/restore suite, a complete configured CMake build, and all 42 native CTest entries after fetching the repository's tiny model fixture. It also passed the following tests on two RTX 3090 GPUs:

| Configuration | Result |
| --- | --- |
| Qwen 3.6 27B Q8_0, 4096 context, Q8_0 target K/V | 2010 source tokens -> 1136 live entries after 994 removed and 120 inserted; 2960 tail tokens refilled all 4096 cells; logical tail ended at 4883 |
| Qwen 3.6 27B Q8_0 managed generation | 20-token range replaced by 7 summary tokens; only 11 new tail tokens evaluated; 32 tokens generated; rebuild state stayed false |
| Same target plus Qwen 3.6 27B DFlash Q8_0, F16 draft K/V | Same full capacity refill; target and draft remained coherent |
| DFlash managed generation | Only 11 new tail tokens evaluated; 64 generated at 56.36 tokens/s; 48 of 204 draft tokens accepted; draft coherence true and rebuild state false |

The target was 29,047,084,160 bytes with SHA-256 `88a2cd21a43cac266c74033f426d2929738960f139405758c11202336f916707`. DFlash was 1,849,481,312 bytes with SHA-256 `82545cd07d06cceaeb12feaa6ca76494214c872f83f71282bc6c009de61f30db`. These results establish implementation behavior, not semantic equivalence to a full rebuild.

## Carrying the fork forward

Run the committed overlap and merge audit before updating:

```powershell
.\scripts\check-kv-surgery-upstream.ps1 -Fetch
```

The script reports the merge base, fork/upstream file counts, overlapping paths, and `git merge-tree` conflicts without changing the worktree. A dirty-tree warning means uncommitted hardening is not included in that audit.

The intended touch points are:

| Layer | Files |
| --- | --- |
| Staging decode API | `src/llama-ext.h`, `src/llama-context.h`, `src/llama-context.cpp`, `src/llama-batch.h`, `src/llama-batch.cpp` |
| Memory capability | `src/llama-memory.h`, `src/llama-memory-hybrid.*`, `src/llama-memory-hybrid-iswa.*` |
| DFlash injection | `common/speculative.cpp` |
| Position ledger | `tools/server/server-common.h`, `tools/server/server-common.cpp` |
| Managed API | `tools/server/server-task.h`, `tools/server/server-task.cpp`, `tools/server/server-context.h`, `tools/server/server-context.cpp` |
| Tests | `tools/server/tests/unit/test_slot_kv_edit.py`, `tools/server/tests/kv_surgery_capacity.py` |

For each upstream update:

1. Fetch upstream and run the overlap audit.
2. Rebase the small logical fork layers instead of copying an old `server-context.cpp` wholesale.
3. Preserve upstream slot scheduling, memory, batching, and streaming changes during conflict resolution.
4. Build `llama-server` and run the complete focused regression.
5. Run both target-only and DFlash real-model lifecycles, including a full capacity refill and managed-native generation.
6. Record the upstream commit, model hashes, backend, context, cache types, throughput, draft acceptance, and capacity JSON.

Do not enable Qwen position compaction, MTP continuation, DSpark dual editing, or multimodal managed slots as incidental conflict resolutions. Each requires a separate model-specific design and validation effort.
