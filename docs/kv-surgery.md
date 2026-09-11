# KV surgery setup and technical guide

## Setup

This fork is the managed runtime used by Spomin. Build and run `llama-server`, then let the router own one fixed managed slot. The documented integration target is Qwen3.8-27B with a matching DFlash2 sidecar.

### Requirements

- CMake and a C++ compiler supported by upstream `llama.cpp`.
- A Qwen3.8-27B GGUF target model for the current integration path.
- Optionally, the matching DFlash2 GGUF sidecar.
- A writable slot-save directory. Managed slots require `--slot-save-path`.
- Enough CPU RAM or VRAM for the model, draft model, and KV-cache configuration.

CUDA is optional. Replace `GGML_CUDA` with the backend appropriate for your machine. On Windows, use Visual Studio 2022 Build Tools with C++, CMake, Git, and a compatible CUDA toolkit. On Linux, use Git, CMake, Python 3, and the compiler/backend toolchain for your system.

### Build

```sh
git clone --branch experimental/kv-surgery-dflash https://github.com/alekk89/llama.cpp-kv-surgical-fork.git
cd llama.cpp-kv-surgical-fork
cmake -S . -B build -DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build --config Release --target llama-server -j 8
```

The normal binary locations are `build\\bin\\Release\\llama-server.exe` on Windows multi-config builds and `build/bin/llama-server` on Linux or macOS single-config builds.

### Start the server

Create a writable slot directory and start one fixed slot. Adjust model filenames, cache types, GPU offload, context size, and draft size for your machine.

Windows PowerShell with DFlash2:

```powershell
New-Item -ItemType Directory -Force .\\tmp\\slots | Out-Null
.\\build\\bin\\Release\\llama-server.exe `
  -m C:\\models\\Qwen3.8-27B-Q8_0.gguf `
  -md C:\\models\\Qwen3.8-27B-DFlash2-Q4_K_M.gguf `
  --spec-type draft-dflash `
  --spec-draft-n-max 15 `
  --ctx-size 12288 `
  --parallel 1 `
  --slot-save-path .\\tmp\\slots `
  --slots `
  --no-webui `
  -fa on `
  -ngl 999
```

Linux or macOS without a draft sidecar:

```sh
mkdir -p ./tmp/slots
./build/bin/llama-server \
  -m /models/Qwen3.8-27B-Q8_0.gguf \
  --ctx-size 12288 \
  --parallel 1 \
  --slot-save-path ./tmp/slots/ \
  --slots \
  --no-webui \
  -fa on \
  -ngl 999
```

The DFlash2 selector is read from the draft GGUF metadata. The draft size is clamped to the sidecar's trained block size. Keep the server bound to `127.0.0.1` unless authentication and a trusted network boundary are configured.

## Use

The managed API is separate from ordinary OpenAI-compatible completion requests. A router must own the exact token IDs, absolute ranges, managed revision, slot ID, and later appends.

### Discover capabilities

The server defaults to `http://127.0.0.1:8080`:

```sh
curl -sS http://127.0.0.1:8080/props
curl -sS http://127.0.0.1:8080/slots
```

In Windows PowerShell, use `curl.exe` if `curl` resolves to `Invoke-WebRequest`. `/props` should advertise `managed_slot` with `api_version: 1`, `edit: true`, `append: true`, `native_completion: true`, and the required DFlash2 dual-edit and compaction capabilities. Each slot reports `managed_revision`, `managed_draft_coherent`, and `managed_requires_rebuild`.

### Endpoint summary

| Endpoint | Purpose |
| --- | --- |
| `GET /props` | Discover managed-slot capabilities and server defaults |
| `GET /slots` | Inspect slot state and managed revision |
| `POST /tokenize` | Convert router text to exact target-model token IDs |
| `POST /detokenize` | Convert token IDs back to text for diagnostics |
| `POST /slots/{id}?action=managed_native_prefill` | Fill an empty slot without generation |
| `POST /slots/{id}?action=kv_edit` | Delete, replace, or compact cached ranges |
| `POST /slots/{id}?action=kv_append` | Decode exact tokens at the logical tail |
| `POST /slots/{id}?action=managed_native_completion` | Append and generate through the normal scheduler |
| `POST /slots/{id}?action=cancel` | Cancel a managed generation task for a slot |
| `POST /slots/{id}?action=erase` | Clear a slot before an authoritative rebuild |

For image-aware slots, use the separate [managed vision guide](managed-vision.md). Do not cut a generic text edit through an image span.

### Managed workflow

Set a base URL for the examples:

```sh
BASE=http://127.0.0.1:8080
```

In PowerShell, use `$BASE = "http://127.0.0.1:8080"` and replace `curl` with `curl.exe`.

#### 1. Tokenize the complete prompt

Tokenize the full initial router prompt once and persist the returned IDs. Normally, set `add_special: true` only for this initial prompt.

```sh
curl -sS -X POST "$BASE/tokenize" \
  -H "Content-Type: application/json" \
  -d '{"content":"Your complete initial router prompt","add_special":true,"parse_special":true}'
```

Record each removable object's absolute `[start_pos, end_pos)` range. `start_pos` is inclusive and `end_pos` is exclusive.

#### 2. Prefill and claim the slot

Replace the illustrative IDs with the complete array returned by `/tokenize`:

```sh
curl -sS -X POST "$BASE/slots/0?action=managed_native_prefill" \
  -H "Content-Type: application/json" \
  -d '{"expected_revision":0,"tokens":[151644,8948,198,2610]}'
```

Persist the returned `managed_revision`. Every successful managed mutation increments it. `managed_native_completion` can bootstrap an empty slot when initial generation is also needed.

#### 3. Replace or delete a range

Delete with an empty replacement, or install a shorter marker or summary. For hybrid Qwen targets, use `experimental_attention_only: true`.

```sh
curl -sS -X POST "$BASE/slots/0?action=kv_edit" \
  -H "Content-Type: application/json" \
  -d '{
    "expected_revision":1,
    "experimental_attention_only":true,
    "compact_positions":false,
    "edits":[
      {"start_pos":1176,"end_pos":1243,"tokens":[]},
      {"start_pos":1650,"end_pos":1706,"tokens":[101,202,303]}
    ]
  }'
```

Edits must be ordered, non-overlapping, inside the cached range, and use valid target-model IDs. A replacement cannot be longer than the removed range. The retained suffix stays at its original positions and is not prefetched again. Empty positions become reusable attention-KV holes.

#### 4. Compact released holes

When `GET /props` advertises the DFlash2 dual-compaction capabilities and the slot reports coherent draft state, close released holes in a later request. The ranges submitted for compaction must contain only released `LLAMA_TOKEN_NULL` positions.

```sh
curl -sS -X POST "$BASE/slots/0?action=kv_edit" \
  -H "Content-Type: application/json" \
  -d '{
    "expected_revision":2,
    "experimental_attention_only":true,
    "compact_positions":true,
    "edits":[
      {"start_pos":1176,"end_pos":1243,"tokens":[]}
    ]
  }'
```

Compaction shifts the target and supported DFlash2 draft positions together without decoding the retained suffix. Do not compact an incoherent slot; rebuild it first.

#### 5. Append exact tokens

Use `kv_append` when the router needs to add tokens without generating:

```sh
curl -sS -X POST "$BASE/slots/0?action=kv_append" \
  -H "Content-Type: application/json" \
  -d '{"expected_revision":3,"tokens":[606,707,808]}'
```

The response includes `pos_start`, `pos_end`, `n_appended`, and `token_probe`.

#### 6. Continue generation

Use `managed_native_completion` after a managed edit or append. It uses the normal scheduler and evaluates only the new tail.

```sh
curl -sS -X POST "$BASE/slots/0?action=managed_native_completion" \
  -H "Content-Type: application/json" \
  -d '{
    "expected_revision":4,
    "tokens":[],
    "n_predict":128,
    "temperature":0.2,
    "stream":false
  }'
```

Set `stream: true` for `managed_native_begin`, `managed_native_delta`, and `managed_native_final` events followed by `data: [DONE]`. Send an empty token array with `continue_generation: true` to continue from the current logical tail.

For rolling generation, set `rolling_headroom` and `rolling_context_limit` on a streaming request. The server emits a managed pause before the configured headroom is exhausted; the router can compact, append, or resume the same task without replaying the prompt.

#### 7. Cancel a managed task

Use the task ID returned by the managed generation stream:

```sh
curl -sS -X POST "$BASE/slots/0?action=cancel" \
  -H "Content-Type: application/json" \
  -d '{"expected_task_id":12}'
```

Cancellation is task-scoped. The response confirms that cancellation was requested; consume the generation stream to observe its final state.

#### 8. Recover a slot

If a mutation reports `managed_requires_rebuild: true`, stop using the slot, erase it, and rebuild it from the router's authoritative token ledger:

```sh
curl -sS -X POST "$BASE/slots/0?action=erase"
```

Revision mismatches and validation failures do not mutate the slot. Do not use ordinary `/completion`, normal prompt-cache matching, or managed slot save/restore as a substitute for the router rebuild.

## Contract

This experimental fork gives a router explicit ownership of a `llama-server` slot. Text operations use exact target-model token IDs, half-open ledger ranges, and an optimistic concurrency revision. Image operations additionally carry acknowledged media spans and causal positions. The server can:

- prefill an empty managed slot without generation;
- delete an interior attention-KV range without decoding the retained suffix;
- replace a range with an equal or shorter marker or summary, decoding only the replacement;
- append exact tokens at the logical tail;
- continue generation through the normal llama.cpp scheduler; and
- continue with a supported DFlash companion cache.

This is not exact context rewriting. Qwen 3.6 and Qwen3.8-Flash-Next are hybrid attention/recurrent models. Attention cells are released, but retained recurrent state is restored after an interior replacement and can still contain influence from removed content. Use an authoritative full rebuild when exact deletion semantics matter.

The API remains version 1 and is intentionally separate from OpenAI-compatible completion routes. The current Spomin deployment uses Qwen3.8-27B and DFlash2 with coordinated target/draft editing and compaction. Its recorded live integration gate continued draft generation after closing 236 holes; see [source and validation](spomin-runtime-sync.md). MTP continuation after an interior edit is not supported. DSpark shares the implemented dual-edit path but needs its own model-specific validation.

Managed slots can carry images when the loaded runtime advertises the required vision capabilities. Media spans distinguish placeholder cells from released holes and track token-cell ranges separately from causal positions. See [managed vision](managed-vision.md); text edits cannot overlap an image span.

## Slot state machine

Every slot exposes:

| Field | Meaning |
| --- | --- |
| `managed_revision` | Monotonic concurrency guard |
| `managed_append_only` | The router owns the slot; ordinary completion is rejected |
| `managed_draft_coherent` | The current implementation permits supported speculative continuation after the managed mutation |
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
  "qwen_attention_only_dflash_dual_compact": true,
  "qwen_compact_positions": true
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

`start_pos` is inclusive and `end_pos` is exclusive. Edits must be ordered, non-overlapping, inside the current attention range, and use valid IDs. A replacement may not exceed the removed range. For Qwen, `experimental_attention_only` must be true.

Empty replacement arrays delete ranges without prefill. Non-empty replacement arrays decode only the supplied marker or summary at the old range start. Later cached positions remain unchanged and are not re-prefilled. Unused positions become logical holes and their attention cells are available for reuse.

After a router finishes a batch of non-compacting edits, it may close all resulting holes in one request by submitting those empty ranges again with `compact_positions: true`. Empty compacting ranges must already contain only released `LLAMA_TOKEN_NULL` positions in the managed prompt ledger. The server applies their cumulative position deltas to the retained target cache and compacts its position-aligned prompt ledger without decoding the retained suffix. The managed text-only path permits Qwen's M-RoPE cache because all rotary axes advance together for text, while the ordinary multimodal shift gate remains disabled. The coherent DFlash/DFlash2 companion cache receives the corresponding removals, replacement features, and cumulative position shifts. Successful dual compaction preserves draft continuation. Check the advertised dual-compaction capability and the acknowledged draft coherence; an already incoherent draft is not repaired by this operation. Other model-backed draft caches are invalidated. This remains KV surgery: retained values and recurrent representations are not recomputed.

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

DFlash and DFlash2 use the shared block-draft path. Explicit nonsequential processing injects replacement target features into the supported draft cache. Hole compaction shifts both caches, and the managed acknowledgement retains draft coherence only after successful coordinated operations. The runtime also carries DFlash2 tensor-split selector and position fixes. Unsupported target-only edits invalidate the draft; supported dual compaction does not.

## Validate the fork

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

### Current: Qwen3.8-27B with DFlash2

Spomin is running this model pair. The live integration gate integration-20260908-07 passed replacement, deletion, compaction of 236 holes, a 13-token append, and 64 generated tokens with 120 draft tokens proposed and coherent target/draft state reported. See [runtime synchronization](spomin-runtime-sync.md). The current long-workload test will add its results when complete.

### Historical: Qwen 3.6 with DFlash

On 2026-08-02, the hardened working tree passed the focused six-test CUDA server regression, the existing slot save/restore suite, a complete configured CMake build, and all 42 native CTest entries after fetching the repository's tiny model fixture. It also passed the following tests on two RTX 3090 GPUs:

| Configuration | Result |
| --- | --- |
| Qwen 3.6 27B Q8_0, 4096 context, Q8_0 target K/V | 2010 source tokens -> 1136 live entries after 994 removed and 120 inserted; 2960 tail tokens refilled all 4096 cells; logical tail ended at 4883 |
| Qwen 3.6 27B Q8_0 managed generation | 20-token range replaced by 7 summary tokens; only 11 new tail tokens evaluated; 32 tokens generated; rebuild state stayed false |
| Same target plus Qwen 3.6 27B DFlash Q8_0, F16 draft K/V | Same full capacity refill; target and draft remained coherent |
| DFlash managed generation | Only 11 new tail tokens evaluated; 64 generated at 56.36 tokens/s; 48 of 204 draft tokens accepted; draft coherence true and rebuild state false |

The target was 29,047,084,160 bytes with SHA-256 `88a2cd21a43cac266c74033f426d2929738960f139405758c11202336f916707`. DFlash was 1,849,481,312 bytes with SHA-256 `82545cd07d06cceaeb12feaa6ca76494214c872f83f71282bc6c009de61f30db`. These results establish implementation behavior, not semantic equivalence to a full rebuild.

The current Qwen3.8-27B/DFlash2 integration record is documented in [runtime synchronization](spomin-runtime-sync.md). The historical Qwen 3.6 numbers above are retained with their original model labels.

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

Do not enable MTP continuation, DSpark dual editing, or multimodal managed slots as incidental conflict resolutions. Each requires a separate model-specific design and validation effort. Preserve the DFlash-only dual-compaction gate when carrying Qwen position compaction forward.
