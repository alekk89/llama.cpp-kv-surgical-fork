# KV surgery fork guide

## Purpose and status

This experimental fork adds a token-only managed-slot interface to
`llama-server`. A router can remove old attention-KV ranges, decode a shorter
summary or marker at the old range start, and append later conversation tokens
without re-prefilling the retained suffix.

The tested target is Qwen 3.6 27B with MTP or a DFlash sidecar. DSpark uses the
same position-local companion-cache injection path as DFlash, but managed
surgery with a DSpark checkpoint has not yet been model-validated. The target is
a hybrid attention/recurrent model, so this path has a strict approximation
contract:

- attention-KV cells for removed tokens are released and can be reused by later
  tail tokens;
- suffix positions are not compacted and logical gaps remain;
- Qwen recurrent state is restored after a middle replacement and is therefore
  stale; deleted material can still influence output;
- normal OpenAI-compatible completion and prompt-cache matching are not a safe
  continuation after an edit;
- the router must own the slot, exact tokenizer IDs, segment ranges, and every
  later append.

The 12K capacity test passed: 9,980 prefetched tokens became 5,301 live
attention entries after router removal markers, then accepted 6,987 new tail
tokens to refill a 12,288-cell cache with no suffix re-decode. The final logical
tail position was 16,780.

Use proxy/rebuild for exact semantics and recovery. This fork is for measuring
whether the capacity saving is useful despite the approximation.

Managed native completion keeps a position-aligned token ledger after
non-compacting edits. Removed ranges remain as `LLAMA_TOKEN_NULL` holes so a
new tail receives monotonically increasing positions. Those holes are not live
KV cells: managed request admission, generation limits, context-shift checks,
and speculative headroom use the non-null token count. The ordinary
high-water `server_tokens::size()` remains valid only as a positional ledger.

## Build

Configure and build only the server target. CUDA is optional; use the backend
appropriate for the machine.

```powershell
cmake -S . -B build -DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build --config Release --target llama-server -j 8
```

On a Windows multi-config build the executable is normally
`build\bin\Release\llama-server.exe`. Start it with a writable slot-save
directory; the managed slot actions are unavailable without one.

```powershell
New-Item -ItemType Directory -Force .\tmp\slots | Out-Null
.\build\bin\Release\llama-server.exe `
  -m C:\models\Qwen3.6-27B-Q8_0.gguf `
  --ctx-size 12288 --parallel 1 --slot-save-path .\tmp\slots -ngl 999
```

## Managed-slot lifecycle

1. Tokenize every router memory object with `POST /tokenize` and persist the
   exact IDs and original ranges.
2. Prefill the initial router prompt with `POST /completion`, a fixed `id_slot`,
   `cache_prompt: true`, and `n_predict: 0`.
3. Send all queued replacements in one `kv_edit` request.
4. Append a router-owned diagnostic tail with `kv_append`, use the explicit
   compatibility `managed_generate` action, or use
   `managed_native_completion` for a revision-checked routed turn. The latter
   runs llama.cpp's normal completion scheduler from a virtual prompt ledger
   that longest-prefix-matches the physical edited slot, so it evaluates only
   the new tail. Do not send an ordinary normal completion request against the
   edited slot.

For Qwen, use the explicit experimental mode and leave compaction disabled:

```json
POST /slots/0?action=kv_edit
{
  "experimental_attention_only": true,
  "compact_positions": false,
  "edits": [
    {
      "start_pos": 1176,
      "end_pos": 1243,
      "tokens": [101, 202, 303]
    },
    {
      "start_pos": 1650,
      "end_pos": 1706,
      "tokens": [404, 505]
    }
  ]
}
```

`start_pos` is inclusive and `end_pos` is exclusive. Edits must be ordered,
non-overlapping, within the cached attention range, and use valid token IDs.
The server reports `n_removed`, `n_inserted`, positions before and after, and
whether the request used attention-only mode.

Append router-owned tokens and use `token_probe` as the greedy next-token probe:

```json
POST /slots/0?action=kv_append
{
  "tokens": [606, 707, 808]
}
```

The response includes `n_appended`, `pos_start`, `pos_end`, and `token_probe`.

For a complete post-edit turn, use the experimental managed generation action:

```json
POST /slots/0?action=managed_generate
{
  "expected_revision": 4,
  "tokens": [606, 707, 808],
  "n_predict": 128,
  "temperature": 0.2,
  "stream": false
}
```

### Native scheduler continuation

`managed_native_completion` is the preferred post-edit router endpoint. It
accepts the same `tokens`, `n_predict`, sampling fields, `expected_revision`,
and `stream` flag as the generation path, but it requires a non-empty tail and
an already router-owned managed slot. Its SSE records are named
`managed_native_delta` and `managed_native_final`; the final record includes
`n_appended`, `managed_revision`, and `managed_append_only`. The endpoint is
not an ordinary completion bypass: it permits only the exact router-owned
tail, never an arbitrary replacement prompt. Speculative/MTP decoding is
enabled only while the fork can prove that target and companion state are
coherent. Qwen attention-only surgery remains target-only for MTP. For DFlash
and DSpark, the same edit removes the companion positions and injects
replacement K/V from the target features at their original positions. The edit
acknowledgement reports `managed_draft_coherent=true` only after both mutations
succeed.

It returns the exact appended and generated token counts, generated token IDs,
decoded text, absolute start/end positions, and the new managed revision. It
remains experimental for Qwen: it does not make interior surgery semantically
exact.

### DFlash dual-edit validation

On 2026-07-27, Qwen 3.6 27B Q8_0 plus its DFlash Q8_0 sidecar completed two
attention-only managed edits without rebuilding the retained prefix. Both edit
acknowledgements returned `managed_draft_coherent=true`. DFlash remained active
after each edit: the first continuation accepted 84 of 130 draft tokens and the
second accepted 53 of 110. The bootstrap acceptance was 80 of 159. No dual-edit
or speculative-processing failure was logged.

The recurrent target warnings for an interior replacement remain expected. The
target recurrent snapshot is restored after the replacement decode; those
warnings do not imply loss of DFlash coherence.

DSpark is admitted by the same dual-edit gate because its draft context reuses
the DFlash encoder, decoder, target-feature injection, and companion KV layout.
If either companion removal or replacement injection fails, the slot does not
claim draft coherence and speculative continuation remains disabled.

## Router markers and decompression

The tested router convention is:

```text
[summary: 2_seg0031, archived]: <short summary>
[section removed: 2_seg0043]
```

The router system prompt tells the model to request source-only details with
`decompress:<id>`. In the Qwen test, an archived-summary question produced
`decompress:2_seg0031`, while a removed-section question recognized a known
content gap. The router must decide whether to rebuild or use another explicit
managed operation when it grants decompression; it must not silently resume a
normal prompt-cache completion after an edit.

## Reproducible capacity test

Start a fresh server with `--ctx-size 12288`, then run:

```powershell
python tools\server\tests\kv_surgery_capacity.py --server-url http://127.0.0.1:8080
```

The script constructs 50 router objects, removes 25 with trace markers, and
uses `kv_append` to fill the released capacity with new tail tokens. It exits
nonzero if the expected cache capacity is not reached. It is an integration
harness requiring a Qwen server, not a normal CI unit test.

## Carrying the fork forward

Keep all fork changes small and explicit. The current upstream touch points are:

| Area | Files |
| --- | --- |
| Public API and batch validation | `include/llama.h`, `src/llama-batch.cpp`, `src/llama-context.cpp` |
| Hybrid SWA capability fix | `src/llama-kv-cache-iswa.cpp` |
| DFlash and DSpark companion injection | `common/speculative.cpp` |
| Managed token ledger | `tools/server/server-common.h`, `tools/server/server-common.cpp` |
| Server task and routes | `tools/server/server-task.h`, `tools/server/server-task.cpp`, `tools/server/server-context.h`, `tools/server/server-context.cpp` |
| Basic server coverage | `tools/server/tests/unit/test_slot_kv_edit.py` |

For every upstream update:

1. Keep `upstream` pointing to `https://github.com/ggml-org/llama.cpp.git`.
2. Rebase or merge the desired upstream revision.
3. Resolve only the files in the table above, preserving upstream task, slot,
   memory, and batch invariants rather than copying old surrounding code.
4. Build `llama-server`, run `test_slot_kv_edit.py`, then run the Qwen capacity
   harness with the pinned model and server settings.
5. Record the upstream commit, model GGUF hash, context size, KV cache types,
   backend, and capacity-test JSON in the router experiment log.

Do not enable position compaction for Qwen. Treat any future RoPE-compaction
work as a separate model-specific feature with its own full-rebuild comparison.
