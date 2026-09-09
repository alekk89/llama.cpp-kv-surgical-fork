# Managed vision (multimodal managed slots)

Companion to `kv-surgery.md`. This is the contract for vision on the
router-owned managed slot: image spans in managed prefill/append, a
read-only geometry endpoint, and surgical image retirement.

2026-09-09 update: `vision_text_edit` and `mrope_compaction` advertise mixed-media
text surgery. `kv_edit` ranges index ledger cells, not causal positions. Text
ranges may occur before, between, or after images, but may not overlap any image
cell. Whole images use `media_retire`. Edits translate validated text boundaries
to causal positions, preserve retained media handles, and acknowledge all live
images in `media` with independent updated token and position ranges. Hole
compaction shifts image rotary and spatial-mask coordinates rigidly; it does
not re-encode images or prefill retained text. DFlash/DSpark applies the same
causal edits to its draft cache and remains enabled. Paused generations return `token_start` and `ledger_cells`
alongside causal `pos_start`, so rolling text edits use the same cell coordinates.
The original v1 restrictions below are historical where superseded by this update.

Status: design + implementation record. Validated on 2026-08-17 against
Qwen3.8-27B-Q8_0, build `b10453-3cb7ffb1a`, external `mmproj-F16.gguf`
(SHA-256 `cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e`),
262,144 context, Q8_0 target/draft KV, embedded MTP, and two RTX 3090s.

The live gate covers geometry, managed media prefill, MTP generation, media
retirement, a second media append, target-only generation after retirement,
invalid-tombstone rejection without revision drift, and the full Spomin API
two-turn retire/replace path. The wider multi-image, rebuild-comparison,
failure-injection, and restore-expiry canaries remain separate acceptance work.

## Why this exists

The text-only managed slot deliberately rejects `mmproj` servers so that
`LLAMA_TOKEN_NULL` position holes could not be confused with multimodal
placeholders. The confusion is resolved structurally: a token cell is a media
cell if and only if `map_idx_to_media` has an entry at its start index (see
`server_tokens`). With that disambiguation the managed slot can own images.

The router renders the model's chat template itself (it already does for text
turns) and forwards the rendered prompt plus the ordered image bytes. The fork
resolves geometry, fills visual KV from features, and acknowledges the exact
token and position ranges. The router never manufactures M-RoPE coordinates
and never derives position extents from token counts (they are independent
quantities, acknowledged independently).

## Capability discovery

`GET /props` -> `managed_slot` gains `supports_mmproj`, and the top-level
props gain `media_marker`:

```json
{
  "managed_slot": { "...": "...", "supports_mmproj": true },
  "media_marker": "<__media_xxx__>"
}
```

- `supports_mmproj` is true only when the server runs with an mmproj context
  for the loaded model. Text-only servers report `false` exactly as today;
  managed text operations are unchanged on such servers.
- `media_marker` is the prompt marker string (random per server start, or
  pinned via `LLAMA_MEDIA_MARKER`). The router does not need to know it: it
  renders the model template, which emits the template's own image token at
  each image occurrence. The marker only matters when a client forwards
  prompts directly.

## Forwarding contract (router obligations)

The router renders the chat template with per-image boundary labels
(`add_vision_id`, i.e. `Picture N: <image>` in the Qwen templates) so that
each image occurrence in the rendered prompt is uniquely labelled and
ordered:

1. **Order preservation.** The `multimodal_data` array order must equal the
   order of the image occurrences in the rendered prompt.
2. **No adjacent unlabelled images.** Two image parts with no non-empty text
   part between them are rejected at ingress. The boundary label is
   mandatory; adjacency without intervening text makes the association
   ambiguous and is never forwarded.
3. **Format normalization.** Incoming image bytes are normalized to
   8-bit RGB JPEG or PNG before forwarding (WebP/GIF/16-bit etc. are
   re-encoded deterministically at ingress). The normalized bytes are what
   the router stores as the blob, what it hashes, and what it forwards;
   geometry and KV depend only on the normalized bytes.
4. **Exact bytes.** The forwarded bytes must match the stored blob byte for
   byte. The fork computes `sha256` for each accepted image and the router
   fails closed on mismatch.

## Managed prefill / append with images

`managed_native_prefill` and `kv_append` accept an image-carrying form
instead of `tokens` (never both):

```json
POST /slots/0?action=managed_native_prefill
{
  "expected_revision": 0,
  "prompt": "<rendered template text with image occurrences in order>",
  "multimodal_data": ["<base64 image 1>", "<base64 image 2>"]
}
```

- `prompt` is the fully rendered chat template (system + history + turn).
  Image occurrences are the template's image tokens; each consumes the next
  entry of `multimodal_data` in order.
- A non-empty `multimodal_data` requires a non-empty `prompt`; an empty
  `multimodal_data` is equivalent to sending `tokens` (text-only path).
- Validation before any mutation: slot ownership/revision as today; every
  image decodes; the number of image occurrences in the prompt equals
  `len(multimodal_data)`; every resolved geometry is finite.
- The fork tokenizes via the standard mtmd path
  (`process_mtmd_prompt`), fills visual KV from features, decodes, and
  commits the slot ledger (`server_tokens` with media cells as
  `LLAMA_TOKEN_NULL` plus `map_idx_to_media`).

### Acknowledgement

Prefill and append acknowledgements gain, alongside `managed_revision`,
`n_appended`, `pos_start`, `pos_end`:

```json
{
  "n_appended": 3569,
  "pos_start": 0,
  "pos_end": 3507,
  "media": [
    {
      "index": 0,
      "media_chunk_id": "fork-media:3",
      "token_start": 912,
      "token_end": 3564,
      "position_start": 912,
      "position_end": 980,
      "kv_cells": 2652,
      "n_positions": 68,
      "grid": { "t": 1, "h": 39, "w": 68 },
      "sha256": "..."
    }
  ],
  "tokens": [ 151644, 8948, null, null, "..." ]
}
```

- `token_start`/`token_end` are half-open stream token indices (the
  bookkeeping run; one `null` cell per KV cell).
- `position_start`/`position_end` are half-open causal positions. They are
  acknowledged, never derived from `kv_cells`.
- `media_chunk_id` is fork-allocated and stable for the chunk's life; it is
  the handle used by retirement.
- `tokens` echoes the appended stream with `null` at media cells so the
  router's durable ledger matches the fork byte-for-byte.
- Text-only requests keep today's acknowledgement shape (no `media`, no
  `tokens` echo) for compatibility.

## Geometry endpoint (read-only, no revision)

```json
POST /slots/{id}?action=image_geometry
{ "data_b64": "..." }
->
{ "sha256": "...", "width": 1920, "height": 1080,
  "kv_cells": 2652, "n_positions": 68,
  "grid": { "t": 1, "h": 39, "w": 68 },
  "position_kind": "mrope_2d" }
```

Deterministic for (model, normalized bytes). No slot state, no revision, no
KV. Implemented with the placeholder bitmap path
(`process_mtmd_prompt(..., is_placeholder = true)`). The router uses
`kv_cells` for admission and placeholder-run sizing and `n_positions` for
causal-position accounting.

## Surgical image retirement

```json
POST /slots/0?action=media_retire
{
  "expected_revision": 5,
  "media": [
    { "media_chunk_id": "fork-media:3",
      "token_start": 912, "token_end": 3564,
      "position_start": 912, "position_end": 980 }
  ],
  "tombstone": [ 101, 202, 303 ]
}
```

- `media` lists every chunk to retire by the acknowledged handle and
  expected ranges. All requested chunks are validated against the slot
  ledger before the first mutation; any mismatch (unknown handle, range
  drift, already retired) rejects the whole request with no mutation.
- The server removes each chunk's attention KV cells and media-ledger entry.
- `tombstone` (optional token IDs) is installed at the released causal
  interval when it fits (`len(tombstone) <=` released positions). When it
  does not fit the response reports `"tombstone_fits": false` and installs
  nothing; the router then installs the tombstone at the ordinary
  addressed-summary position as a text append.
- Position compaction: retired causal intervals are closed with the same
  cumulative text-only shift the text `kv_edit` compaction uses. All
  requested chunks are retired in one operation; the shift is applied once,
  after every requested chunk is gone.
- Retained recurrent/SSM state is preserved (the shift reuses the
  text-only RoPE path); this is the same approximation boundary as text
  surgery, and a full rebuild remains the exact-deletion escape hatch.
- DFlash/DSpark: remove the same causal intervals from the draft, shift its
  retained rows by the same causal deltas, and inject only the replacement
  features. Image patch counts differ from draft row counts; both caches
  share causal positions. A coherent draft stays coherent and speculative
  generation remains enabled. Other model-backed draft types retain the
  target-only fallback (`managed_draft_coherent: false`).

### Acknowledgement

```json
{
  "managed_revision": 6,
  "removed_kv_cells": 2652,
  "removed_positions": 68,
  "shifted_text_cells": 1234,
  "positions_compacted": true,
  "tombstone_fits": true,
  "tombstone_token_start": 912,
  "tombstone_position_start": 912,
  "media": [
    { "media_chunk_id": "fork-media:3", "retired": true },
    { "media_chunk_id": "fork-media:7",
      "token_start": 4001, "token_end": 6600,
      "position_start": 3870, "position_end": 3938 }
  ]
}
```

- `media` reports every live chunk after the operation with its **updated**
  ranges (surviving chunks shift), plus one `retired: true` entry per
  requested chunk. The router rewrites its span records from this list; it
  does not recompute coordinates itself.

## Failure and rollback

Validation failures never mutate and never bump the revision. If a mutation
has started and then fails, the existing managed-slot rule applies:
revision increments, `managed_requires_rebuild: true`, draft coherence is
cleared, and all later managed operations are rejected until `erase`.
KV and metadata therefore fail together: a partially mutated slot cannot
continue. The router marks the session desynchronized. When uninterrupted
KV is required, it rejects continuation instead of replaying the ledger.

## v1 implementation notes

- All managed endpoints use the shared action form
  `POST /slots/{id_slot}?action=<name>`; the vision actions are
  `kv_append`, `managed_native_prefill`, `image_geometry`, and
  `media_retire`.
- The `media`/`tokens` acknowledgement fields are emitted only for
  image-carrying requests; text-only prefill/append keep today's shape.
- `supports_mmproj` reports `false` while the server is sleeping (the
  endpoint must stay readable during sleep; go through the queue state).
- v1 tombstone rule: one optional `tombstone` list per operation, installed
  at the earliest released interval only when it fits
  (`tombstone_fits`); nothing is installed when it does not. There are no
  per-chunk tombstones in v1; the router installs any uninstalled tombstone
  as an ordinary text append.
- Text `kv_edit` supports media-containing ledgers using the dual-coordinate
  contract above. Any overlap with an image is rejected before mutation;
  text holes are distinct from media placeholders. `media_retire` may leave
  other images resident and acknowledges their shifted coordinates.
- Mid-`media_retire` failure follows the managed-slot rule: revision bumps,
  `managed_requires_rebuild` is set, draft coherence is cleared; KV and
  metadata fail together, never partially.
- `managed_native_completion` uses DFlash after text edits around images,
  selected-image retirement, and rolling edits while generation is paused.
  None of these operations replay retained text or images. The legacy
  `managed_generate` compatibility path remains target-only.

## Out of scope (this revision)

- Video chunks (image only).
- Restoring a retired chunk without re-encoding (feature-cache
  optimization; the restore path re-forwards the stored bytes, so
  `kv_append` with `multimodal_data` at the tail is the restore primitive).
