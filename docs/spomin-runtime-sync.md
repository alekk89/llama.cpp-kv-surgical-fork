# Spomin runtime synchronization

## Source

The September 2026 synchronization brings the deployed Spomin runtime changes
into the public experimental/kv-surgery-dflash branch. The source is the
working DFlash2 integration based on art-den/llama.cpp commit
3286b033c5dc8bcce9958f724d4e4a93ae589646 (fix/dflash2-tensor-split), including
the local managed-slot, mixed-media, rolling-generation, and cache fixes.
The base commit alone does not contain those local changes.

This update preserves the public fork's newer upstream context allocation
and state-copy implementation. It imports the required DFlash2 tensor-split
selector and block-draft allocation changes together with the managed runtime.
Model weights, runtime binaries, private profiles, logs, and Spomin training
data are not part of this source synchronization.

## Current behavior

- Exact token ranges are replaced under an expected managed revision.
- Supported DFlash/DFlash2 edits update the target and draft caches together.
- Compaction closes released holes and shifts both caches' positions.
- A coherent supported draft remains coherent after successful compaction.
- Replacement decodes only replacement tokens; the retained suffix is not
  prefetched again. Recurrent tensors are preserved, so surgery remains an
  approximation to a fresh full-context evaluation.
- Managed native generation supports revisioned rolling pause/resume and
  task-scoped cancellation.
- Managed image spans track token cells and causal positions independently;
  the current implementation supports text edits around retained images.

Before enabling draft-preserving compaction, check GET /props for
qwen_compact_positions, qwen_attention_only_dflash_dual_edit, and
qwen_attention_only_dflash_dual_compact. Verify managed_draft_coherent and
managed_requires_rebuild in the mutation and generation acknowledgements.
Capability discovery does not itself prove that a particular model pair has
passed a real-model integration run.

## Existing deployment evidence

The Spomin integration run integration-20260908-07 used Qwen3.8-27B Q8_0
and DFlash2. It prefetched 2,048 tokens, replaced a region, deleted another,
compacted 236 holes, appended 13 tokens, and generated 64 tokens with 120 draft
tokens proposed and coherent target/draft state reported. It also checked
task-scoped cancellation and a real summary-worker installation.

These are short integration results from the deployed source. They establish
the working draft-preserving lifecycle; they do not establish long-session
quality or benchmark performance for every model. The current long-workload
test results belong in Spomin's results section when complete.

## Verification of this publication

Verification on Windows with Visual Studio 2022, Release, CPU backend:

- llama-server and the selected native test targets built successfully.
- Existing managed-slot regression: 7 passed, using the cached stories260K
  fixture on an isolated port. The capability expectation includes rolling
  generation and mixed-media text-edit discovery.
- Native CPU ROPE operation tests: 466/466 passed.
- Source diff whitespace checks passed.

The focused pytest runner selected only the local tiny model, avoiding the
suite-wide preload of unrelated model presets. All seven test bodies and the
normal server cleanup fixture ran. The final source matches the deployed
runtime in the synchronized implementation files, while retaining the public
fork's newer context allocation/state-copy changes.

These checks did not reload or mutate the running Qwen slot. A fresh CUDA
Qwen3.8/DFlash2 integration run against this publication is not claimed;
the real-model evidence above belongs to the deployed source.

## Harness requirements

The harness talks to Spomin's OpenAI-compatible endpoint; Spomin talks to this
fork's managed-slot endpoints. Preserve one session ID and the exact returned
assistant/tool objects. Disable both automatic and manual harness compaction.
Spomin owns the token ledger, chunk IDs, summaries, slot revisions, and cache
maintenance. Ordinary completion requests must not reuse an edited managed slot.
