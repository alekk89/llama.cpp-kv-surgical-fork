# llama.cpp KV surgery fork

A focused `llama.cpp` runtime fork for the [Spomin router](https://github.com/alekk89/Spomin). It adds managed KV-cache operations that let a router edit, compact, and continue a live `llama-server` slot without rebuilding the retained suffix.

> This is experimental runtime software. The current deployed target is Qwen3.8-27B with DFlash2. The live integration passed replacement, deletion, position compaction, append, and coherent target/draft generation, but this is not a general-purpose replacement for upstream `llama.cpp`.

## What it does

- Exposes revision-checked managed-slot endpoints for exact token prefill, range replacement, deletion, append, and native continuation.
- Reclaims released attention-KV cells and can compact holes while keeping the target and supported DFlash2 draft caches aligned.
- Supports rolling managed generation, task-scoped cancellation, and managed text/image span tracking.
- Preserves the router's absolute token ledger so the retained suffix is not prefetched after an edit.

This repository contains the runtime only. Spomin owns durable history, summaries, context admission, session identity, and the router-side token ledger. The [llama.cpp Windows Manager](https://github.com/alekk89/llama-cpp-windows-manager) can install and run custom `llama.cpp` builds on Windows, but it is not required.

## Current support

- Qwen3.8-27B with DFlash2: current live integration target.
- Qwen 3.6 27B with DFlash: historical validation retained in the technical guide.
- DSpark: implementation is present, but separate model-specific validation is still required.
- MTP: not supported after an interior KV edit.

Interior edits on hybrid models preserve recurrent state instead of recomputing it, so surgery remains an approximation of a fresh full-context evaluation. Keep the server on loopback and use an authoritative router rebuild when exact deletion semantics matter.

## Quick start

See the [setup and usage guide](docs/kv-surgery.md#setup) for model requirements, build commands, server launch examples, API requests, recovery, and tests.

## Documentation

- [Setup and usage](docs/kv-surgery.md#setup)
- [Managed-slot workflow](docs/kv-surgery.md#use)
- [Technical contract](docs/kv-surgery.md#contract)
- [Managed vision](docs/managed-vision.md)
- [Spomin runtime synchronization and validation](docs/spomin-runtime-sync.md)
- [Upstream speculative decoding reference](docs/speculative.md)

## Development

The maintained branch is `experimental/kv-surgery-dflash`. Read [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md) before changing the fork. Use the [upstream-update checklist](docs/kv-surgery.md#carrying-the-fork-forward) when syncing newer `llama.cpp` changes.

## License

This fork follows the upstream `llama.cpp` license. Bundled dependencies retain their own licenses.
