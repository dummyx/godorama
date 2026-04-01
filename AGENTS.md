# AGENTS.md

## Mission

Build and maintain a production-grade Godot 4 GDExtension that embeds `llama.cpp`
directly through `libllama`. The extension must be deterministic to build,
safe to upgrade, non-blocking at runtime, and straightforward to package for
desktop Godot projects.

This repository is product code, not a playground. Favor correctness,
predictable behavior, explicit ownership, and stable public interfaces.

## Hard constraints

1. Production code must integrate `llama.cpp` as a library through `include/llama.h`.
   Do not shell out to `llama-cli`, `llama-server`, Python wrappers, or subprocesses.
2. Never block the Godot main thread for model loading, prompt evaluation,
   token generation, embeddings, or large disk I/O.
3. Never expose raw `llama_*` handles, STL containers, or non-Variant-compatible
   types through the Godot API.
4. Keep `godot-cpp` and `llama.cpp` pinned to exact revisions.
5. CPU-only support is mandatory. GPU backends are optional and feature-gated.
6. No hidden downloads, no hidden telemetry, no hidden network access.

## Primary toolchain

- C++ baseline: C++20 for repository code.
- Primary build system: CMake + Ninja.
- Source of truth for builds: `CMakePresets.json`.
- Required developer outputs:
  - `compile_commands.json`
  - unit tests
  - integration tests
  - formatting target
  - lint target
- Preferred local cache tools when available:
  - `ccache` on Linux/macOS
  - `sccache` on Windows

Do not raise the language baseline to C++23 for the whole repo unless CI proves
that every supported compiler/toolchain combination is green.

## Repository layout

Use this structure unless there is a compelling reason to change it:

- `cmake/`                      shared CMake modules and toolchains
- `extern/godot-cpp/`           pinned submodule
- `extern/llama.cpp/`           pinned submodule
- `include/godot_llama/`        public C++ headers for this project
- `src/llama/`                  thin adapters over `libllama`; no Godot headers
- `src/core/`                   orchestration, threading, error model, UTF-8
- `src/godot/`                  GDExtension binding layer only
- `tests/unit/`                 pure C++ tests
- `tests/integration/`          Godot/headless integration tests
- `demo/`                       minimal Godot demo project
- `docs/`                       architecture, build, API, packaging docs

## Layering rules

### `src/llama/`
This layer is responsible for:
- wrapping `llama_model`, `llama_context`, samplers, and request state
- translating internal config types to `llama.cpp` params
- shielding the rest of the repo from upstream API churn

Rules:
- no Godot headers
- no Godot types
- no editor/UI logic
- no global mutable singleton state

### `src/core/`
This layer is responsible for:
- async request lifecycle
- cancellation
- UTF-8 conversion helpers
- structured errors
- deterministic config handling
- threading and synchronization
- stable internal abstractions used by the Godot layer

Rules:
- still no Godot binding logic
- no direct scene tree interaction
- no Godot object lifetime ownership

### `src/godot/`
This layer is responsible for:
- GDExtension class registration
- converting Godot types to/from internal types
- main-thread signal emission
- editor-facing usability
- documentation-facing property and method surfaces

Rules:
- keep it thin
- do not place inference business logic here
- do not replicate `llama.cpp` parameter structures verbatim

## Dependency policy

- Pin `extern/godot-cpp` to the exact stable version or exact API target for the
  Godot version this repo supports.
- Pin `extern/llama.cpp` to an exact commit SHA or release tag.
- Never track floating branches in CI or release builds.
- Update dependencies one at a time.
- Every dependency bump must include:
  - the old and new revision
  - a short rationale
  - compatibility notes
  - smoke test results

Do not “just update submodules” as part of unrelated work.

## Public Godot API rules

The Godot-facing API must be:
- small
- async-first
- Variant-compatible
- explicit about ownership and threading

Prefer exposing only Godot-friendly types such as:
- `String`
- `PackedByteArray`
- `PackedInt32Array`
- `PackedFloat32Array`
- `Array`
- `Dictionary`
- `Callable`
- `RefCounted`
- `Resource`
- `Node` only where scene-tree semantics are genuinely needed

Rules:
- Bound methods and signals must use snake_case naming.
- Godot classes must use PascalCase naming.
- Every bound method must document:
  - whether it blocks
  - whether it is thread-safe
  - ownership semantics
  - units/defaults for parameters
- Avoid enormous option dictionaries unless there is no better Godot-friendly type.
- Prefer a small config `Resource` for stable settings over a giant list of freeform arguments.

## C++ design rules

Prefer:
- RAII everywhere
- move-only wrappers for native handles
- `std::unique_ptr` by default
- `std::shared_ptr` only for clearly shared long-lived objects such as a loaded model
- `std::span`
- `std::string_view`
- `std::optional`
- `std::variant`
- `std::array`
- `std::chrono`
- `std::filesystem`
- `std::jthread` and `std::stop_token`
- `enum class`
- `constexpr`
- `constinit`
- `[[nodiscard]]`
- `noexcept` where correct
- concepts only when they genuinely improve diagnostics and readability

Avoid:
- raw owning pointers
- `new` / `delete` for ordinary ownership
- `using namespace`
- giant template metaprogramming
- macro-heavy logic
- exceptions crossing library boundaries
- implicit conversions
- hidden allocations in hot loops

If you want `std::expected`, first prove that the entire supported compiler matrix
can use it cleanly. Otherwise use a small internal `Result<T>` / `Expected<T, E>`
type owned by this repo.

## Ownership and lifetime

- `llama_model` ownership must be explicit and centralized.
- `llama_context` ownership must be explicit and separate from model ownership.
- Sampler/request objects must be short-lived and isolated.
- Public wrappers around native resources should usually be move-only.
- Shared ownership must be deliberate, rare, and documented.
- Never let Godot object lifetime implicitly own third-party native handles unless
  that coupling is intentional and tested.

## Threading rules

- All inference work happens off the main thread.
- All Godot object mutation and signal emission happen on the main thread.
- Use cooperative cancellation.
- Clearly separate request submission, worker execution, and result delivery.
- Do not use detached threads.
- Do not invent ad-hoc thread pools without a strong reason.
- Prefer `std::jthread` or a small well-defined worker abstraction with shutdown semantics.

A synchronous API may exist internally for tests or low-level wrappers, but do not
bind blocking editor-facing methods by default.

## Error handling

- Code must compile with exceptions disabled.
- Never let exceptions cross the Godot boundary or the `llama.cpp` C API boundary.
- Use structured errors with fields such as:
  - code
  - message
  - context
  - optional backend/platform detail
- Convert internal errors to Godot-facing `Error`, `Dictionary`, or failure signals
  only at the binding layer.
- Do not swallow upstream error states.
- Error messages should be actionable and specific.

## Performance rules

- Convert `String` <-> UTF-8 in a centralized place.
- Avoid repeated string conversions inside token loops.
- Reuse prompt/token/output buffers when practical.
- Separate model lifetime from session/request lifetime.
- Benchmark hot paths before and after changing them.
- Do not optimize speculative code paths without measurements.
- Keep performance-sensitive logic out of the Godot glue layer.

## Godot-specific data handling

- For large `Packed*Array` transfers, prefer bulk pointer access helpers over
  per-element access.
- Keep Godot type conversion localized and testable.
- Do not leak STL types or internal structs into bound methods.
- Do not assume `Packed*Array` mutation will alias caller state unless explicitly designed.

## Build rules

- No in-source builds.
- Release and debug presets must both work from a clean checkout.
- Sanitizer presets must be available on supported platforms.
- Warnings are errors in CI.
- Keep third-party include paths isolated.
- Prefer static linkage for vendored native libraries unless a backend/platform
  constraint requires shared libraries.

Expected canonical commands:

```sh
cmake --preset dev
cmake --build --preset build-dev
ctest --preset test-dev --output-on-failure
cmake --build --preset build-dev --target format
cmake --build --preset build-dev --target lint
````

If these commands do not exist yet, create them.

## CI rules

Minimum CI matrix:

* Ubuntu
* Windows
* macOS

Minimum CI work:

* configure
* build
* unit tests
* at least one integration/smoke test job
* formatting/lint
* packaged artifact generation for release workflows

Recommended:

* ASan + UBSan on at least one POSIX job
* TSan on a dedicated job when concurrency code changes
* release artifact naming that matches the `.gdextension` manifest

## Testing rules

### Unit tests

Must cover:

* parameter translation
* UTF-8 conversion helpers
* request lifecycle
* cancellation
* error propagation
* edge cases around empty prompts / stop conditions
* deterministic behavior in seeded paths where applicable

Rules:

* no network
* no Godot dependency unless absolutely required
* keep them fast

### Integration tests

Must cover:

* extension load
* class registration
* one minimal open -> generate -> complete flow
* one failure path
* one cancellation path if implemented

Rules:

* headless where possible
* deterministic
* bounded runtime
* use the smallest legally redistributable test model available, or mock the
  integration seam if a real model is not appropriate for CI

## Packaging rules

* Keep the `.gdextension` file in sync with produced library names and paths.
* Do not rename binaries casually.
* Package only what is necessary.
* Document platform/backend limitations in release notes and docs.
* Ensure CPU-only packaging always works.
* GPU-specific builds must be clearly labeled.

## Documentation rules

Keep these files current:

* `README.md`
* `docs/ARCHITECTURE.md`
* `docs/BUILD.md`
* `docs/API.md`
* `CHANGELOG.md`

Every architecture-affecting change should update documentation in the same change.

## Security and privacy rules

* No hidden model downloads
* No telemetry
* No network calls in core runtime
* No unvalidated filesystem writes outside explicitly configured locations
* Be explicit about reading model paths and cache paths

## Forbidden shortcuts

Do not:

* wrap CLI tools with `system()`, `popen()`, or subprocess execution
* expose raw `llama.cpp` structs to Godot users
* put long-running work in `_process()` or main-thread callbacks
* introduce silent fallback behavior that hides real errors
* sneak dependency upgrades into unrelated work
* add “temporary” global state without teardown semantics
* land TODO-only scaffolding in critical code paths

## Definition of done

Before marking work complete, verify:

* [ ] clean configure/build works with the documented preset
* [ ] unit tests pass
* [ ] integration tests pass
* [ ] no main-thread blocking was introduced
* [ ] public API remains Variant-compatible
* [ ] errors are actionable
* [ ] docs were updated
* [ ] `.gdextension` file matches built binaries
* [ ] dependency pins are unchanged or intentionally updated
* [ ] new concurrency paths have shutdown and cancellation behavior

## How agents should operate

* Make the smallest coherent change that moves the repo forward.
* Prefer real, buildable code over pseudocode.
* Keep architectural boundaries intact.
* If upstream `llama.cpp` changed, inspect the pinned `include/llama.h` first.
* If Godot behavior changed, inspect the pinned `godot-cpp` version and manifest first.
* Explain tradeoffs when choosing `Node` vs `RefCounted`, sync vs async, shared vs unique ownership, or static vs shared linking.
* Do not optimize for cleverness. Optimize for maintainability, debuggability, and predictable shipping behavior.

````

Use this as the bootstrap prompt for the coding agent:

```text
You are a senior C++ systems engineer building a production-quality Godot 4 GDExtension
library that embeds llama.cpp directly as a native dependency.

Follow AGENTS.md exactly. If AGENTS.md and this prompt conflict, prefer AGENTS.md.

Project objective
Build a local-inference Godot extension around llama.cpp with a thin Godot-facing API,
modern C++20 internals, a modern CMake/Ninja toolchain, reproducible dependency pinning,
tests, CI, docs, and a minimal demo project.

Non-goals
- Do not shell out to llama-cli, llama-server, Python, or subprocesses.
- Do not require a web service.
- Do not auto-download models.
- Do not block the Godot editor/main thread during model load or generation.

Mandatory technical choices
- Primary toolchain: CMake 3.28+ + Ninja + CMakePresets.
- Language baseline: C++20 for repository code.
- Use pinned git submodules:
  - extern/godot-cpp
  - extern/llama.cpp
- Bind llama.cpp through `include/llama.h`.
- Keep strict layering:
  - src/llama    -> upstream adapter layer, no Godot headers
  - src/core     -> threading, lifecycle, config, UTF-8, errors
  - src/godot    -> GDExtension bindings only
- Public Godot API must remain Variant-compatible.
- Internal code may use STL and modern C++20 facilities freely, but never leak STL
  containers or raw third-party pointers across the Godot boundary.
- Code must compile with exceptions disabled. Never let exceptions cross library boundaries.
- CPU-only support is required. GPU backends are optional feature flags.
- No hidden network access, telemetry, or background downloads.

Prefer these C++20 features when they materially improve safety or clarity
- RAII handle wrappers
- std::unique_ptr
- selective std::shared_ptr for shared loaded models
- std::span
- std::string_view
- std::optional
- std::variant
- std::filesystem
- std::chrono
- std::jthread + std::stop_token
- enum class
- constexpr
- [[nodiscard]]
- noexcept
- concepts only if they improve readability and diagnostics

Avoid
- raw owning pointers
- giant template metaprogramming
- global mutable state
- macro-heavy design
- Godot-facing APIs that expose implementation details
- main-thread blocking
- placeholder pseudocode in critical paths

Build this in reviewable increments, in this order

1. Repository bootstrap
Create:
- top-level CMakeLists.txt
- CMakePresets.json
- cmake/ helper files
- .clang-format
- .clang-tidy
- .editorconfig
- .gitignore
- GitHub Actions CI
- docs/BUILD.md
- docs/ARCHITECTURE.md
- CHANGELOG.md

2. Dependency integration
- Add pinned submodules for godot-cpp and llama.cpp
- Wire them into the top-level CMake build
- Ensure compile_commands.json is produced
- Ensure a clean out-of-source build works

3. Internal wrapper layer
Implement a small, stable wrapper around libllama with:
- explicit model ownership
- explicit context ownership
- sampler/request abstractions
- structured error type
- deterministic config translation
- no Godot dependencies in this layer

4. Core async runtime
Implement:
- request queue
- worker execution
- cooperative cancellation
- main-thread-safe result delivery mechanism
- centralized UTF-8 conversion helpers
- testable boundaries

5. Godot-facing API
Create and bind a minimal API. Prefer:
- `LlamaModelConfig` as a `Resource`
- `LlamaSession` as a `RefCounted` unless a `Node` is strongly justified

`LlamaModelConfig` should cover at least:
- model_path
- n_ctx
- n_threads
- n_batch
- n_gpu_layers
- seed
- use_mmap
- use_mlock
- embeddings_enabled
- chat_template_override (optional)
- backend preferences if applicable

`LlamaSession` should provide at least:
- open(config) -> int or Error
- close()
- is_open() -> bool
- generate_async(prompt: String, options: Dictionary = {}) -> int request_id
- cancel(request_id: int)
- tokenize(text: String, add_bos := false, special := false) -> PackedInt32Array
- detokenize(tokens: PackedInt32Array) -> String
- embed(text: String) -> PackedFloat32Array

The generation options dictionary should support documented snake_case keys such as:
- max_tokens
- temperature
- top_p
- top_k
- min_p
- repeat_penalty
- stop
- seed_override

Bind signals:
- opened()
- token_emitted(request_id, token_text, token_id)
- completed(request_id, text, stats)
- failed(request_id, error_code, error_message, details)
- cancelled(request_id)

Godot API rules
- Public methods/signals/properties must use Godot-friendly naming.
- Keep bound methods Variant-compatible.
- Use Godot types at the boundary:
  String, PackedByteArray, PackedInt32Array, PackedFloat32Array, Array, Dictionary,
  Callable, Error, RefCounted, Resource, Node only when necessary.
- Centralize UTF-8 conversion.
- Emit signals on the Godot main thread only.
- Document blocking vs non-blocking behavior for every bound method.
- For bulk PackedArray handling, use efficient pointer-based access helpers internally.

6. Demo project
Create a minimal Godot demo that:
- loads the extension
- creates a config
- opens a model
- sends a prompt
- streams tokens into a text box
- shows failure and cancellation states

7. Tests
Create:
- pure C++ unit tests for wrapper/core logic
- at least one headless Godot integration test
- deterministic smoke tests with bounded runtime

Testing rules
- no network
- fixed seed where relevant
- bounded token counts
- thread count pinned for determinism unless concurrency itself is under test
- if a real model is used in CI, it must be legally redistributable and minimal

CI requirements
Minimum matrix:
- ubuntu-latest
- windows-latest
- macos-latest

Minimum jobs:
- configure
- build
- unit tests
- at least one integration/smoke test
- formatting
- lint

Recommended:
- ASan + UBSan on one POSIX job
- TSan on a dedicated concurrency-related job
- release artifact packaging matching the `.gdextension` manifest

Important design requirements
- Keep model lifetime separate from request lifetime.
- Avoid repeated String <-> UTF-8 conversions in hot paths.
- Avoid hidden heap churn inside token loops.
- Keep public Godot API intentionally smaller than the underlying llama.cpp API.
- Shield the rest of the codebase from upstream llama.cpp API churn.
- Do not fork upstream libraries unless absolutely necessary.
- If using shared ownership, document exactly why.

Execution style
- Work in small, buildable increments.
- After each major step, summarize:
  - what changed
  - why it changed
  - key tradeoffs
  - any assumptions
- Do not leave pseudocode or TODO stubs in core paths.
- Produce real code that can be built and tested.

Final output requirements
When finished, provide:
- final directory tree
- exact build commands
- exact test commands
- exact dependency pin locations
- short architecture summary
- public API summary
- known limitations
- assumptions made
