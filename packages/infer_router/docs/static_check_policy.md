# Static-check scope and compatibility exceptions

The PR !107 CodeCheck report dated 2026-11-04 contained 984 findings.
The exported report is a snapshot; local checks do not establish that the
remote CodeCheck gate has passed. Run the gate again after updating the PR.

The follow-up export contained 141 findings and no severe findings. This round
uses function-scoped disable/enable pairs instead of disable-next for required
test-double and protocol signatures, for compatibility with older checkers.
Each exception remains limited to the relevant function and rule. Acceptance
of these comments must still be verified by the remote scanner.

Slice delimiters follow CodeCheck's no-space-before-colon convention, which
differs from Black-style formatting for complex slices. The normalization was
checked to preserve the Python AST. Remaining comprehension, conditional and
dictionary-access findings were addressed directly.

## Code changes

- Production invariant checks use explicit exceptions instead of `assert`, so
  optimized Python execution retains the checks.
- Exception handlers do not redundantly list parent and child exception types.
- Diagnostic fallback and rollback failures are logged without skipping the
  existing cleanup paths.
- Cross-component callers use public tokenizer, worker-control, prediction and
  offload interfaces. Existing private implementations remain for compatibility.
- Complex comprehensions and conditionals are split into named data or explicit
  loops, preserving the capacity checks, locks and reservation lifecycle.
- Analysis and benchmark commands use `write_report` for stdout/stderr result
  data. These results are not log records: adding logging prefixes or moving
  them to stderr would change their CLI contract. Output separators, newlines,
  target streams and explicitly requested flushing are preserved.
- The cache-lock decorator preserves method metadata using `functools.wraps`.

## Narrow exceptions

Exceptions must name an individual rule and explain the relevant contract.
Do not disable all checks or exclude production source directories.

| Scope | Rule | Reason |
| --- | --- | --- |
| White-box test modules marked at their header | `protected-access` | Tests deliberately exercise private hooks, failures, reservations and internal state. Making all those internals public solely for tests would weaken the interface boundary. |
| Individual fake bound methods | `no-self-use` | vLLM monkey-patch tests require the original bound method signature. Changing these doubles to static methods can change descriptor and wrapper behavior. |
| `BaseHTTPRequestHandler.do_GET` / `do_POST` | `invalid-name` | The HTTP server dispatches by these exact standard-library method names. |
| vLLM block rollback compatibility branch | `protected-access` | Older blocks have no `reset_hash()` method; the fallback clears `_block_hash` only during rollback. |
| vLLM-Ascend EngineCore entrypoint restoration | `protected-access` | The upstream patch module requires restoring its original spawn entrypoint. |

Built-in scorers now use explicit registration immediately after each class
definition. `register_scorer` still returns the original class for compatibility
with existing callers; no wrapper or class metadata mutation is introduced.

The Rust completion callback now adapts its positional arguments into a named
completion event before processing. Existing short and full calls remain
compatible without a `too-many-arguments` exception.

The next code-only cleanup removes 82 of the 114 reported test-method patterns:
59 stateless methods become static methods, and 23 repeated ModelRunner methods
use a shared stateful runner implementation. Local subclasses remain distinct
to isolate hook installation and restoration. The remaining 32 methods retain
bound signatures because production wrappers retrieve them from the class and
explicitly pass the instance. Converting these indiscriminately to static
methods causes real regression failures. HTTP dispatch names and the Rust
callback ABI remain unchanged. These remaining findings are not claimed fixed.

At the user's request, six tests that primarily asserted diagnostic log text
were removed. Native connector metadata forwarding, async computed-token
restoration, runtime scheduler wrapping/restoration, setup compatibility and
Ascend entrypoint recursion prevention remain covered. This removes another
12 reported unused-instance methods; 20 bound-method patterns remain. Four
runner subclasses with custom constructors now initialize their base state.
Production diagnostics and capacity safety mechanisms are unchanged.

The shared Ascend hook suite subsequently removes only the scheduler cache-hit
log and detailed prefix-lookup log tests. Safety and performance regressions
remain covered. Its remaining fake managers expose configured event lists and
computed results, fake block pools retain inserted block mappings, and the
scheduler records model outputs with an explicit assertion on those outputs.
All 15 unused-instance suppressions in that file are removed without changing
production hooks or replacing their required bound-method signatures.

The final five reported test-method patterns are also stateful now: the
Mooncake scheduler stores registered requests, readiness and scheduling output;
the 0.23 block pool retains cached block mappings; the scheduler-owned allocation
test configures its computed-block result on the manager. Assertions verify
these states while retaining all existing functional tests. The three standard
HTTP dispatch names and the completion callback ABI are not modified.

## Validation

Run the Router, Worker and KVBM unit suites together:

```bash
PYTHONPATH=packages/infer_router/src:packages/jiuwen-kvbm/src \
  .venv/bin/python -m pytest \
  packages/infer_router/tests/unit \
  packages/infer_router/src/openjiuwentools/infer_router/vllm/tests \
  packages/jiuwen-kvbm/tests -q
```

The TCP/HTTP tests require permission to bind loopback ports. This suite uses
test doubles and does not replace real NPU/vLLM-Ascend deployment validation.
