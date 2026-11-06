# PD Prompt Consistency Diagnostics

Set `JIUWEN_PD_PROMPT_DIAGNOSTICS=1` in the Router and ALL P/D Worker startup
environments, then restart them. The flag is off by default. EngineCore must
inherit the Worker environment. No scheduling, tokenization, allocation, or
fallback policy is changed. Disable after diagnosis: hashing costs O(prompt size).

Collect one request and its engine alias across Router/P/D logs:

```bash
REQ='chatcmpl-...'
RID='engine-alias-without-vllm-suffix'
rg "$REQ|$RID" router*.log p*.log d*.log |
  rg 'PDPromptDiagnostics|PDSourceCaptureDiagnostics|PDTargetPromptDiagnostics|PDFallbackDiagnostics|PDGenerationDiagnostics|Complete'
```

- `PDPromptDiagnostics`: request/engine/logical IDs, raw messages/tools digests,
  token count/digest, raw tokenizer result container type, template identity and
  effective argument digest. Router additionally logs its formatted-message
  digest. Tool-choice and caller-provided template arguments are hashed, not
  asserted to be honored. Router cache/fallback paths may not use the template;
  the Router's argument digest describes its configured chat-template path.
- `PDSourceCaptureDiagnostics`: intent/engine/capture lengths and both engine
  and captured token digests before partial block export. `truncated=True` does
  not fix anything; it identifies the length clamp.
- `PDTargetPromptDiagnostics`: actual EngineCore D token identity at gate entry.
- `PDFallbackDiagnostics`: gate release and first scheduled step afterward,
  followed by request removal from Scheduler or observation expiry. Removal is
  NOT proof of successful completion (it can be abort/failure). Observations are
  bounded to 2048 requests and 600 seconds, checked on scheduler cycles.
- `PDGenerationDiagnostics`: first engine output and completion/error/cancellation
  for chat generation. First engine output is not necessarily the client's first
  visible token; final client success still requires Router Complete / HTTP result.

No complete prompts, token lists, tool definitions or credentials are added to
these markers. Hashes can still correlate repeated content; restrict log access.
Existing DEBUG body logs are independent and can expose content: do not enable
global DEBUG just for these INFO markers.

Compare token digests as well as lengths. If P/D agree but Router differs, inspect
Router normalization/template inputs; if source capture differs from P's actual
engine identity, inspect the export intent length source. Never bypass partial
block validation to make an import succeed.

## Tool-history consistency fix

Router chat-template input now preserves complete message dictionaries, including
`tool_calls`, `tool_call_id`, `name`, structured content and JSON null. Only plain
role/text conversations use the previously validated incremental message-prefix
tokenizer cache; tool schemas and structured histories use full template
tokenization. This does not disable vLLM prefix caching or HIXL delta reuse.

Restart Router after updating. For an identical tool-history request, Router and
P/D `effective_messages_sha256`, token count and token digest should agree when
their tokenizer/template configuration agrees. Source capture should no longer
be truncated due to dropped tool history. Export safeguards and partial-block
fingerprint validation remain unchanged; any remaining mismatch must still fail
safe rather than attaching unverified KV.

## Duplicate physical cache replicas

A logical cache hash can have multiple physical copies in one cache group.
P cached export now selects one valid copy per `(logical_index, group_id)`:
prefer a referenced copy, then the lowest block ID for deterministic selection.
Different groups are retained. Stale index entries, unpublished and null blocks
are excluded, and conflicting packed hashes within one group fail safely.
Resolution and source lease acquisition remain under the shared BlockPool lock.

For the single-group 4992-token/128-block case, P should export 39 logical and
39 physical blocks, even if the final hash has two cached physical copies.
Do not generalize equal logical/physical counts to multiple cache groups.
