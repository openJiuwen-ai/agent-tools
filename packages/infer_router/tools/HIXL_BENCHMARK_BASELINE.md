# HIXL Benchmark Baseline

Best measured baseline recorded on 2026-09-17:

- Commit: `c4939444a2f6b2099a827e7a5222a363c3f25989`
- Branch: `dev_opt`
- 4P/4D, concurrent 32, 640 requests, prompt 20500, output 256.
- TTFT mean 1761.50 ms, P99 18354.50 ms.
- TPOT mean 65.66 ms/token; E2E mean 18504.28 ms.
- Router route mean 675.44 ms; capacity wait mean 32.51 ms.
- Throughput 1.676 requests/s over the Router-inferred window.
- Near-full reuse 87.81%; aggregate block reuse 85.63%.
- Zero proxy errors and health timeouts; all 640 outputs were 256 tokens.

This is a single-run reference, not a claim of statistically established
superiority. Subsequent diagnostics must not amend this baseline commit.

## Optional Model-Step Diagnostics

Set `JIUWEN_KVBM_MODEL_STEP_SAMPLE_EVERY=20` in HIXL Worker processes and
restart them to sample every twentieth `execute_model` call. Set
`JIUWEN_KVBM_MODEL_STEP_MAX_SAMPLES=1000` to bound records per runner (default
1000). Sampling is disabled by default. `[KVBMModelStepTiming]` reports host
call duration and scheduled request/token counts, not device-completion time
or per-request ITL. It does not synchronize the NPU. Execution intervals are
reported as Unix timestamps for correlation with import/attach logs.

The analyzer accepts explicit worker mappings, for example:

```sh
python3 packages/infer_router/tools/analyze_pd_logs.py \
  --router-log router-prefix-pd.log --decode-log d1.log d2.log d3.log d4.log \
  --worker-log worker-2-decode=d1.log --worker-log worker-4-decode=d2.log \
  --worker-log worker-6-decode=d3.log --worker-log worker-8-decode=d4.log
```

Generation-window joins use only snapshots inside the reconstructed
first-token-to-completion interval. No nearest-snapshot extrapolation is used.
At least two samples and no boundary/internal gap greater than 1.5 times the
configured sample interval are required for adequate coverage. These are coarse
observations, not time-weighted batch trajectories or causal evidence.
