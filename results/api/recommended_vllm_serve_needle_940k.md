# needle (recommended_vllm_serve_940k)

- base_url: http://127.0.0.1:8000  model: mimo-v2.6-flash
- started: 2026-10-22T17:09:14Z  finished: 2026-10-22T17:12:41Z
- counts: pass 1, partial 0, fail 0, info 0

| case | verdict | desc | reason |
|---|---|---|---|
| reuse:N2@940000@50 | pass | reuse N2 len 940000 depth 0.5 (from needle_fp8_ctx1m_prompts.json) | correct: content='Biscuit-42' answer='Biscuit-42' prompt_tokens=940006 ttft=204.764s t_seg=0.8903s |
