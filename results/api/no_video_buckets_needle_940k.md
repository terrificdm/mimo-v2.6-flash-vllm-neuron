# needle (no_video_buckets)

- base_url: http://127.0.0.1:8000  model: mimo-v2.6-flash
- started: 2026-10-21T23:01:54Z  finished: 2026-10-21T23:18:09Z
- counts: pass 3, partial 0, fail 0, info 0

| case | verdict | desc | reason |
|---|---|---|---|
| reuse:N1@940000@10 | pass | reuse N1 len 940000 depth 0.1 (from needle_fp8_ctx1m_prompts.json) | correct: content='7492-ALPHA' answer='7492-ALPHA' prompt_tokens=940004 ttft=323.171s t_seg=1.4051s |
| reuse:N2@940000@50 | pass | reuse N2 len 940000 depth 0.5 (from needle_fp8_ctx1m_prompts.json) | correct: content='Biscuit-42' answer='Biscuit-42' prompt_tokens=940006 ttft=323.249s t_seg=1.4054s |
| reuse:N3@940000@90 | pass | reuse N3 len 940000 depth 0.9 (from needle_fp8_ctx1m_prompts.json) | correct: content='18' answer='18' prompt_tokens=940006 ttft=323.27s t_seg=1.4055s |
