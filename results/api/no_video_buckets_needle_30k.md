# needle (no_video_buckets_30k)

- base_url: http://127.0.0.1:8000  model: mimo-v2.6-flash
- started: 2026-10-21T23:01:24Z  finished: 2026-10-21T23:01:49Z
- counts: pass 3, partial 0, fail 0, info 0

| case | verdict | desc | reason |
|---|---|---|---|
| reuse:N1@30000@10 | pass | reuse N1 len 30000 depth 0.1 (from needle_ctx32k_prompts.json) | correct: content='7492-ALPHA' answer='7492-ALPHA' prompt_tokens=30006 ttft=7.972s t_seg=0.9965s |
| reuse:N2@30000@50 | pass | reuse N2 len 30000 depth 0.5 (from needle_ctx32k_prompts.json) | correct: content='Biscuit-42' answer='Biscuit-42' prompt_tokens=30005 ttft=7.971s t_seg=0.9964s |
| reuse:N3@30000@90 | pass | reuse N3 len 30000 depth 0.9 (from needle_ctx32k_prompts.json) | correct: content='18' answer='18' prompt_tokens=30003 ttft=7.96s t_seg=0.995s |
