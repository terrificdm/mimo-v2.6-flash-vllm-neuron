# params (config_a)

- base_url: http://127.0.0.1:8000  model: mimo-v2.6-flash
- started: 2026-10-22T05:44:05Z  finished: 2026-10-22T05:44:29Z
- counts: pass 3, partial 0, fail 0, info 0

| case | verdict | desc | reason |
|---|---|---|---|
| temperature_top_p | pass | temperature=1.0 + top_p=0.95 | 5/5 distinct answers; top_p=0.001 == greedy: True; nucleus 0 out of 72 covered steps (82 total) |
| stream_usage_chat | pass | stream + stream_options.include_usage（chat） | last chunk usage {'prompt_tokens': 22, 'total_tokens': 54, 'completion_tokens': 32} == non-stream |
| stream_usage_completions | pass | stream + stream_options.include_usage（completions） | last chunk usage {'prompt_tokens': 5, 'total_tokens': 21, 'completion_tokens': 16} == non-stream |
