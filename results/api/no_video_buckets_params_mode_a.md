# params (no_video_buckets)

- base_url: http://127.0.0.1:8000  model: mimo-v2.6-flash
- started: 2026-10-21T22:44:51Z  finished: 2026-10-21T23:01:10Z
- counts: pass 16, partial 0, fail 0, info 1

| case | verdict | desc | reason |
|---|---|---|---|
| temperature_top_p | pass | temperature=1.0 + top_p=0.95 | 5/5 distinct answers; top_p=0.001 == greedy: True; nucleus 0 out of 78 covered steps (85 total) |
| top_k | pass | top_k | top_k=1 == greedy; top-5 membership ok over 54 steps |
| seed | pass | seed | same seed identical, different seed differs |
| stop | pass | stop（字符串） | content '1, 2, 3, 4' |
| stop_token_ids | pass | stop_token_ids | stopped before token 20; content '1, 2, 3, 4, ' |
| max_tokens | pass | max_tokens | completion_tokens=7 finish_reason='length' |
| logprobs_chat | pass | logprobs / top_logprobs（chat） | 32 greedy entries ok; 32 T=1.0 entries consistent |
| logprobs_completions | pass | logprobs（completions） | 16 greedy entries ok; 16 T=1.0 entries consistent |
| n_chat | pass | n>1（chat） | 3/3 distinct |
| n_completions | pass | n>1（completions） | 3/3 distinct |
| stream_usage_chat | pass | stream + stream_options.include_usage（chat） | last chunk usage {'prompt_tokens': 22, 'total_tokens': 54, 'completion_tokens': 32} == non-stream |
| stream_usage_completions | pass | stream + stream_options.include_usage（completions） | last chunk usage {'prompt_tokens': 5, 'total_tokens': 21, 'completion_tokens': 16} == non-stream |
| presence_penalty | pass | presence_penalty | greedy apple x39 unchanged with presence_penalty=1.5 (a presence penalty is subtracted once per distinct output token, too small to break the repeat instruction); seeded T=1.0 check: same seed without penalty identical, with presence_penalty=2.0 differs -> penalty applied |
| frequency_penalty | pass | frequency_penalty | apple 39 -> 9 |
| json_object | pass | response_format: json_object | JSON object keys ['name', 'population'] |
| json_schema | pass | response_format: json_schema | schema valid (jsonschema=yes): '{\n  "name": "北京",\n  "population": 21540000,\n  "landmarks": [\n    "长城",\n    "故宫"\n  ]\n}' |
| generation_config | info | --generation-config | request error / timeout: {'error': "TimeoutError('timed out')"} |
