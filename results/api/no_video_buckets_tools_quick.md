# tools (no_video_buckets)

- base_url: http://127.0.0.1:8000  model: mimo-v2.6-flash
- started: 2026-10-21T22:44:48Z  finished: 2026-10-21T22:44:50Z
- counts: pass 2, partial 0, fail 0, info 0

| case | verdict | desc | reason |
|---|---|---|---|
| T1 | pass | single tool call (auto) | calls [('get_weather', {'city': '北京'})] |
| T4 | pass | streaming tool calls (T1 only) | T1: 1 calls match non-stream |
