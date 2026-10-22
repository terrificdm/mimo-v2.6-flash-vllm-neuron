# reasoning (recommended)

- base_url: http://127.0.0.1:8000  model: mimo-v2.6-flash
- started: 2026-10-22T14:44:16Z  finished: 2026-10-22T14:44:26Z
- counts: pass 2, partial 0, fail 0, info 0

| case | verdict | desc | reason |
|---|---|---|---|
| R1 | pass | thinking on, non-stream | has 'reasoning' key=True, has 'reasoning_content' key=False |
| R2 | pass | thinking on, stream | concat == non-stream; 96 chunks; delta keys ['reasoning'] |
