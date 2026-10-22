# mm (no_video_buckets)

- base_url: http://127.0.0.1:8000  model: mimo-v2.6-flash
- started: 2026-10-21T22:43:40Z  finished: 2026-10-21T22:43:53Z
- counts: pass 8, partial 0, fail 0, info 2

| case | verdict | desc | reason |
|---|---|---|---|
| I1 | pass | red_square.png (file:// URL) | answer 'Red' contains 'red' (prompt_tokens 222) |
| I2 | pass | number_739.png (file:// URL) | answer '739' contains '739' (prompt_tokens 222) |
| I3 | pass | three_circles.png (file:// URL) | answer '3' contains '3' (prompt_tokens 221) |
| I4 | pass | word_trainium.png (file:// URL) | answer 'TRAINIUM' contains 'TRAINIUM' (prompt_tokens 222) |
| B1 | pass | number_739.png (base64 data URL) | answer '739' contains '739' (prompt_tokens 222) |
| A1 | info | architecture.png (base64, qualitative, not scored) | answer 'Based on the image provided, this figure illustrates the architecture of a multimodal AI model named **MiMo**.\n\nHere is a brief breakdown of its components:\n\n*   **Multimodal Inputs:** On the left, the model processes two types of input: **Visual Encoder** (handling images) and **Audio Encoder' (prompt_tokens 1012, not scored) |
| V1 | pass | digits_1234.mp4 (file:// video_url) | answer '1, 2, 3, 4' contains '1, 2, 3, 4' (prompt_tokens 636) |
| U1 | pass | jfk.wav (file:// audio_url, transcription) | answer 'And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.' contains 'And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.' (prompt_tokens 87) |
| U2 | pass | beeps_3.wav (input_audio base64 wav) | answer '3' contains '3' (prompt_tokens 43) |
| U3 | info | tone_440.wav (file:// audio_url, qualitative, not scored) | answer 'This is a continuous, steady electronic tone, and there is no speech.' (prompt_tokens 37, not scored) |
