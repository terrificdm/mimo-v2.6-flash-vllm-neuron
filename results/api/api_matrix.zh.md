# API 能力矩阵（推理解析、工具调用、采样 / API 参数）

结论：推理解析、工具调用和 17 个采样 / API 参数在 Trainium2 上都能用，默认的设备采样模式（模式 A）和 CPU 采样模式（模式 B）下都生效。两处限定：

- `--generation-config` 用例在 1M 配置（模式 A）上记为 info：用例不设 max_tokens 且忽略 EOS（序列结束 token），1M 配置下会生成到接近 1M token、900 s 超时；只在 4096 档验证通过。1M 配置（去掉视频档位）的 params A：16 pass + 1 info（[no_video_buckets_params_mode_a.md](no_video_buckets_params_mode_a.md)）。
- 原生 FP8 下模式 B 的原始结果是 15 pass + 2 partial（`temperature_top_p`、`top_k`）。两个 partial 每次分叉都发生在前两名 logprob 完全相等的位置（BF16 logits 的平局），判为通过（[design-notes.md](../../docs/design-notes.md#sampling-ties)）。

设备采样器本身不支持 `seed`、`presence_penalty`、`frequency_penalty`（不处理时会被静默忽略）。插件在批里有这类请求时，用模型已经输出的全词表 logits 在 CPU 上跑 vLLM 原生采样器重采样（`runner_patches._wants_cpu_resample`，默认开，`MIMO_ODS_CPU_RESAMPLE=0` 关），其他请求仍用设备采样；不改编译图（[design-notes.md](../../docs/design-notes.md#cpu-resample)）。

配置：
- 默认（模式 A）：README 的配置 A（1M + 多模态，`--generation-config vllm`）。本页的完整结果是用配置 A 的命令**去掉视频档位**（不设 `MIMO_MM_VISION_BUCKETS`、不装 soundfile）测的；带视频档位和 soundfile 的推荐配置只做了缩减回归，见文末。
- 模式 B：4096 档，`MIMO_API=1 MIMO_SAMPLING=cpu MIMO_GENCFG=auto`（neuron_config 里 `"on_device_sampling_config": null`），即 README 的配置 B 再加这三个变量（README 配置 B 下面的“验证范围”一段）。
- 4096 档模式 A：`MIMO_API=1`（不加其他开关）。

| 能力 | 模式 A | 模式 B | 证据 |
|---|---|---|---|
| 推理解析 `--reasoning-parser mimo`（流式 / 非流式，思考开 / 关） | 支持（R1–R8） | 支持（R1–R8） | [no_video_buckets_reasoning_quick.md](no_video_buckets_reasoning_quick.md)（1M 快速回归）；`scripts/api_tests/test_reasoning.py` |
| 工具调用 `--tool-call-parser mimo`（单个、并行、多轮、流式、auto / none / required / 指定函数、4 并发混批） | 支持（T1–T10 + T2b） | 支持（T1–T10 + T2b） | [no_video_buckets_tools_quick.md](no_video_buckets_tools_quick.md)；`scripts/api_tests/test_tools.py` |
| temperature + top_p、top_k、stop、stop_token_ids、max_tokens | 支持 | 支持（原生 FP8 的 top_p / top_k 两个 partial 是平局，见上） | [no_video_buckets_params_mode_a.md](no_video_buckets_params_mode_a.md) |
| logprobs / top_logprobs（chat、completions） | 支持 | 支持 | 同上 |
| n>1、stream + stream_options.include_usage（chat、completions） | 支持 | 支持 | 同上 |
| response_format json_object / json_schema | 支持 | 支持 | 同上 |
| `--generation-config`（A 用 vllm，B 用 auto） | 支持（4096 档验证；1M 配置用例超时，记 info） | 支持 | 同上 |
| seed | 支持（CPU 重采样） | 支持 | 同上 |
| presence_penalty、frequency_penalty | 支持（同上） | 支持 | 同上 |

设备采样器不支持这 3 项的原因：vllm-neuron 的设备采样器每个请求只接收 [top_k, top_p, temperature] 三个值（`vllm_neuron/nn/sampler.py:70`、`152-154`），没有随机数生成器和惩罚项输入。插件用上面的 CPU 重采样绕过，不改设备采样器。

插件为这两种模式修了 vllm-neuron 0.24.0.1.1.0 运行器的 5 个“单个合法请求打挂整个引擎”的问题。修法是运行时补丁 `plugin/mimo_v2_neuron/runner_patches.py`，只在构建 MiMoV2 模型时安装，不改已安装的文件：
1. 带 `seed`：在 Neuron 设备上建 `torch.Generator` 报 NotImplementedError → 改在 CPU 上建（A、B）。
2. 模式 A，temperature>0 时要 logprobs：用设备上的采样元数据跑 CPU 采样器 → 直接对设备实际输出的 token 取 logprobs。
3. 模式 B，temperature>0：CPU logits 配上设备上的采样元数据 → 每步用 CPU 副本。
4. 模式 B，n>1 或并发数不等于桶大小：logits 保留 decode 补齐行，与元数据行数不符 → 采样前去掉补齐行。
5. 模式 B，多个结构化输出请求在补齐过的 decode 批里：语法掩码行数不符 → 掩码补成“全允许”行。

其他发现：
- vLLM 0.24 的响应字段叫 `reasoning`（不再返回 `reasoning_content`），请求里两种写法都接受（[design-notes.md](../../docs/design-notes.md#reasoning-field)）。
- `tool_choice=none` 时 vLLM 默认仍把工具放进提示，模型可能在 content 里写出 `<tool_call>` 原文。GPU 上的 vLLM 默认行为也一样；可选开关 `--exclude-tools-when-tool-choice-none` 没有启用。
- `/v1/completions` 不传 `max_tokens` 时默认只生成 16 个 token（vLLM / OpenAI 的行为）。
- `--generation-config auto`（模式 B）会把没有显式给出 temperature 的请求默认成 T=1.0、top_p=0.95，max_tokens 默认 2048。

完整的逐参数矩阵（每个参数在 A / B 下的结论、原因和代码位置）可以用 `scripts/api_tests/param_matrix.py` 从两种模式的 `test_params.py` 结果重新生成；原始结果文件没有随仓库提供。

## 推荐配置（README 配置 A）的缩减回归

视频档位和 soundfile 只影响多模态输入（视频最多 16384 个视觉 token、每个请求合计最多 16 个编码块，超了返回 HTTP 400；音频支持 WAV / FLAC / MP3），文本侧 API 不变。带这两项跑的部分全部通过：

| 套件 | 结果 | 文件 |
|---|---|---|
| smoke（10 个请求） | PASS | [config_a_smoke.md](config_a_smoke.md) |
| 推理解析 `--quick` | 2/2 | [config_a_reasoning_quick.md](config_a_reasoning_quick.md) |
| 工具调用 `--quick` | 2/2 | [config_a_tools_quick.md](config_a_tools_quick.md) |
| 采样 / API 参数，模式 A，`--quick` | 3/3 | [config_a_params_mode_a_quick.md](config_a_params_mode_a_quick.md) |
| 多模态全套（图片、base64、视频、音频） | 8 pass + 2 info | [config_a_mm.md](config_a_mm.md) |
| needle 30K / 940K | 3/3；1/1（940K 只跑了深度 0.5） | [config_a_needle_30k.md](config_a_needle_30k.md)、[config_a_needle_940k_depth50.md](config_a_needle_940k_depth50.md) |

推理解析、工具调用、参数的完整版（R1–R8、T1–T10 + T2b、17 个参数）没有带视频档位重跑。
