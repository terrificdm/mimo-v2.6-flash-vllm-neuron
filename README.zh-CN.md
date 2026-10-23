# 教程：用 vLLM Neuron 部署 MiMo-V2.6-Flash-MOPD

[English](README.md) | 简体中文

本教程在一台 AWS **trn2.48xlarge** 上，用 vLLM 和 vllm-neuron 插件部署
[XiaomiMiMo/MiMo-V2.6-Flash-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD)：一条 `vllm serve` 命令同时提供
**1M token 上下文和图片、视频、音频输入**。模型支持是 [plugin/](plugin/) 里的一个 vLLM 通用插件（`mimo_v2_neuron`）；
权重在实例上转换成 trn2 能用的 FP8 格式。

流程：启动实例 → 安装插件 → 下载权重 → 转换权重 → `vllm serve` → 发请求。

## 目录

- [概述](#概述)
- [前置条件](#前置条件)
- [步骤 1 安装插件](#步骤-1-安装插件)
- [步骤 2 下载权重](#步骤-2-下载权重)
- [步骤 3 转换权重](#步骤-3-转换权重)
- [步骤 4 启动服务](#步骤-4-启动服务)
- [步骤 5 发请求](#步骤-5-发请求)
- [步骤 6 基准测试](#步骤-6-基准测试)
- [步骤 7 精度（可选）](#步骤-7-精度可选)
- [已知限制](#已知限制)
- [后续计划](#后续计划)
- [故障排查](#故障排查)
- [相关资料](#相关资料)
- [许可证](#许可证)

## 概述

推荐命令下可用的功能（作者用本仓库代码验证过）：

- **1M 上下文**（1,048,576 token），FP8 权重（线性层 FP8 权重 × FP8 激活；专家 FP8 编码 × BF16 激活，数值上等于 GPU 的
  MXFP4 × BF16），BF16 KV 缓存——与官方 H100 vLLM 部署的精度相同。
- **图片、视频、音频**在同一个服务里；视觉编码器和两个音频塔都在 Trainium 上跑。
- **前缀缓存**：重复或续写的长提示几秒内出首个 token。
- **推理解析和工具调用**用 vLLM 的 `mimo` 解析器；采样参数包括 `seed` 和各种惩罚项。

关键数字（一台 trn2.48xlarge，详见[步骤 6](#步骤-6-基准测试)）：

| 测量项 | 结果 |
|---|---|
| 短请求（输入 512 / 输出 128），并发 1 | TTFT（首 token 时间）198.3 ms，TPOT（每输出 token 时间）47.15 ms |
| 940K / 30K token 首个 token（第一次请求） | 204.2 s / 3.83–3.88 s |
| 同一提示再发一次（命中前缀缓存），940K / 30K | 2.89 s / 0.25 s |
| 大海捞针，30K / 940K | 3/3 / 测过的深度全部正确 |
| GSM8K / MMLU（各 200 题，0-shot，贪心） | 98.0 % / 80.5 % |

## 前置条件

**实例**

- 一台 **trn2.48xlarge**（16 颗 Trainium2 芯片，LNC=2 时 64 个逻辑 NeuronCore，每核 24 GiB HBM），来自 EC2 Capacity Block
  for ML 或按需容量。
- 安全组入站**只开 SSH**。服务监听 `127.0.0.1:8000`，用 `ssh -L 8000:127.0.0.1:8000 ...` 访问。
- 根卷用的是 500 GiB gp3；模型文件放在本地 NVMe 盘上（见下）。

**AMI 和软件**——**Neuron SDK 2.32.0** 的 Neuron 多框架 DLAMI（Ubuntu 24.04）。在
[Neuron 多框架 DLAMI 指南（2.32.0）](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/setup/multiframework-dlami.html)
里通过 SSM 参数查你所在 Region 的 AMI ID。它自带 venv `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0`。

> [!IMPORTANT]
> 只在这些版本上验证过：
>
> | 组件 | 版本 |
> |---|---|
> | Neuron SDK | 2.32.0 |
> | vllm-neuron | 0.24.0.1.1.0 |
> | vLLM | 0.24.0 |
> | torch / torch-xla | 2.11 / 2.11 |
> | neuronx-cc | 2.27.5334.0 |
> | 后端 | vllm-neuron 默认的 XLA 后端（`neuron_libtorch`）；没有用 PyTorch 原生后端（`VLLM_NEURON_BACKEND=neuron_native`） |
>
> 不要在 venv 里升级或新增包（插件用 `--no-deps` 安装）。

**磁盘**——原始权重 166 GB + FP8 目录 349.6 GB（326 GiB）+ 多模态目录约 1.4 GB + 编译缓存（几十 GB）：至少准备 1 TB。
把四块实例存储 NVMe 盘做成 RAID 0（约 6.9 TiB；设备名用 `lsblk` 确认）：

```bash
sudo mdadm --create /dev/md0 --level=0 --raid-devices=4 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1
sudo mkfs.xfs /dev/md0
sudo mkdir -p /mnt/nvme && sudo mount -o noatime /dev/md0 /mnt/nvme && sudo chown "$USER:$USER" /mnt/nvme
mkdir -p /mnt/nvme/mimo-work /mnt/nvme/models /mnt/nvme/vllm-cache /mnt/nvme/hf-cache
ln -s /mnt/nvme/mimo-work ~/mimo-work
```

> [!WARNING]
> 实例停止后实例存储上的数据会丢失。重启后：
> `sudo mdadm --assemble --scan && sudo mount -o noatime /dev/md0 /mnt/nvme`。

**基础环境**

```bash
neuron-ls                                    # 16 个设备，logical-neuroncore-config 2
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/activate
command -v hf                                # venv 里的 Hugging Face CLI（huggingface_hub）
export HF_TOKEN=<your token>                 # 只有账号需要时才设
```

## 步骤 1 安装插件

约 1 分钟。

```bash
cd ~/mimo-work
git clone https://github.com/terrificdm/mimo-v2.6-flash-vllm-neuron.git
cd mimo-v2.6-flash-vllm-neuron
pip install --no-deps -e plugin
python -c "import mimo_v2_neuron; print('plugin ok')"
```

插件通过 vLLM 的 `vllm.general_plugins` 入口注册。插件开关的默认值就是验证过的配置，所以 `vllm serve` 不需要额外的环境变量
（[设计说明](docs/design-notes.md#plugin-defaults)）。

可选——**MP3 / FLAC / Ogg 音频**（WAV 不需要）：把 soundfile 装到 venv 外的一个目录，启动服务时把这个目录放进 `PYTHONPATH`。

```bash
pip install --no-deps --target ~/mimo-work/pydeps soundfile==0.14.0
PYTHONPATH=~/mimo-work/pydeps python -c 'import soundfile; print(soundfile.__version__, soundfile.__libsndfile_version__)'
# 预期：0.14.0 1.2.2
```

soundfile 自带 libsndfile 1.2.2，它的 IRCAM 读取器有一个公开 CVE（CVE-2025-52194）；插件只把 WAV / FLAC / Ogg / MP3 交给它。
安装前请自行评估（[音频设计说明](docs/design-notes.md#audio-input)）。

## 步骤 2 下载权重

```bash
export MIMO_MODELS=/mnt/nvme/models
tmux new -d -s download "bash scripts/download_weights.sh > ~/mimo-work/download.log 2>&1"
tail -f ~/mimo-work/download.log             # 最后一行是 "DONE <时间> <大小>"；连续失败 5 次退出码为 1
bash scripts/prepare_hf_reference.sh         # 模型 remote code 的补丁副本，转换工具要用
```

166 GB（129 个分片，revision `b12f5573350bb1704513fcf119fca48e498b0423`）下载到 `$MIMO_MODELS/MiMo-V2.6-Flash-MOPD`；
耗时取决于带宽（没有记录）。`hf download` 支持断点续传，失败后重跑同一条命令即可。

## 步骤 3 转换权重

**为什么要转换。** 权重里的线性层是 FP8 **E4M3FN**（最大 448），而 trn2 的 FP8 是 **E4M3**（最大 240）；专家是 **MXFP4**
（4 bit 数值，每 32 个元素共享一个指数），Neuron 的 MoE（混合专家）内核读不了。转换工具把线性层重新缩放到 E4M3，把专家
重新编码成每个输出通道一个缩放系数的 FP8。专家数值完全不变；线性层约 7.2×10⁻⁵ 的元素有变化（按新缩放取整）。
详见 [docs/gpu-alignment.md](docs/gpu-alignment.md)。

```bash
python tools/convert_checkpoint.py \
  --src $MIMO_MODELS/MiMo-V2.6-Flash-MOPD --dst $MIMO_MODELS/MiMo-V2.6-Flash-MOPD-fp8 --workers 16 --fp8 \
  --modeling-file reference/hf_patched/modeling_mimo_v2.py \
  --configuration-file reference/hf_patched/configuration_mimo_v2.py
```

约 **5 分钟**（16 个 worker 用时 321 s）；49 个分片，72,766 个张量，349.6 GB。可断点续跑。

逐元素对照原始权重校验（约 **40 分钟**）：

```bash
python tools/check_fp8_converted.py --src $MIMO_MODELS/MiMo-V2.6-Flash-MOPD \
  --dst $MIMO_MODELS/MiMo-V2.6-Flash-MOPD-fp8 --workers 12 --out ~/mimo-work/fp8_convert_check.json
```

预期：终端最后一行是 `experts lossless=True; linear changed 222672 / 3073376256 (7.25e-05); errors 0; -> ...`。
输出 JSON 里对应的字段是 `"pass": true`、`experts_mxfp4_to_fp8.changed: 0`、
`linear_e4m3fn_to_trn_e4m3.changed_fraction` 约 7.2×10⁻⁵
（参考：[results/conversion/fp8_convert_check_fulldir.json](results/conversion/fp8_convert_check_fulldir.json)）。

加上视觉编码器和音频塔（几秒；它们保持 BF16，文本分片是符号链接，所以 `-fp8` 目录要保留）：

```bash
OMP_NUM_THREADS=16 python tools/make_mm_checkpoint.py --audio --src $MIMO_MODELS/MiMo-V2.6-Flash-MOPD \
  --text $MIMO_MODELS/MiMo-V2.6-Flash-MOPD-fp8 --dst $MIMO_MODELS/MiMo-V2.6-Flash-MOPD-fp8-mm-audio
```

把测试媒体拷到服务允许读取本地文件的目录：

```bash
mkdir -p /mnt/nvme/mimo-work/mm_assets && cp -r assets/. /mnt/nvme/mimo-work/mm_assets/
(cd /mnt/nvme/mimo-work/mm_assets && sha256sum -c SHA256SUMS)
```

## 步骤 4 启动服务

推荐配置——**1M 上下文 + 图片 / 视频 / 音频 + 前缀缓存**：

```bash
tmux new -s serve                                       # Ctrl-b d 退出 tmux 窗口
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/activate
export VLLM_CACHE_ROOT=/mnt/nvme/vllm-cache             # 编译缓存放在 NVMe RAID 上
export PYTHONPATH=$HOME/mimo-work/pydeps                # 只有 MP3 / FLAC / Ogg 需要（步骤 1，可选）
vllm serve /mnt/nvme/models/MiMo-V2.6-Flash-MOPD-fp8-mm-audio \
  --served-model-name mimo-v2.6-flash --trust-remote-code \
  --tensor-parallel-size 64 --enable-expert-parallel \
  --max-model-len 1048576 --max-num-batched-tokens 4096 --max-num-seqs 4 --block-size 32 \
  --num-gpu-blocks-override 36864 --enable-prefix-caching \
  --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [4096], "num_seqs_buckets": [1, 4],
      "kv_segment_size_buckets": [4096], "enable_structured_outputs": true,
      "decode_context_length_buckets": [4096, 32768, 131072, 524288]},
    "vision_neuron_config": {"tp_size": 8, "dp_size": 8, "num_vision_tokens_buckets": [4096, 65536],
      "vision_attention_block_size": 4096, "encoder_cache_num_blocks": 65}}' \
  --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice --no-async-scheduling \
  --generation-config vllm --limit-mm-per-prompt '{"image": 2, "video": 1, "audio": 1}' \
  --allowed-local-media-path /mnt/nvme/mimo-work/mm_assets --mm-processor-kwargs '{"max_pixels": 1048576}' \
  --host 127.0.0.1 --port 8000
```

**耗时：**第一次启动要编译约 **1 小时**；之后复用编译缓存，约 **18 分钟**就绪（把 350 GB 加载到 64 个核上）。
`curl -sf http://127.0.0.1:8000/v1/models` 列出 `mimo-v2.6-flash` 就是就绪了。启动日志里每个 worker 打印一行
`MiMoV2 Neuron: switches: MIMO_FP8_IMPL=nki(default) ...`，列出每个插件开关的取值和来源。

第一次编译时在另一个 tmux 窗口跑 `bash scripts/mem_guard.sh`：主机内存低于 200 GB 时它会杀掉编译进程
（[原因](docs/design-notes.md#host-oom)）。

| 参数 | 作用 |
|---|---|
| `--tensor-parallel-size 64 --enable-expert-parallel` | 用满 64 个核；专家分成 64 份 |
| `--max-model-len 1048576` | 1M 上下文 |
| `--max-num-batched-tokens 4096`，`kv_segment_size_buckets [4096]` | 长提示按 4096 token 一段做 prefill（预填充）；插件为短提示加了一个 512 token 的档位 |
| `--num-gpu-blocks-override 36864` | KV 池 36,864 × 32 token；必须设，否则按剩余 HBM 算池大小，decode 图放不下 |
| `decode_context_length_buckets` | decode 注意力按 4K / 32K / 128K / 512K / 1M 分档编译，而不是总按 1M |
| `--enable-prefix-caching` | 重复的长前缀直接从 KV 缓存读 |
| `num_vision_tokens_buckets [4096, 65536]`，`encoder_cache_num_blocks 65` | 视频最多 16,384 个视觉 token；每个请求最多 16 个编码块（超过返回 HTTP 400） |
| `--limit-mm-per-prompt` | 每个请求最多 2 张图、1 段视频、1 段音频 |
| `--mm-processor-kwargs max_pixels 1048576` | 大于约 1024 × 1024 的图片会缩小（1,024 个视觉 token） |
| `enable_structured_outputs`（在 `neuron_config` 里），`--no-async-scheduling` | 在 vllm-neuron 上支持结构化输出（JSON schema、指定函数的 `tool_choice`） |
| `--generation-config vllm` | 不让模型的 generation_config.json 覆盖请求默认值 |

**变体——32K 上下文**（多模态功能相同，启动更快）：改成 `--max-model-len 32768`、`--num-gpu-blocks-override 8192`，并去掉
`"decode_context_length_buckets": [...]`。同样在零环境变量下验证过（smoke、多模态 8/8、30K 大海捞针 3/3 用时 3.26 s、基准测试）。
其他上下文（128K、512K、纯文本、BF16）是用更早版本的代码测的：[docs/configuration.md](docs/configuration.md)。

排查问题时如果要把某项插件优化切回旧实现，设对应的变量即可，例如 `MIMO_FP8_IMPL=emul`
（[docs/configuration.md](docs/configuration.md#plugin-switches)）。

> [!CAUTION]
> 服务**没有鉴权**。保持 `--host 127.0.0.1`；其他机器要访问时，在前面加一层带鉴权的代理。

## 步骤 5 发请求

模型名 `mimo-v2.6-flash`；默认开启思考（增量先出现在 `reasoning`，再出现在 `content`）。

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "mimo-v2.6-flash", "max_tokens": 2048, "temperature": 1.0, "top_p": 0.95,
  "messages": [{"role": "user", "content": "用三句话介绍长城。"}]}'
```

关闭思考：加 `"chat_template_kwargs": {"enable_thinking": false}`。流式：加 `"stream": true`。

图片 / 视频 / 音频用 `file://` URL（文件必须在 `--allowed-local-media-path` 下）：

```bash
A=/mnt/nvme/mimo-work/mm_assets
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "mimo-v2.6-flash", "temperature": 0, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": false},
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "file://'"$A"'/number_739.png"}},
    {"type": "text", "text": "What number is written in the image? Answer with the number only."}]}]}'
# 视频：{"type": "video_url", "video_url": {"url": "file://'"$A"'/digits_1234.mp4"}}   -> "1, 2, 3, 4"
# 音频：{"type": "audio_url", "audio_url": {"url": "file://'"$A"'/audio/jfk.wav"}}     -> 肯尼迪演讲原文
```

工具调用（OpenAI 格式，`"tools": [...]`、`"tool_choice": "auto"`）和 OpenAI Python SDK 的用法与任何 vLLM 服务相同。更多例子
（base64 音频、工具、SDK）：[docs/development.md](docs/development.md#request-examples)。

`scripts/api_tests/ask.py` 是一个小的流式客户端，把本地文件以 base64 发送（只用标准库）：

```bash
python3 scripts/api_tests/ask.py "17×23 等于多少？"
python3 scripts/api_tests/ask.py --no-think --image assets/number_739.png "图里是什么数字？"
python3 scripts/api_tests/ask.py --no-think --video assets/digits_1234.mp4 "视频里依次出现了哪些数字？"
python3 scripts/api_tests/ask.py --no-think --audio assets/audio/jfk.wav "Transcribe this audio exactly."
```

它会打印 `思考`、`回答`、`[结束原因]`、`[用量]`、`[首字时间]`。

快速检查：`bash scripts/smoke_test.sh` 发 10 个请求，预期 `10/10 HTTP 200`（1M 服务上用 `env SMOKE_MAX_TOKENS=128`）。

## 步骤 6 基准测试

```bash
vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model mimo-v2.6-flash \
  --tokenizer /mnt/nvme/models/MiMo-V2.6-Flash-MOPD-fp8-mm-audio --trust-remote-code \
  --dataset-name random --random-input-len 512 --random-output-len 128 --ignore-eos \
  --max-concurrency 1 --num-prompts 8
```

`scripts/bench.sh c1` 包装了同一条命令，外加 2 个预热请求。推荐配置的结果：

| 测量项 | 结果 | 条件 |
|---|---|---|
| 短请求，并发 1 | TTFT 198.3 ms，TPOT 47.15 ms | 输入 512 / 输出 128，8 个请求 |
| 长提示，第一次请求 | 940K：204.2 s（每 4096 token 一段 0.89 s）；30K：3.83–3.88 s | 大海捞针提示，一次一个请求 |
| 长提示，命中前缀缓存 | 940K：2.89 s；30K：0.25 s | 同一提示再发一次；回答相同 |
| 约 30K 上下文时的 decode | 47.5 ms/token | — |
| 720p 视频，2 / 3 / 4 分钟 | 每段都是 14,211 个输入 token，首个 token 3.3–5.2 s | 关闭思考 |
| 运行时每核 HBM（最忙的核） | 20.13 / 24 GiB | neuron-monitor |

上表是同一套编译产物用 `scripts/serve.sh`、把插件开关写成环境变量启动时测的（代码是插件默认值改成这些值之前的版本）。
之后按步骤 4 的命令原样重测：TTFT 198.8 ms（中位数）、TPOT 47.46 ms，940K 首个 token 204.5–204.8 s，命中前缀缓存
2.82 s，30K 3.87 s，HBM 20.13 GiB，图片 / 视频 / 音频回归 8/8。
32K 上下文变体（32K）：TTFT 125.9 ms（中位数）、TPOT 38.35 ms，30K 大海捞针 3/3、首个 token 3.26 s，回归 8/8。
文件见 [results/README.md](results/README.md)。

1M 配置一次只能跑一个请求（KV 池只放得下一条 1M 序列），更高并发会排队。全部数字、更早的配置和原始文件：
[results/bench_summary.md](results/bench_summary.md)。

## 步骤 7 精度（可选）

- **logit 校验**：在 FP8 模型的 3 层裁剪版（原始第 0、1、5 层）上，用相同的切分和内核在设备上跑，对照 GPU 算法的 FP32 模拟，
  按 vllm-neuron 的 `multi_prompt_logit_validation` 判据：sigma 比 0.907（≤ 1 为通过）——**通过**。
- **GSM8K 98.0 %，MMLU 80.5 %**（各 200 题，0-shot，关闭思考，贪心）。与 BF16 权重相比：分别有 199/200 和 197/200 题答案相同。
- 视觉编码器对照 Hugging Face：余弦相似度 0.998–0.999；音频编码器：0.978（HF 自己的 BF16：0.979）。

用 `scripts/api_tests/task_eval.py` 重跑任务评测；方法、条件和全部数字：
[results/accuracy_summary.md](results/accuracy_summary.md)、[docs/development.md](docs/development.md#accuracy)。

## 已知限制

详情和原因：[docs/limitations.md](docs/limitations.md)。

- 只在 Neuron SDK 2.32.0 / vllm-neuron 0.24.0.1.1.0 + XLA 后端上验证过。
- 1M 配置一次只跑一个请求。
- 长提示的第一次请求慢：940K token 约 204 s（命中前缀缓存只要几秒）。
- 单张图片最大约 1024 × 1024（1,024 个视觉 token；更大的会缩小）。
- 视频上限是 **16,384 个视觉 token**，由分辨率决定，不由时长决定：2–4 分钟的 720p 视频都能放下；长视频抽帧更稀疏
  （4 分钟视频约每 7.5 s 一帧）。1080p 及以上没测。
- 每个请求最多 **16 个编码块**（每张图或每 1,024 token 的视频组 = 1 块，每段音频 = 1 块）：长 720p 视频（16 块）不能和图片或音频
  放在同一个请求里（HTTP 400）。总数放得下时，图片 + 视频 + 音频 + 文本可以在一个请求里；模型有时会混淆不同模态的内容。
- 音频：WAV，装了 soundfile 后还支持 FLAC / MP3 / Ogg；最长 32 s，每个请求一段。
- 开启思考时，模型有时漏掉思考结束标记，整段回复都落在 `reasoning` 里。
- 没有推测解码、FP8 KV 缓存或 DP（数据并行）注意力。
- 第一次编译约 1 小时；权重和编译缓存都在实例存储上。

## 后续计划

详情和估算：[docs/roadmap.md](docs/roadmap.md)。

1. **迁移到 Neuron SDK 2.33 + vllm-neuron 0.29** 和 PyTorch 原生后端（约 1 周）。
2. **推测解码**：权重自带 3 层 MTP（多 token 预测）草稿模型和 5 层 DFlash 草稿模型。vllm-neuron 0.29 已经有 DFlash 框架；MiMo
   还需要目标模型侧的支持（辅助隐藏状态、在 decode 图里一次验证 k+1 个 token）。预计迁移后 2–4 天，没有实测。
3. 更快的长 prefill（现在最大的部分是全局注意力前缀内核）、用 DP 注意力提高并发、按 KV 容量准入、更大的单张图片、编译缓存放 S3、
   把插件合入上游。

## 故障排查

更多情况：[docs/troubleshooting.md](docs/troubleshooting.md)。

- **20 分钟后还没就绪**——第一次启动要编译约 1 小时；看 `serve` tmux 里的日志。
- **编译期间服务或编译器消失**——主机内存不足；跑 `scripts/mem_guard.sh`。
- **`NCC_EOOM002` / decode 图超过 24 GB HBM**——缺 `--num-gpu-blocks-override`。
- **HTTP 400 `this request needs N vision/audio encoder blocks ...`**——一个请求超过 16 块；把内容分开发，或降低视频分辨率。
- **`file does not start with RIFF id`**——服务的 `PYTHONPATH` 里没有 soundfile（MP3 / FLAC / Ogg）。
- **`file://` 媒体被拒**——把文件放到 `--allowed-local-media-path` 下，或改用 base64。
- **只有思考没有回答**——调大 `max_tokens` 或关闭思考。
- **重启后 `/mnt/nvme` 是空的**——重新组装 RAID（停止 / 终止实例后数据已经没了）。

## 相关资料

- [Tutorial: Deploy Llama 3.3 70B (FP8) with vLLM Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/tutorials/tutorial-llama3-70b.html)（本教程参照的格式）
- [Hugging Face 上的 XiaomiMiMo/MiMo-V2.6-Flash-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD)
- [vllm-project/vllm-neuron](https://github.com/vllm-project/vllm-neuron)
- [Neuron 发布说明](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/index.html)
- [用 Autoport skill 把 GPU 兼容模型移植到 vLLM Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-agentic-development/developer_guides/neuron-framework-autoport-vllm-neuron.html)
  （本项目没有用；它大致覆盖第一次 BF16 移植的范围）
- 本仓库内：[设计说明](docs/design-notes.md)、[配置](docs/configuration.md)、[GPU 对齐](docs/gpu-alignment.md)、
  [开发和测试工具](docs/development.md)、[结果](results/README.md)、[发布前检查](docs/release-checklist.md)。

## 许可证

Apache-2.0，见 [LICENSE](LICENSE)。模型权重由小米以 MIT 许可发布，不包含在本仓库中。第三方来源列在 [NOTICE](NOTICE) 里。
