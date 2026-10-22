#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# MiMo-V2.6-Flash-MOPD 的 vLLM OpenAI 兼容服务。
# 用法（实例上，经 run_bg.sh 在 tmux 会话 serve 里启动；环境变量一律用 env 前缀）：
#   start_bg.sh serve env -u VLLM_NEURON_CPU_MODE -u NKI_SIMULATOR MIMO_DECODE_MOE_IMPL=<impl> \
#     MODEL_DIR=$MIMO_MODELS/MiMo-V2.6-Flash-MOPD-bf16 bash scripts/serve.sh
# Paths (env, with defaults): MIMO_VENV=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0
#   MIMO_MODELS=/mnt/nvme/models  MIMO_WORK=$HOME/mimo-work  MIMO_MM_ASSETS=$MIMO_WORK/mm_assets
#   VLLM_CACHE_ROOT=/mnt/nvme/vllm-cache
# serve.sh 是可选的便利脚本：它只是拼出一条 vllm serve 命令（SERVE_DRY_RUN=1 打印），部署时直接写那条命令（README Step 4）。
# 插件开关（下面标“只透传并打印”的 MIMO_*）不设时由插件取默认值 = 已验证的推荐配置
# （plugin/mimo_v2_neuron/switches.py；VLLM_NEURON_CPU_MODE=1 时仍是旧默认值），设了用于切回旧实现排障。
# 本脚本自己的开关（MIMO_CTX、MIMO_MM、MIMO_API 等）只决定命令行参数。
# 开关（全部不设时 vllm 命令为基础配置：4096 上下文、BF16、纯文本，插件开关按上面的默认值；取值不合法 exit 2）：
#   MIMO_CTX=4096|2048|32768|131072|524288|1048576   上下文长度（2048 = 编译超时时的回退档，buckets [1024, 2048]；
#                        ≥32768 时 --max-num-batched-tokens 4096 + 分段 prefill kv_segment_size_buckets [4096]）
#   MIMO_KV_SEG=4096|2048|1024|512   ≥32768 时的分段长度（默认 4096）：--max-num-batched-tokens、
#                        num_batched_tokens_buckets 与 kv_segment_size_buckets 都取该值（注意力暂存 ∝ 段长，docs/design-notes.md#segmented-prefill）
#   MIMO_KV_BLOCKS=<n>|off   --num-gpu-blocks-override（默认 4096→4096、32K→8192、128K→12288、512K→20480、1M→36864、
#                        2048→不加）；追加在命令末尾；SERVE_EXTRA 里已有时不重复
#   MIMO_API=0|1         1：--reasoning-parser $MIMO_REASONING_PARSER --tool-call-parser mimo --enable-auto-tool-choice；
#                        再加 MIMO_SO=1（默认）时 --no-async-scheduling + neuron_config enable_structured_outputs
#   MIMO_REASONING_PARSER=mimo|qwen3   MIMO_SO=0|1
#   MIMO_GENCFG=auto|vllm  --generation-config（MIMO_API=1 时默认 vllm；MIMO_API=0 且未设时不加）
#   MIMO_SAMPLING=device|cpu   cpu：neuron_config "on_device_sampling_config": null
#   MIMO_KV_DTYPE=bf16|fp8     fp8：--kv-cache-dtype fp8（MIMO_KV_SCALES 经环境透传给插件）
#   MIMO_MM=0|1          1：模型目录默认换成 -bf16-mm，加多模态参数与 --additional-config 顶层 vision_neuron_config
#                        （MIMO_MM_PROCESSOR_KWARGS 可覆盖 --mm-processor-kwargs 的 JSON；MIMO_MM_CACHE_BLOCKS 为
#                        vision_neuron_config.encoder_cache_num_blocks，默认 65：自动推导按 token 数算块，每个图片
#                        独占一块，第 5 张不同的图就会让 worker 报 Encoder cache full，docs/design-notes.md#encoder-cache-blocks）
#   MIMO_MM_VISION_BUCKETS=a,b,...  （需 MIMO_MM=1）vision_neuron_config.num_vision_tokens_buckets（patch 数，4096 的
#                        倍数，最大 ≤ 65536）。不设时为 [4096]，每个图片 / 视频 1 块、最多 1024 token，插件不做
#                        按块数的检查；设了以后插件把视频编码预算报成 ≥ max/4、在 API 进程按块计划检查每个请求（合计
#                        ≤ max/4096 块，超了返回 400）。直接写 vllm serve 时插件从 --additional-config 读这个档位
#                        列表，不用设环境变量。4096,65536 = 每个视频最多 16384 视觉 token（3 分钟 720p
#                        默认 32 帧是 14080）；docs/design-notes.md#vision-bucket
#   MIMO_AUDIO=0|1       1（需 MIMO_MM=1）：--limit-mm-per-prompt 的 audio 改为 1（模型目录要有 audio_config，
#                        make_mm_checkpoint.py --audio）
#   MIMO_ATTN_PREFILL_KEY_BLOCK / MIMO_ATTN_DECODE_KEY_BLOCK   只透传并打印（插件读取）
#   MIMO_DECODE_CTX_BUCKETS=a,b,...   neuron_config "decode_context_length_buckets": [a, b, ...]
#   SERVE_DRY_RUN=1      只打印环境变量取值和完整 vllm 命令，然后 exit 0
#   SERVE_VENV_ACTIVATE=0  跳过 venv 激活（只给没有 venv 的 dry-run 测试用）
#   SERVE_TP / SERVE_PORT / SERVE_EXTRA   tiny 演练、回退方案，或两个独立副本（未验证；SERVE_TP=32，端口 8001/8002，
#                        命令前加 NEURON_VISIBLE_DEVICES=0-31 / 32-63；默认 64 / 8000 / 空）
#   MIMO_MAX_SEQS=<n>    --max-num-seqs（默认 4 = 基础配置）；>4 时 num_seqs_buckets 为 [1, 4, n]（两个独立副本时用，未验证）
#   MIMO_PROFILE_DIR=<目录>  --additional-config 加 neuron_profiler（rank 0 的 device + system 轨迹写到该目录），
#                        并加 --profiler-config（/start_profile、/stop_profile 端点）；不设时命令不变；
#                        MIMO_PROFILE_ACTIVITIES='"system_profile"' 只抓 system trace（不占 profiler HBM 池）
#   MIMO_KV_WRITE=index_put|nki / MIMO_KV_GATHER=xla|nki   只透传并打印（插件读取；nki = 原地 scatter 写 KV（docs/design-notes.md#kv-write）、
#                        间接 DMA 读块；插件默认 KV_WRITE=nki、KV_GATHER=xla）
#   MIMO_CC_MODULAR_MAC=<n>|none|off  只透传并打印（插件改 neuronx-cc 的 --modular-flow-mac-threshold=10；docs/design-notes.md#kv-write；
#                        插件默认 100000000000000，off = 不改）
#   MIMO_CC_MODULAR_MAC_MAXTOK=<n>|off 只透传并打印（上一项只用于 token 数 ≤ n 的图，即 decode 图；docs/design-notes.md#kv-write；默认 64）
#   MIMO_PREFILL_BUCKETS=512,4096|off 只透传并打印（分段 prefill 时再加小于段长的 prefill 档，最后一个必须等于段长；
#                        插件默认按段长推导：段长 > 512 且 < 上下文长度时为 512,<段长>；off = 不加）
#   MIMO_KV_CONCURRENCY=<n> 只透传并打印（替换调度器按 max_model_len 算的并发上限，1M 配置下是 1；测试用，docs/design-notes.md#concurrency）
#   MIMO_VIT_DP_GROUP=0|1|world 只透传并打印（world = 视觉按块切分，用整个 TP 组 all-gather；docs/design-notes.md#vision-dp-group；插件默认 world）
#   MIMO_PREFILL_ATTN=torch|nki  只透传并打印（nki = 分段 prefill 的 GA 前缀注意力用 NKI flash 核；docs/design-notes.md#segmented-prefill；插件默认 nki）
#   MIMO_MOE_CTE_IMPL=shard_on_i|hybrid  只透传并打印（FP8 row 专家的 prefill；hybrid = nkilib shard_on_i_hybrid，
#                        只按非空块数循环，docs/design-notes.md#moe-hybrid；插件默认 hybrid）
#   MIMO_PREFILL_KIDX=gather|arith  只透传并打印（arith = GA 前缀核的 key 行号用广播算术代替 1M 元素 index_select，
#                        docs/design-notes.md#moe-hybrid；插件默认 arith）
#   MIMO_FP8_IMPL=emul|nki / MIMO_FP8_EXPERT_IMPL=expand|row / MIMO_KV_VNOPAD=0|1  只透传并打印（FP8 权重的线性层 /
#                        专家实现、V 不补零；插件默认 nki / row / 1）
#   MIMO_PREFIX_CACHE=0|1  1 = --enable-prefix-caching 代替 --no-enable-prefix-caching（docs/limitations.md；vllm-neuron 要求分段
#                        prefill，所以只允许 MIMO_CTX >= 32768）；默认 0，命令不变
#   多模态时 vision_neuron_config 的 dp_size = SERVE_TP / 8（tp_size 固定 8；SERVE_TP=64 时与原来相同）
#   SERVE_ALLOW_CPU=1    只用于 tiny CPU 演练：unset 之后重新 export VLLM_NEURON_CPU_MODE=1
#   MIMO_PYDEPS=<目录>|off  venv 外附加包目录（默认 $MIMO_WORK/pydeps，存在时追加到 PYTHONPATH 末尾；docs/design-notes.md#audio-input）
# 不用 set -e；退出码直接传给 run_bg.sh（exec）。
set -u
MIMO_WORK=${MIMO_WORK:-$HOME/mimo-work}
DEFAULT_MODEL_DIR=${MIMO_MODELS:-/mnt/nvme/models}/MiMo-V2.6-Flash-MOPD-bf16
MODEL_DIR=${MODEL_DIR:-$DEFAULT_MODEL_DIR}
if [ "${SERVE_VENV_ACTIVATE:-1}" != 0 ]; then
  source "${MIMO_VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0}/bin/activate" || exit 10
fi

# 正式服务不能继承 CPU 模式 / NKI 模拟器 / 关闭 NKI 核的开关
unset VLLM_NEURON_CPU_MODE NKI_SIMULATOR VLLM_NEURON_DISABLE_NKI_KERNELS
if [ "${SERVE_ALLOW_CPU:-0}" = 1 ]; then export VLLM_NEURON_CPU_MODE=1; fi

# venv 外装的纯附加包（docs/design-notes.md#audio-input）（soundfile 0.14.0 + 自带 libsndfile 1.2.2，MP3/FLAC/Ogg 音频解码）。
# 追加在 PYTHONPATH 末尾；目录里只有 venv 没有的顶层名字，不遮蔽 venv 已装的包。MIMO_PYDEPS=off 关闭。
MIMO_PYDEPS=${MIMO_PYDEPS:-$MIMO_WORK/pydeps}
if [ "$MIMO_PYDEPS" != off ] && [ -d "$MIMO_PYDEPS" ]; then
  export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$MIMO_PYDEPS"
fi

die() { echo "serve.sh: $*" >&2; exit 2; }
check() {  # check <name> <value> <allowed...>
  local n=$1 v=$2; shift 2
  for a in "$@"; do [ "$v" = "$a" ] && return 0; done
  die "$n must be one of: $*; got '$v'"
}

MIMO_CTX=${MIMO_CTX:-4096}
case "$MIMO_CTX" in
  4096) BUCKETS='[1024, 4096]'; MNBT=4096; KVB_DEF=4096 ;;
  2048) BUCKETS='[1024, 2048]'; MNBT=2048; KVB_DEF= ;;
  32768) BUCKETS='[4096]'; MNBT=4096; KVB_DEF=8192 ;;
  131072) BUCKETS='[4096]'; MNBT=4096; KVB_DEF=12288 ;;
  524288) BUCKETS='[4096]'; MNBT=4096; KVB_DEF=20480 ;;
  1048576) BUCKETS='[4096]'; MNBT=4096; KVB_DEF=36864 ;;
  *) die "MIMO_CTX must be one of: 4096 2048 32768 131072 524288 1048576; got '$MIMO_CTX'" ;;
esac
MIMO_PREFIX_CACHE=${MIMO_PREFIX_CACHE:-0};  check MIMO_PREFIX_CACHE "$MIMO_PREFIX_CACHE" 0 1
MIMO_KV_SEG=${MIMO_KV_SEG:-4096};           check MIMO_KV_SEG "$MIMO_KV_SEG" 4096 2048 1024 512
if [ "$MIMO_CTX" -ge 32768 ]; then BUCKETS="[$MIMO_KV_SEG]"; MNBT=$MIMO_KV_SEG
elif [ "$MIMO_KV_SEG" != 4096 ]; then die "MIMO_KV_SEG only applies to MIMO_CTX >= 32768; got MIMO_CTX=$MIMO_CTX"; fi
if [ "$MIMO_PREFIX_CACHE" = 1 ] && [ "$MIMO_CTX" -lt 32768 ]; then die "MIMO_PREFIX_CACHE=1 needs segmented prefill (MIMO_CTX >= 32768)"; fi
MIMO_API=${MIMO_API:-0};                     check MIMO_API "$MIMO_API" 0 1
MIMO_REASONING_PARSER=${MIMO_REASONING_PARSER:-mimo}; check MIMO_REASONING_PARSER "$MIMO_REASONING_PARSER" mimo qwen3
MIMO_SO=${MIMO_SO:-1};                       check MIMO_SO "$MIMO_SO" 0 1
if [ -z "${MIMO_GENCFG:-}" ] && [ "$MIMO_API" = 1 ]; then MIMO_GENCFG=vllm; fi
MIMO_GENCFG=${MIMO_GENCFG:-}
[ -z "$MIMO_GENCFG" ] || check MIMO_GENCFG "$MIMO_GENCFG" auto vllm
MIMO_SAMPLING=${MIMO_SAMPLING:-device};      check MIMO_SAMPLING "$MIMO_SAMPLING" device cpu
MIMO_KV_DTYPE=${MIMO_KV_DTYPE:-bf16};        check MIMO_KV_DTYPE "$MIMO_KV_DTYPE" bf16 fp8
MIMO_MM=${MIMO_MM:-0};                       check MIMO_MM "$MIMO_MM" 0 1
MIMO_AUDIO=${MIMO_AUDIO:-0};                 check MIMO_AUDIO "$MIMO_AUDIO" 0 1
[ "$MIMO_AUDIO" = 0 ] || [ "$MIMO_MM" = 1 ] || die "MIMO_AUDIO=1 needs MIMO_MM=1"
MIMO_KV_BLOCKS=${MIMO_KV_BLOCKS:-$KVB_DEF}
case "$MIMO_KV_BLOCKS" in
  ''|off) MIMO_KV_BLOCKS= ;;
  *[!0-9]*|0*) die "MIMO_KV_BLOCKS must be a positive integer or 'off'; got '$MIMO_KV_BLOCKS'" ;;
esac
MIMO_DECODE_CTX_BUCKETS=${MIMO_DECODE_CTX_BUCKETS:-}
if [ -n "$MIMO_DECODE_CTX_BUCKETS" ]; then
  [[ "$MIMO_DECODE_CTX_BUCKETS" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] \
    || die "MIMO_DECODE_CTX_BUCKETS must be comma-separated positive integers; got '$MIMO_DECODE_CTX_BUCKETS'"
fi
for v in MIMO_ATTN_PREFILL_KEY_BLOCK MIMO_ATTN_DECODE_KEY_BLOCK; do
  if [ -n "${!v:-}" ] && ! [[ "${!v}" =~ ^[1-9][0-9]*$ ]]; then die "$v must be a positive integer; got '${!v}'"; fi
done
if [ "$MIMO_MM" = 1 ] && [ "$MODEL_DIR" = "$DEFAULT_MODEL_DIR" ]; then MODEL_DIR=$DEFAULT_MODEL_DIR-mm; fi
MM_ASSETS=${MIMO_MM_ASSETS:-$MIMO_WORK/mm_assets}
MIMO_MM_PROCESSOR_KWARGS=${MIMO_MM_PROCESSOR_KWARGS:-'{"max_pixels": 1048576}'}
MIMO_MM_CACHE_BLOCKS=${MIMO_MM_CACHE_BLOCKS:-65}
MIMO_MAX_SEQS=${MIMO_MAX_SEQS:-4}
[[ "$MIMO_MAX_SEQS" =~ ^[1-9][0-9]*$ ]] && [ "$MIMO_MAX_SEQS" -ge 4 ] || die "MIMO_MAX_SEQS must be an integer >= 4; got '$MIMO_MAX_SEQS'"
[[ "$MIMO_MM_CACHE_BLOCKS" =~ ^[1-9][0-9]*$ ]] || die "MIMO_MM_CACHE_BLOCKS must be a positive integer; got '$MIMO_MM_CACHE_BLOCKS'"
VISION_BUCKETS='[4096]'
if [ -n "${MIMO_MM_VISION_BUCKETS:-}" ]; then
  [ "$MIMO_MM" = 1 ] || die "MIMO_MM_VISION_BUCKETS needs MIMO_MM=1"
  [[ "$MIMO_MM_VISION_BUCKETS" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] \
    || die "MIMO_MM_VISION_BUCKETS must be comma-separated positive integers; got '$MIMO_MM_VISION_BUCKETS'"
  prev=0
  for b in ${MIMO_MM_VISION_BUCKETS//,/ }; do
    [ $((b % 4096)) = 0 ] && [ "$b" -le 65536 ] && [ "$b" -gt "$prev" ] \
      || die "MIMO_MM_VISION_BUCKETS must be increasing multiples of 4096, at most 65536; got '$MIMO_MM_VISION_BUCKETS'"
    prev=$b
  done
  VISION_BUCKETS="[${MIMO_MM_VISION_BUCKETS//,/, }]"
  export MIMO_MM_VISION_BUCKETS   # the plugin reads it in the API process and the workers
fi

VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/mnt/nvme/vllm-cache}
mkdir -p "$VLLM_CACHE_ROOT"
export VLLM_NEURON_COMPILATION_TIMEOUT=7200          # 单个 NEFF 编译上限（libtorch_neuronx_lite envs 别名）
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200       # vLLM 默认 300 s，首次 forward 触发编译会超时
export VLLM_ENGINE_READY_TIMEOUT_S=14400             # vLLM 默认 600 s；加载 618 GB + 编译 4 个 NEFF
if [ "$MIMO_CTX" -ge 524288 ]; then export VLLM_ENGINE_READY_TIMEOUT_S=21600; fi   # 512K/1M 编译更久
export VLLM_CACHE_ROOT                              # 编译缓存 → $VLLM_CACHE_ROOT/neuron/compile_cache（vllm_neuron/envs.py:380）
export MIMO_DECODE_MOE_IMPL=${MIMO_DECODE_MOE_IMPL:-tkg}   # 取 L3 验证通过时的值
SERVE_TP=${SERVE_TP:-64}
SERVE_PORT=${SERVE_PORT:-8000}
read -r -a EXTRA <<< "${SERVE_EXTRA:-}"

echo "serve.sh: $(date -u +%FT%TZ) effective environment:"
for v in VLLM_NEURON_CPU_MODE NKI_SIMULATOR VLLM_NEURON_DISABLE_NKI_KERNELS MIMO_DECODE_MOE_IMPL VLLM_CACHE_ROOT \
         VLLM_NEURON_COMPILATION_TIMEOUT VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS VLLM_ENGINE_READY_TIMEOUT_S \
         MODEL_DIR MIMO_CTX MIMO_KV_SEG SERVE_TP SERVE_PORT \
         MIMO_API MIMO_REASONING_PARSER MIMO_SO MIMO_GENCFG MIMO_SAMPLING MIMO_KV_BLOCKS MIMO_KV_DTYPE MIMO_KV_SCALES \
         MIMO_MM MIMO_AUDIO MIMO_ATTN_PREFILL_KEY_BLOCK MIMO_ATTN_DECODE_KEY_BLOCK MIMO_DECODE_CTX_BUCKETS \
         MIMO_MAX_SEQS NEURON_VISIBLE_DEVICES MIMO_FP8_EXPERT_IMPL MIMO_FP8_IMPL MIMO_KV_WRITE MIMO_KV_GATHER MIMO_CC_MODULAR_MAC MIMO_CC_MODULAR_MAC_MAXTOK MIMO_VIT_DP_GROUP MIMO_PREFILL_BUCKETS MIMO_KV_CONCURRENCY MIMO_PREFILL_ATTN MIMO_MOE_CTE_IMPL MIMO_PREFILL_KIDX MIMO_PREFIX_CACHE MIMO_PYDEPS PYTHONPATH MIMO_MM_VISION_BUCKETS; do
  if [ -n "${!v+x}" ]; then echo "  $v=${!v}"; else echo "  $v=<unset>"; fi
done

# neuron_config：基础配置的两个键在前，其他开关新增的键依次追加在后面
SEQB='[1, 4]'; if [ "$MIMO_MAX_SEQS" -gt 4 ]; then SEQB="[1, 4, $MIMO_MAX_SEQS]"; fi
NC="\"num_batched_tokens_buckets\": $BUCKETS, \"num_seqs_buckets\": $SEQB"
if [ "$MIMO_CTX" -ge 32768 ]; then NC+=", \"kv_segment_size_buckets\": [$MIMO_KV_SEG]"; fi
if [ "$MIMO_API" = 1 ] && [ "$MIMO_SO" = 1 ]; then NC+=", \"enable_structured_outputs\": true"; fi
if [ "$MIMO_SAMPLING" = cpu ]; then NC+=", \"on_device_sampling_config\": null"; fi
if [ -n "$MIMO_DECODE_CTX_BUCKETS" ]; then NC+=", \"decode_context_length_buckets\": [${MIMO_DECODE_CTX_BUCKETS//,/, }]"; fi
ADD="{\"neuron_config\": {$NC}"
if [ "$MIMO_MM" = 1 ]; then
  ADD+=", \"vision_neuron_config\": {\"tp_size\": 8, \"dp_size\": $((SERVE_TP / 8)), \"num_vision_tokens_buckets\": $VISION_BUCKETS, \"vision_attention_block_size\": 4096, \"encoder_cache_num_blocks\": $MIMO_MM_CACHE_BLOCKS}"
fi
if [ -n "${MIMO_PROFILE_DIR:-}" ]; then
  ADD+=", \"neuron_profiler\": {\"output_dir\": \"$MIMO_PROFILE_DIR\", \"activities\": [${MIMO_PROFILE_ACTIVITIES:-\"device_profile\", \"system_profile\"}]}"
fi
ADD+="}"

cmd=(vllm serve "$MODEL_DIR"
  --served-model-name mimo-v2.6-flash --trust-remote-code
  --tensor-parallel-size "$SERVE_TP")
# vllm-neuron 只在 ep_degree > 1 时建 EP 组（tiny 模型演练中发现），TP=1 的 tiny 演练不开 EP
if [ "$SERVE_TP" -gt 1 ]; then cmd+=(--enable-expert-parallel); fi
cmd+=(--max-model-len "$MIMO_CTX" --max-num-batched-tokens "$MNBT" --max-num-seqs "$MIMO_MAX_SEQS"
  --block-size 32)
if [ "$MIMO_PREFIX_CACHE" = 1 ]; then cmd+=(--enable-prefix-caching); else cmd+=(--no-enable-prefix-caching); fi
cmd+=(--host 127.0.0.1 --port "$SERVE_PORT"
  --additional-config "$ADD")
if [ "$MIMO_API" = 1 ]; then
  cmd+=(--reasoning-parser "$MIMO_REASONING_PARSER" --tool-call-parser mimo --enable-auto-tool-choice)
  if [ "$MIMO_SO" = 1 ]; then cmd+=(--no-async-scheduling); fi
fi
if [ -n "$MIMO_GENCFG" ]; then cmd+=(--generation-config "$MIMO_GENCFG"); fi
if [ "$MIMO_KV_DTYPE" = fp8 ]; then cmd+=(--kv-cache-dtype fp8); fi
if [ "$MIMO_MM" = 1 ]; then
  cmd+=(--limit-mm-per-prompt "{\"image\": 2, \"video\": 1, \"audio\": $MIMO_AUDIO}"
    --allowed-local-media-path "$MM_ASSETS"
    --mm-processor-kwargs "$MIMO_MM_PROCESSOR_KWARGS")
fi
if [ -n "${MIMO_PROFILE_DIR:-}" ]; then cmd+=(--profiler-config '{"profiler": "cuda"}'); fi
if [ ${#EXTRA[@]} -gt 0 ]; then cmd+=("${EXTRA[@]}"); fi
if [ -n "$MIMO_KV_BLOCKS" ]; then
  have_kvb=0
  for a in "${EXTRA[@]+"${EXTRA[@]}"}"; do
    case "$a" in --num-gpu-blocks-override|--num-gpu-blocks-override=*) have_kvb=1 ;; esac
  done
  if [ "$have_kvb" = 1 ]; then
    echo "serve.sh: WARNING: SERVE_EXTRA already has --num-gpu-blocks-override; not adding MIMO_KV_BLOCKS=$MIMO_KV_BLOCKS"
  else
    cmd+=(--num-gpu-blocks-override "$MIMO_KV_BLOCKS")
  fi
fi

echo "serve.sh: command: $(printf '%q ' "${cmd[@]}")"
if [ "${SERVE_DRY_RUN:-0}" = 1 ]; then echo "serve.sh: SERVE_DRY_RUN=1, not starting"; exit 0; fi
exec "${cmd[@]}"
