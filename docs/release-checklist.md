# 发布前检查清单（PR / 发布前）

每次提 PR 或发布前，在仓库根目录逐项运行下面的命令，结果全部符合“通过标准”再提交。下面的“最近一次结果”是
2026-10-22 提交前的本地工作区上跑的；每一项都在仓库目录和一份 `cp -R` 到 `mktemp -d` 临时目录、只含本仓库的副本里各跑了一次，结果相同。

| # | 检查 | 通过标准 | 最近一次结果 |
|---|---|---|---|
| 1 | 文档路径校验 | `missing=0`，退出码 0 | 28 个 .md，检查 458 条路径引用，缺失 0 |
| 2 | 通用敏感模式扫描 | `total_findings=0`，退出码 0 | 10 类模式全部 0 命中 |
| 3 | 私有字面量扫描 | 0 命中 | 对私有字面量清单的扫描：0 命中（含二进制文件） |
| 4 | Python 语法（`py_compile`） | 0 个失败 | 98 个 .py，0 个失败；之后已清理 `__pycache__` |
| 5 | Shell 语法（`bash -n`） | 0 个失败 | 16 个 .sh，0 个失败 |
| 6 | 不含权重 / NEFF / HF 缓存 | 0 个匹配，最大文件远小于 1 MB | 0 个匹配；最大文件 528,044 字节；全仓 2.2 MB、200 个文件 |
| 7 | LICENSE、NOTICE | 都存在，LICENSE 是 Apache-2.0 | 都存在；LICENSE 为 Apache License 2.0 全文 |
| 8 | 自包含 | 隔离副本里各项全过，退出码 0 | 外部引用 0、提交号 0、开发过程用语 0；路径 458 / 0 缺失；py 98 / 0；sh 16 / 0；敏感模式 0 |
| 9 | 没有内部编号 | 两条 grep 都是 0 行 | `P-xx` / `D2-xx` 全仓 0 行；`plugin tools scripts tests` 里任务编号 0 行 |
| 10 | `download_weights.sh` 退出码（mock） | 成功 0、失败 1、不写自己的状态文件；配合 `run_bg.sh` / `check_bg.sh` 报 `DONE` / `FAILED rc=1` | 全部符合 |

## 1. 文档路径校验

```bash
python3 scripts/dev/check_doc_paths.py            # 只列缺失项
python3 scripts/dev/check_doc_paths.py --verbose  # 列出每条引用的解析方式和忽略原因
```

扫描所有 `.md` 里行内反引号和 Markdown 链接中的路径式引用（fenced 代码块不扫），反引号相对仓库根、链接相对所在文件目录
解析。忽略规则（URL、命令片段、占位符、实例上的路径、`~/` 路径、vllm-neuron 包内路径、HTTP 接口、仓库 ID、本地
生成的文件）写在脚本文件头。

最近一次结果：`checked=458 ignored=225 missing=0 files=28`，退出码 0。

## 2. 通用敏感模式扫描

```bash
bash scripts/dev/scan_sensitive.sh                # 每类模式的命中数
SHOW=1 bash scripts/dev/scan_sensitive.sh         # 同时打印命中位置
```

检查 `/Users/…` 路径、`/home/` 下除 `ubuntu` 以外的用户目录、EC2 实例 ID、ami / sg / subnet / vpc / cr 资源 ID、
12 位账号号码、`.pem` 密钥路径、私有 IPv4 与 `ip-10-…` 主机名、公网 IPv4、AWS 访问密钥。脚本自身不扫描（正则定义会
匹配自己），也不含任何真实敏感值。白名单逐条写了理由：回环 / 通配地址、4 段式的组件版本号、FP8 转换校验结果里的元素数
和字节数。

最近一次结果：

```text
users_path       0
home_non_ubuntu  0
instance_id      0
resource_id      0
account_id       0
pem_path         0
private_ipv4     0
private_host     0
public_ipv4      0
aws_access_key   0
total_findings=0
```

## 3. 私有字面量扫描

维护者用一份放在仓库外的脚本，按作者环境的私有字面量清单（本机路径、用户名、实例与网络信息、账号等）对全仓（排除
`.git`）做固定字符串匹配。清单本身是敏感信息，不随仓库发布，这里也不列出。

最近一次结果：对私有字面量清单的扫描：0 命中（文本行和二进制文件都是 0）。

## 4. Python 语法

```bash
find . -path ./.git -prune -o -name '*.py' -type f -print0 | xargs -0 -n1 python3 -m py_compile
find . -name __pycache__ -type d -not -path './.git/*' -prune -exec rm -rf {} +
```

最近一次结果：98 个 .py 文件，0 个失败；`__pycache__` 已清理（剩余 0 个）。

## 5. Shell 语法

```bash
find . -path ./.git -prune -o -name '*.sh' -type f -print0 | xargs -0 -n1 bash -n
```

最近一次结果：16 个 .sh 文件，0 个失败。

## 6. 不含权重、NEFF、HF 缓存

```bash
find . -path ./.git -prune -o -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.pth' \
  -o -name '*.gguf' -o -name '*.npy' -o -name '*.neff' -o -name '*.ntff' -o -name '*.parquet' -o -name '*.pem' \
  -o -name '.env' \) -print | wc -l
find . -path ./.git -prune -o -type d \( -iname '*hf*cache*' -o -name 'models--*' -o -name 'vllm-cache' \
  -o -name 'neuron-compile-cache' \) -print | wc -l
find . -path ./.git -prune -o -type f -print0 | xargs -0 ls -l | awk '{print $5, $9}' | sort -rn | head -10
du -sh .
```

最近一次结果：权重 / 编译产物 / 密钥类文件 0 个，缓存目录 0 个；全仓 2.2 MB、200 个文件（不含 `.git`）。最大的 10 个文件：

| 字节数 | 文件 |
|---:|---|
| 528,044 | `assets/audio/jfk.wav` |
| 120,044 | `assets/audio/beeps_3.wav` |
| 96,044 | `assets/audio/tone_440.wav` |
| 63,411 | `results/accuracy/l3_fp8_seg4k_torch_b1_vs_gpusim.json` |
| 63,392 | `results/accuracy/l3_fp8_final_b1_vs_gpusim.json` |
| 57,413 | `plugin/mimo_v2_neuron/model/mimo_v2/model.py` |
| 54,809 | `results/conversion/fp8_convert_check_fulldir.json` |
| 51,777 | `README.md` |
| 49,726 | `README.zh-CN.md` |
| 34,313 | `scripts/api_tests/test_params.py` |

`.gitignore` 另外按扩展名排除了权重（safetensors、pt 等）、编译产物（neff、ntff）、HF 缓存、编译缓存，以及由
`scripts/prepare_hf_reference.sh` 本地生成的 reference 目录。

## 7. LICENSE 与 NOTICE

```bash
ls -l LICENSE NOTICE && grep -m1 'Apache License' LICENSE && grep -m1 'Version 2.0' LICENSE
```

最近一次结果：`LICENSE`（Apache License, Version 2.0 全文，202 行）和 `NOTICE`（第三方来源与改编文件清单，81 行）都存在。

## 8. 自包含

```bash
bash scripts/dev/check_independence.sh
```

把仓库单独拷到一个临时目录（不带 `.git`），在那里依次检查：① 不引用仓库外的报告 / 计划 / 决策记录文件名、内部决策编号、
内部运维路径和其他仓库名（模式清单写在脚本里，脚本自身除外）；①b 没有 git 提交号（7–40 位十六进制、至少含一个 a–f 字母；
SHA-256 值、`SHA256SUMS` 文件、浮点数尾数和下载用的 HF 模型 revision 不算）和开发过程用语（开发记账文件名、开发
过程的编号、编排用语、配置版本后缀等，模式清单写在脚本里，不区分大小写）；② 文档路径校验；③ `py_compile`；④ `bash -n`；
⑤ 通用敏感模式扫描。代码注释和文档里的设计说明集中在 [design-notes.md](design-notes.md) 带锚点的条目里。

模式里也包括任务编号的格式（字母 T 加两位数字、可带一个小写字母，但不含 `T10`；评审意见和风险条目的编号；`T1`–`T10`
是工具调用套件的用例编号，允许），以后再有这类编号进入仓库会被这一项拦住。

最近一次结果：引用 `hits=0`；`hash_hits=0`；`stage_hits=0`；`checked=458 ignored=225 missing=0 files=28`；py 98 个 0 失败；sh 16 个 0 失败；
`total_findings=0`；`== result: PASS`。

## 9. 没有内部编号

```bash
grep -rnIE --exclude-dir=.git "P-[0-9]{2,3}|D2-[0-9]{2}" .
grep -rnIE "(^|[^A-Za-z0-9_])T(0[0-9]|1[1-9]|[2-9][0-9])[a-z]?([^A-Za-z0-9_]|$)|R3-#[0-9]|design R[0-9]|R[5]/R6" plugin tools scripts tests
```

代码注释里的说明要么就地写清楚，要么指向 [design-notes.md](design-notes.md) 的锚点（`docs/design-notes.md#<锚点>`）；
提交号和开发过程用语由第 8 项的 ①b 检查（全仓）。再检查一遍代码和文档里引用的锚点都存在：

```bash
for a in $(grep -rhoIE --exclude-dir=.git "design-notes\.md#[a-z0-9-]+" . | sed 's/.*#//' | sort -u); do
  grep -q "<a id=\"$a\">" docs/design-notes.md || grep -qi "^## ${a//-/ }" docs/design-notes.md || echo "MISSING $a"
done
```

最近一次结果：两条 grep 都是 0 行；引用的 28 个锚点全部存在（`validation-method` 是二级标题自动生成的锚点）。

## 10. `download_weights.sh` 退出码（mock）

`scripts/download_weights.sh` 不写自己的状态文件，下载完成退出 0、5 次都失败退出 1，所以可以用
`scripts/start_bg.sh download bash scripts/download_weights.sh` 运行、用 `scripts/check_bg.sh download` 查看。不联网的检查方法：
在临时目录里放一个假的 venv 激活脚本（把 mock 目录放到 `PATH` 最前面）、一个按 `MOCK_HF_RC` 返回退出码的假 `hf` 和一个
直接返回的假 `sleep`，再用 `MIMO_VENV`、`HF_HOME`、`MIMO_MODELS`、`MIMO_WORK` 指向临时目录运行：

```bash
T=$(mktemp -d); mkdir -p "$T/venv/bin" "$T/mock"
echo "export PATH=\"$T/mock:\$PATH\"" > "$T/venv/bin/activate"
printf '#!/usr/bin/env bash\nexit "${MOCK_HF_RC:-0}"\n' > "$T/mock/hf"; printf '#!/usr/bin/env bash\nexit 0\n' > "$T/mock/sleep"
chmod +x "$T/mock/hf" "$T/mock/sleep"
E="MIMO_VENV=$T/venv HF_HOME=$T/hf MIMO_MODELS=$T/models MIMO_WORK=$T/work"
env $E MOCK_HF_RC=0 bash scripts/download_weights.sh; echo "rc=$?"          # DONE ...; rc=0
env $E MOCK_HF_RC=1 bash scripts/download_weights.sh; echo "rc=$?"          # 5 x retry, FAILED ... after 5 attempts; rc=1
env $E MOCK_HF_RC=1 bash scripts/run_bg.sh dl bash scripts/download_weights.sh >/dev/null
env MIMO_WORK=$T/work bash scripts/check_bg.sh dl                         # dl FAILED FAILED rc=1 ...
rm -rf "$T"
```

最近一次结果：`bash -n` 通过；成功退出码 0、输出 `DONE`；失败退出码 1、5 行 `retry` 加 `FAILED ... after 5 attempts`；两种情况
都不写 `$MIMO_WORK/status/download.status`；经 `run_bg.sh` 跑完后 `check_bg.sh` 分别报 `DONE`（退出码 0）和 `FAILED rc=1`
（退出码 1）。
