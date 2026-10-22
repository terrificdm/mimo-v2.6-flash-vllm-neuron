#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Download full MiMo-V2.6-Flash-MOPD weights to NVMe RAID; retries up to 5 times.
# Exit code: 0 when the download is complete, 1 after 5 failed attempts. Run it as a background task:
#   bash scripts/start_bg.sh download bash scripts/download_weights.sh   (log and status are written by run_bg.sh;
#   check with: bash scripts/check_bg.sh download)
# Path env (defaults): MIMO_VENV=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0,
#   HF_HOME=/mnt/nvme/hf-cache, MIMO_MODELS=/mnt/nvme/models (DST=$MIMO_MODELS/MiMo-V2.6-Flash-MOPD).
#   Gated/private access needs HF_TOKEN in the environment.
source "${MIMO_VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0}/bin/activate"
export HF_HOME=${HF_HOME:-/mnt/nvme/hf-cache} HF_XET_HIGH_PERFORMANCE=1
DST=${DST:-${MIMO_MODELS:-/mnt/nvme/models}/MiMo-V2.6-Flash-MOPD}
for i in 1 2 3 4 5; do
  hf download XiaomiMiMo/MiMo-V2.6-Flash-MOPD --revision b12f5573350bb1704513fcf119fca48e498b0423 --local-dir $DST --max-workers 32 && { echo "DONE $(date -u +%FT%TZ) $(du -sh $DST | cut -f1)"; exit 0; }
  echo "retry $i $(date -u)"; sleep 20
done
echo "FAILED $(date -u +%FT%TZ) after 5 attempts"
exit 1
