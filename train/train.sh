#!/usr/bin/env bash
# Run LoRA SFT on the SciDER trajectory dataset using LlamaFactory.
#
# LF refuses CLI overrides when launched with a YAML config, so this script
# materializes a *temporary* YAML (in /tmp) with env-var overrides applied,
# then hands it to llamafactory-cli.
#
# Examples:
#   bash train/train.sh                                            # defaults
#   CUDA_VISIBLE_DEVICES=0,1 bash train/train.sh
#   OUTPUT_DIR=/scratch/me/scider_lora bash train/train.sh
#   NUM_TRAIN_EPOCHS=2 LEARNING_RATE=5e-5 bash train/train.sh
#   CONFIG=examples/train_lora/qwen3_lora_sft.yaml bash train/train.sh

set -euo pipefail

# --- Resolve paths relative to this script ------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
LF_DIR="${SCRIPT_DIR}/LlamaFactory"

# --- Defaults (env override) -------------------------------------------------
CONFIG="${CONFIG:-examples/train_lora/qwen3_6_27b_scider_lora_sft.yaml}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS="$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)"

# Optional YAML overrides via env vars. Add more here as needed.
#   key in YAML       env var to read
declare -A OVERRIDES=(
    ["output_dir"]="OUTPUT_DIR"
    ["num_train_epochs"]="NUM_TRAIN_EPOCHS"
    ["learning_rate"]="LEARNING_RATE"
    ["per_device_train_batch_size"]="PER_DEVICE_BATCH_SIZE"
    ["gradient_accumulation_steps"]="GRADIENT_ACCUMULATION_STEPS"
    ["cutoff_len"]="CUTOFF_LEN"
    ["lora_rank"]="LORA_RANK"
    ["lora_alpha"]="LORA_ALPHA"
    ["max_steps"]="MAX_STEPS"
    ["max_samples"]="MAX_SAMPLES"
    ["report_to"]="REPORT_TO"
    ["run_name"]="WANDB_RUN_NAME"
)

# --- Sanity checks -----------------------------------------------------------
if [[ ! -d "${LF_DIR}" ]]; then
    echo "LlamaFactory dir not found: ${LF_DIR}" >&2
    exit 2
fi
if [[ ! -f "${LF_DIR}/${CONFIG}" ]]; then
    echo "Config not found: ${LF_DIR}/${CONFIG}" >&2
    exit 2
fi
if ! command -v llamafactory-cli >/dev/null 2>&1; then
    echo "llamafactory-cli not on PATH. From train/: 'uv sync' and re-source .venv/bin/activate." >&2
    exit 2
fi

# --- Materialize an effective YAML with overrides ----------------------------
TMP_YAML="$(mktemp -t scider_lora_XXXXXX.yaml)"
trap 'rm -f "${TMP_YAML}"' EXIT
cp "${LF_DIR}/${CONFIG}" "${TMP_YAML}"

apply_override() {
    local key="$1"
    local value="$2"
    # Replace `key: ...` (anywhere on a line, ignoring trailing comments).
    if grep -qE "^[[:space:]]*${key}:" "${TMP_YAML}"; then
        sed -i -E "s|^([[:space:]]*${key}:)[[:space:]].*|\1 ${value}|" "${TMP_YAML}"
    else
        printf "\n%s: %s\n" "${key}" "${value}" >> "${TMP_YAML}"
    fi
    echo "[train.sh]   override ${key} = ${value}"
}

echo "[train.sh] LF dir:  ${LF_DIR}"
echo "[train.sh] config:  ${CONFIG}  →  ${TMP_YAML}"
echo "[train.sh] GPUs:    ${CUDA_VISIBLE_DEVICES} (count: ${NUM_GPUS})"

for yaml_key in "${!OVERRIDES[@]}"; do
    env_var="${OVERRIDES[$yaml_key]}"
    if [[ -n "${!env_var:-}" ]]; then
        apply_override "${yaml_key}" "${!env_var}"
    fi
done

# --- Launch ------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES
echo "[train.sh] launching llamafactory-cli train ..."

cd "${LF_DIR}"
llamafactory-cli train "${TMP_YAML}"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}
RUN_NAME="${WANDB_RUN_NAME:-$(basename "${CONFIG}" .yaml)}"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
  apprise -b "OpenSciDER (${RUN_NAME}) Succeed" || true
else
  apprise -b "OpenSciDER (${RUN_NAME}) Failed (exit code: $TRAIN_EXIT_CODE)" || true
fi
