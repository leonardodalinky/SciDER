#!/usr/bin/env bash
# Merge a trained LoRA adapter into the base model so the result loads with
# AutoModelForCausalLM.from_pretrained(...).
#
# Same env-var override pattern as train.sh: the script materializes a
# /tmp YAML (LF doesn't accept CLI overrides when launched with a config),
# applies overrides, then hands it to llamafactory-cli export.
#
# Examples:
#   ADAPTER_DIR=/path/to/saves/qwen3.5-9b-scider/lora/sft \
#   EXPORT_DIR=/path/to/scider_qwen3_5_9b_merged \
#       bash train/merge.sh
#
#   # Merge a specific checkpoint instead of the final adapter
#   ADAPTER_DIR=/path/to/saves/.../checkpoint-800 \
#   EXPORT_DIR=/path/to/scider_merged_ckpt800 \
#       bash train/merge.sh
#
#   # Use a different base model
#   MODEL_NAME=Qwen/Qwen3-8B ADAPTER_DIR=... EXPORT_DIR=... bash train/merge.sh
#
#   # GPU-accelerated merge (faster but needs free VRAM ≥ ~25 GB)
#   EXPORT_DEVICE=auto ADAPTER_DIR=... EXPORT_DIR=... bash train/merge.sh

set -euo pipefail

# --- Resolve paths relative to this script -----------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
LF_DIR="${SCRIPT_DIR}/LlamaFactory"

# --- Defaults (env override) -------------------------------------------------
CONFIG="${CONFIG:-examples/merge_lora/qwen3_6_27b_scider_lora_sft.yaml}"

# Required (no sane defaults)
ADAPTER_DIR="${ADAPTER_DIR:-}"
EXPORT_DIR="${EXPORT_DIR:-}"

# Optional YAML overrides via env vars.
declare -A OVERRIDES=(
    ["model_name_or_path"]="MODEL_NAME"
    ["adapter_name_or_path"]="ADAPTER_DIR"
    ["export_dir"]="EXPORT_DIR"
    ["export_size"]="EXPORT_SIZE"
    ["export_device"]="EXPORT_DEVICE"
    ["template"]="TEMPLATE"
)

# --- Sanity checks -----------------------------------------------------------
if [[ -z "${ADAPTER_DIR}" ]]; then
    echo "ADAPTER_DIR is required. Set it to your trained LoRA save dir, e.g.:" >&2
    echo "  ADAPTER_DIR=/path/saves/qwen3.5-9b-scider/lora/sft \\" >&2
    echo "  EXPORT_DIR=/path/scider_qwen3_5_9b_merged bash train/merge.sh" >&2
    exit 2
fi
if [[ -z "${EXPORT_DIR}" ]]; then
    echo "EXPORT_DIR is required. Set it to where the merged model should be written." >&2
    exit 2
fi
if [[ ! -d "${LF_DIR}" ]]; then
    echo "LlamaFactory dir not found: ${LF_DIR}" >&2
    exit 2
fi
if [[ ! -f "${LF_DIR}/${CONFIG}" ]]; then
    echo "Config not found: ${LF_DIR}/${CONFIG}" >&2
    exit 2
fi
if [[ ! -d "${ADAPTER_DIR}" ]]; then
    echo "ADAPTER_DIR does not exist: ${ADAPTER_DIR}" >&2
    exit 2
fi
if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
    echo "ADAPTER_DIR has no adapter_config.json: ${ADAPTER_DIR}" >&2
    echo "Did you point at the right LoRA save dir / checkpoint?" >&2
    exit 2
fi
if ! command -v llamafactory-cli >/dev/null 2>&1; then
    echo "llamafactory-cli not on PATH. From train/: 'uv sync' and re-source .venv/bin/activate." >&2
    exit 2
fi

# --- Materialize an effective YAML with overrides ----------------------------
TMP_YAML="$(mktemp -t scider_merge_XXXXXX.yaml)"
trap 'rm -f "${TMP_YAML}"' EXIT
cp "${LF_DIR}/${CONFIG}" "${TMP_YAML}"

apply_override() {
    local key="$1"
    local value="$2"
    if grep -qE "^[[:space:]]*${key}:" "${TMP_YAML}"; then
        sed -i -E "s|^([[:space:]]*${key}:)[[:space:]].*|\1 ${value}|" "${TMP_YAML}"
    else
        printf "\n%s: %s\n" "${key}" "${value}" >> "${TMP_YAML}"
    fi
    echo "[merge.sh]   override ${key} = ${value}"
}

echo "[merge.sh] LF dir:    ${LF_DIR}"
echo "[merge.sh] config:    ${CONFIG}  →  ${TMP_YAML}"
echo "[merge.sh] adapter:   ${ADAPTER_DIR}"
echo "[merge.sh] export to: ${EXPORT_DIR}"

for yaml_key in "${!OVERRIDES[@]}"; do
    env_var="${OVERRIDES[$yaml_key]}"
    if [[ -n "${!env_var:-}" ]]; then
        apply_override "${yaml_key}" "${!env_var}"
    fi
done

mkdir -p "${EXPORT_DIR}"

# --- Launch ------------------------------------------------------------------
echo "[merge.sh] launching llamafactory-cli export ..."

cd "${LF_DIR}"
llamafactory-cli export "${TMP_YAML}"

echo "[merge.sh] done. Merged model at: ${EXPORT_DIR}"
echo "[merge.sh] usage:"
echo "    from transformers import AutoModelForCausalLM, AutoTokenizer"
echo "    model = AutoModelForCausalLM.from_pretrained('${EXPORT_DIR}',"
echo "        torch_dtype='bfloat16', device_map='auto', trust_remote_code=True)"
