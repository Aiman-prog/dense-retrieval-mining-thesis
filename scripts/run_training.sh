#!/usr/bin/env bash

#SBATCH --job-name=dense_train
#SBATCH --partition=gpu-a100
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=training_%j.out
#SBATCH --error=training_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/dense-retrieval-mining-thesis

# --- Load modules ---
module purge
module load 2025
module load cuda/12.9
module load miniconda3/4.12.0
# Java is installed via conda (openjdk=11), no need to load module

# --- Activate Conda environment ---
eval "$(conda shell.bash hook)"
conda activate dense-retrieval

# --- Set up scratch space for model output (same as evaluation script) ---
SCRATCH_DIR="/scratch/${USER}/dense-retrieval"
mkdir -p "${SCRATCH_DIR}"
OUTPUT_DIR="${SCRATCH_DIR}/models"

# --- Set Hugging Face to offline mode (use cache only) ---
# Use scratch space for HuggingFace cache
SCRATCH_CACHE="/scratch/${USER}/caches"
mkdir -p "${SCRATCH_CACHE}/huggingface"
export HF_HOME="${SCRATCH_CACHE}/huggingface"
export HF_DATASETS_CACHE="${SCRATCH_CACHE}/huggingface"
export TRANSFORMERS_CACHE="${SCRATCH_CACHE}/huggingface"
export SENTENCE_TRANSFORMERS_HOME="${SCRATCH_CACHE}/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Debug: show cache location
echo "DEBUG: HF_HOME=${HF_HOME}"
echo "DEBUG: Checking if model exists in cache..."
ls -la "${SCRATCH_CACHE}/huggingface/" | head -10 || echo "Cache directory empty or not accessible"

# --- Set PYTHONPATH to project root ---
export PYTHONPATH=/home/aimanabdulwaha/dense-retrieval-mining-thesis:${PYTHONPATH}

# --- Run training ---
# Default arguments (can be overridden by setting environment variables before sbatch)
# Example: STRATEGY=random sbatch scripts/run_training.sh
STRATEGY=${STRATEGY:-in-batch}
MODEL_NAME=${MODEL_NAME:-sentence-transformers/all-MiniLM-L6-v2}
NUM_EXAMPLES=${NUM_EXAMPLES:-100000}
BATCH_SIZE=${BATCH_SIZE:-32}
EPOCHS=${EPOCHS:-1}

echo "=========================================="
echo "Training Configuration:"
echo "  Strategy: ${STRATEGY}"
echo "  Model: ${MODEL_NAME}"
echo "  Examples: ${NUM_EXAMPLES}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Epochs: ${EPOCHS}"
echo "  Output: ${OUTPUT_DIR}"
echo "=========================================="

python src/train.py \
  --strategy "${STRATEGY}" \
  --model-name "${MODEL_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-examples "${NUM_EXAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}"

echo "Training completed. Model saved to: ${OUTPUT_DIR}"

