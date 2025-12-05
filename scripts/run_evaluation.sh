#!/usr/bin/env bash

#SBATCH --job-name=dense_eval
#SBATCH --partition=gpu-a100
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=evaluation_%j.out
#SBATCH --error=evaluation_%j.err
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

# --- Set up scratch space for caches ---
SCRATCH_CACHE="/scratch/${USER}/caches"
mkdir -p ${SCRATCH_CACHE}/huggingface
mkdir -p ${SCRATCH_CACHE}/datasets

# --- Set up PyTerrier dataset symlink (keep PyTerrier in home, only dataset in scratch) ---
# PyTerrier stays in ~/.pyterrier, but dataset goes to scratch
mkdir -p ~/.pyterrier/corpora
if [ ! -e ~/.pyterrier/corpora/msmarco_passage ]; then
    # Create symlink to dataset in scratch
    ln -s ${SCRATCH_CACHE}/datasets/msmarco_passage ~/.pyterrier/corpora/msmarco_passage
elif [ ! -L ~/.pyterrier/corpora/msmarco_passage ]; then
    # If dataset exists but isn't a symlink, move it to scratch and create symlink
    mv ~/.pyterrier/corpora/msmarco_passage ${SCRATCH_CACHE}/datasets/msmarco_passage
    ln -s ${SCRATCH_CACHE}/datasets/msmarco_passage ~/.pyterrier/corpora/msmarco_passage
fi

# --- Set Hugging Face cache location ---
# Use scratch space for HuggingFace cache
export HF_HOME=${SCRATCH_CACHE}/huggingface
export HF_DATASETS_CACHE=${SCRATCH_CACHE}/huggingface
export TRANSFORMERS_CACHE=${SCRATCH_CACHE}/huggingface
export SENTENCE_TRANSFORMERS_HOME=${SCRATCH_CACHE}/huggingface

# --- Set Hugging Face to offline mode (required for cluster without internet) ---
# Models must be pre-downloaded to cache before running
# Pre-download using: python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2')"
# See: https://huggingface.co/docs/transformers/installation#offline-mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
echo "Offline mode enabled (models must be pre-downloaded to cache)"
echo "Cache location: ${SCRATCH_CACHE}/huggingface"

# --- Set scratch space for BM25 index and dense index ---
SCRATCH_DIR="/scratch/${USER}/dense-retrieval"
mkdir -p ${SCRATCH_DIR}
INDEX_PATH="${SCRATCH_DIR}/bm25_index"
DENSE_INDEX_PATH="${SCRATCH_DIR}/dense_index"  # FAISS creates .faiss and .meta files

# Export SCRATCH_DIR for auto-detection in evaluate.py
export SCRATCH_DIR=${SCRATCH_DIR}

# --- Set PYTHONPATH to project root (use --chdir path) ---
export PYTHONPATH=/home/aimanabdulwaha/dense-retrieval-mining-thesis:${PYTHONPATH}

# --- Run evaluation ---
# Debug: show what paths are being used
echo "DEBUG: INDEX_PATH=${INDEX_PATH}"
echo "DEBUG: DENSE_INDEX_PATH=${DENSE_INDEX_PATH}"
echo "DEBUG: SCRATCH_DIR=${SCRATCH_DIR}"

python src/evaluate.py \
  --model-path distilbert-base-uncased \
  --variant test-2019 \
  --index-path "${INDEX_PATH}" \
  --dense-index-path "${DENSE_INDEX_PATH}" \
  --output-file evaluation_results.json

