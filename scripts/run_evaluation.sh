#!/usr/bin/env bash

#SBATCH --job-name=dense_eval
#SBATCH --partition=gpu-a100
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=evaluation_%j.out
#SBATCH --error=evaluation_%j.err
#SBATCH --chdir=/home/aimanabdulwaha/newproject/dense-retrieval-mining-thesis

# --- Load modules ---
module purge
module load 2025
module load cuda/12.9
module load miniconda3/4.12.0

# --- Activate Conda environment ---
eval "$(conda shell.bash hook)"
conda activate ai-assistant-env

# --- Set Hugging Face to offline mode (use cache only) ---
export HF_HOME=/home/aimanabdulwaha/datasets/huggingface_cache
export HF_DATASETS_CACHE=/home/aimanabdulwaha/datasets/huggingface_cache
export TRANSFORMERS_CACHE=/home/aimanabdulwaha/datasets/huggingface_cache
export SENTENCE_TRANSFORMERS_HOME=/home/aimanabdulwaha/datasets/huggingface_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# --- Set PyTerrier cache (if different from default) ---
# PyTerrier will use ~/.pyterrier by default, ensure it's accessible

# --- Set scratch space for BM25 index ---
SCRATCH_DIR="/scratch/${USER}/dense-retrieval"
mkdir -p ${SCRATCH_DIR}
INDEX_PATH="${SCRATCH_DIR}/bm25_index"

# --- Run evaluation ---
python src/evaluate.py \
  --model-path sentence-transformers/all-MiniLM-L6-v2 \
  --variant test-2019 \
  --index-path ${INDEX_PATH} \
  --output-file evaluation_results.json

