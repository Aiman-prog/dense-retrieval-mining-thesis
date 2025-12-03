# Cluster Setup Instructions

This guide explains how to set up and run evaluation on DelftBlue cluster.

## Prerequisites

1. **Clone the repository** on the cluster:
   ```bash
   git clone <repository-url>
   cd dense-retrieval-mining-thesis
   ```

2. **Set up conda environment** with dependencies:
   ```bash
   # Load miniconda3 module
   module load 2025
   module load miniconda3/4.12.0
   
   # Initialize conda
   eval "$(conda shell.bash hook)"
   
   # Create and activate environment
   conda create -n dense-retrieval python=3.10
   conda activate dense-retrieval
   
   # Install Java (required for PyTerrier)
   conda install -c conda-forge openjdk=11
   
   # Install Python dependencies (includes faiss-gpu for GPU-accelerated indexing)
   pip install -r requirements.txt
   ```

## ⚠️ REQUIRED: Setup Before Running sbatch

**You MUST complete these steps BEFORE submitting your job!** The cluster has no internet during job execution.

### Step 1: Set Up Conda Environment

```bash
# Load modules
module load 2025
module load miniconda3/4.12.0
eval "$(conda shell.bash hook)"

# Create and activate environment
conda create -n dense-retrieval python=3.10
conda activate dense-retrieval

# Install Java (required for PyTerrier)
conda install -c conda-forge openjdk=11

# Install Python dependencies (includes faiss-gpu for GPU-accelerated indexing)
pip install -r requirements.txt
```

### Step 2: Pre-download PyTerrier Java Dependencies

On the **login node** (with internet access):

```bash
# Load modules and activate environment
module load 2025
module load miniconda3/4.12.0
eval "$(conda shell.bash hook)"
conda activate dense-retrieval

# Initialize PyTerrier with explicit versions (required for offline mode)
# This downloads Java dependencies to ~/.pyterrier/ and caches the version info
python -c "import pyterrier as pt; pt.init(version='5.11', helper_version='0.0.8'); print('PyTerrier initialized and Java dependencies downloaded')"
```

### Step 3: Pre-download MS MARCO Dataset to Scratch

**This is the main required pre-download** since the dataset is large (~20GB) and must be cached before offline execution. We'll download it directly to scratch space.

On the **login node** (with internet access), from your project directory:

```bash
# Load modules and activate environment
module load 2025
module load miniconda3/4.12.0
eval "$(conda shell.bash hook)"
conda activate dense-retrieval

# Set up scratch directory and symlink BEFORE downloading
# This ensures dataset downloads directly to scratch
mkdir -p /scratch/${USER}/caches/datasets/msmarco_passage
mkdir -p ~/.pyterrier/corpora

# Create symlink (target directory must exist first)
ln -s /scratch/${USER}/caches/datasets/msmarco_passage ~/.pyterrier/corpora/msmarco_passage

# Now download - PyTerrier will download to the symlink location (scratch)
python << EOF
import pyterrier as pt
# Use explicit versions for offline mode
pt.terrier.set_version('5.11')
pt.terrier.set_helper_version('0.0.8')
pt.java.init()
dataset = pt.get_dataset('msmarco_passage')

# Download test-2019 variant (topics and qrels) - these are small but REQUIRED
print("Downloading test-2019 topics and qrels (REQUIRED for evaluation)...")
print("This will download and cache the files to ~/.pyterrier/corpora/msmarco_passage/")
topics = dataset.get_topics(variant='test-2019')
print(f'Topics downloaded: {len(topics)} queries')
qrels = dataset.get_qrels(variant='test-2019')
print(f'Qrels downloaded: {len(qrels)} judgments')
print('IMPORTANT: These files are now cached in ~/.pyterrier/corpora/msmarco_passage/')
print('Verifying files are cached...')
import os
pyterrier_home = os.path.expanduser('~/.pyterrier/corpora/msmarco_passage')
if os.path.exists(pyterrier_home):
    files = os.listdir(pyterrier_home)
    print(f'Cached files: {files}')
else:
    print(f'WARNING: Directory {pyterrier_home} does not exist!')

# Download full corpus (~8.8M documents, ~20GB)
# This will take 30+ minutes but ensures the full dataset is cached
print("Downloading full corpus (this will take 30+ minutes)...")
corpus_iter = dataset.get_corpus_iter()
count = 0
for doc in corpus_iter:
    count += 1
    if count % 100000 == 0:  # Progress update every 100k documents
        print(f"Downloaded {count} documents...")
print(f'Full corpus download complete: {count} documents cached in scratch')
EOF
```

**Note:** This downloads the complete corpus (~8.8M documents, ~20GB) directly to scratch. This will take 30+ minutes but ensures the full dataset is available for evaluation.

### Step 3b: Download Training Data (Required for train.py)

**If you already completed Step 3, you only need to download the training data separately:**

On the **login node** (with internet access), from your project directory:

```bash
# Load modules and activate environment
module load 2025
module load miniconda3/4.12.0
eval "$(conda shell.bash hook)"
conda activate dense-retrieval

# Download training data (train variant and eval.small)
python << EOF
import pyterrier as pt
pt.terrier.set_version('5.11')
pt.terrier.set_helper_version('0.0.8')
pt.java.init()
dataset = pt.get_dataset('msmarco_passage')

# Download TRAINING data (required for train.py)
print("Downloading training topics and qrels...")
topics_train = dataset.get_topics(variant='train')
print(f'Training topics downloaded: {len(topics_train)} queries')
qrels_train = dataset.get_qrels(variant='train')
print(f'Training qrels downloaded: {len(qrels_train)} judgments')

# Download dev.small data (required for training evaluation - has both topics and qrels)
print("Downloading dev.small topics and qrels...")
topics_eval = dataset.get_topics(variant='dev.small')
print(f'Dev.small topics downloaded: {len(topics_eval)} queries')
qrels_eval = dataset.get_qrels(variant='dev.small')
print(f'Dev.small qrels downloaded: {len(qrels_eval)} judgments')
print('All training data is now cached!')
EOF
```

**Note:** This only downloads the topics and qrels files (small, fast download). The corpus was already downloaded in Step 3.

### Step 4: Download HuggingFace Model to Scratch

**Download HuggingFace model directly to scratch:**

Run from your project directory (`/home/aimanabdulwaha/dense-retrieval-mining-thesis`):

```bash
# Load modules and activate environment
module load 2025
module load miniconda3/4.12.0
eval "$(conda shell.bash hook)"
conda activate dense-retrieval

# Create scratch directory for HuggingFace cache
mkdir -p /scratch/${USER}/caches/huggingface

# Set cache location to scratch and download model
export HF_HOME=/scratch/${USER}/caches/huggingface
export TRANSFORMERS_CACHE=/scratch/${USER}/caches/huggingface
export SENTENCE_TRANSFORMERS_HOME=/scratch/${USER}/caches/huggingface

# Download the model (will be cached directly in scratch)
# Both evaluation and training use the same model: sentence-transformers/all-MiniLM-L6-v2
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### Step 5: Verify Everything is Ready

Run these checks before submitting:

```bash
# 1. Verify conda environment exists and has packages
conda activate dense-retrieval
which python
# Should show: /home/aimanabdulwaha/.conda/envs/dense-retrieval/bin/python

pip list | grep -E "sentence-transformers|pyterrier|pandas|numpy|faiss-gpu"
python -c "import sentence_transformers, pyterrier, pandas, numpy, faiss; print('All packages installed (including FAISS-GPU)')"

# 2. Verify Java is available
java -version

# 3. Verify PyTerrier can initialize
python -c "import pyterrier as pt; pt.init(); print('PyTerrier OK')"

# 4. Verify dataset is in scratch and symlinked
ls -la ~/.pyterrier/corpora/msmarco_passage
# Should show a symlink pointing to /scratch/${USER}/caches/datasets/msmarco_passage

# 5. Verify HuggingFace model is cached in scratch
echo "Checking HuggingFace cache..."
ls -la /scratch/${USER}/caches/huggingface/ | head -20
# Should show model directory for sentence-transformers/all-MiniLM-L6-v2

# 6. Test offline model loading (CRITICAL - must work before submitting job)
export HF_HOME=/scratch/${USER}/caches/huggingface
export TRANSFORMERS_CACHE=/scratch/${USER}/caches/huggingface
export SENTENCE_TRANSFORMERS_HOME=/scratch/${USER}/caches/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Test loading the model in offline mode (used for both training and evaluation)
python -c "from sentence_transformers import SentenceTransformer; import os; os.environ['HF_HUB_OFFLINE']='1'; m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('✓ Model loads offline')"

# Verify dataset actually exists in scratch
ls -la /scratch/${USER}/caches/datasets/msmarco_passage/
# Should show dataset files

# 5. Verify FAISS-GPU is installed
python -c "import faiss; print(f'FAISS version: {faiss.__version__}'); print(f'GPU support: {hasattr(faiss, \"StandardGpuResources\")}')"

    # 9. Verify HuggingFace model is in scratch
# Note: sentence-transformers uses different directory structure (with underscores)
ls -la /scratch/${USER}/caches/huggingface/sentence-transformers_all-MiniLM-L6-v2/

    # 10. Verify Python script syntax
export PYTHONPATH=$(pwd):${PYTHONPATH}
python -m py_compile src/evaluate.py
python src/evaluate.py --help

    # 11. Verify SLURM script path matches your directory
pwd
# Should match --chdir in scripts/run_evaluation.sh
```


If all checks pass, you're ready to submit!

## Running Evaluation

**⚠️ Make sure you completed all 5 steps above before submitting!**

### 1. Verify SLURM Script Path

Check that `--chdir` in `scripts/run_evaluation.sh` matches your project directory:
```bash
pwd
# Should be: /home/aimanabdulwaha/dense-retrieval-mining-thesis
# Compare with line 13 in scripts/run_evaluation.sh
```

### 2. Submit Job

```bash
sbatch scripts/run_evaluation.sh
```

### 3. Monitor Job

```bash
# Check job status
squeue -u $USER

# View output (after job completes)
cat evaluation_*.out
cat evaluation_*.err
```

## Output

Results will be saved to `evaluation_results.json` in the project directory, containing:
- Model path
- Dataset variant
- nDCG@10 and MRR@10 metrics for both Dense and BM25
- Number of queries and qrels evaluated

