# Dual Encoder Thesis Project

## Quick Start

### Local Installation

**Prerequisites:**
- Python 3.8+
- Java (required for PyTerrier)

**Mac users with Homebrew Java:**
```bash
# Install Java if not already installed
brew install openjdk

# Add to your ~/.zshrc (or ~/.bash_profile):
export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"
export JAVA_HOME="/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home"
export JVM_PATH="/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home/lib/server/libjvm.dylib"

# Then reload your shell or run:
source ~/.zshrc
```

**Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**Set PYTHONPATH (required for imports to work):**
```bash
# From the project root directory:
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Or add to your ~/.zshrc (or ~/.bashrc) for persistence:
echo 'export PYTHONPATH="${PYTHONPATH}:$(pwd)"' >> ~/.zshrc
source ~/.zshrc
```

**Note:** 
- Model encoding (PyTorch) will automatically use CUDA/MPS/CPU based on availability
- FAISS indexing/search uses CPU (works on all platforms)
- PYTHONPATH must be set for the `src` package imports to work when running scripts directly

### Build Multi-Platform Image (First Time)

```bash
# Create buildx builder (one-time setup)
docker buildx create --name multiarch --use
docker buildx inspect --bootstrap

# Build for both amd64 and arm64
docker buildx build --platform linux/amd64,linux/arm64 -t dense-retrieval-mining-thesis:latest --load .
```

### Run Container

**Mac:**
```bash
docker-compose up training -d
docker-compose exec training bash
```

**Linux/Windows (with GPU):**
```bash
docker-compose --profile gpu up training-gpu -d
docker-compose exec training-gpu bash
```

### Stop Container

```bash
docker-compose down
```

## Cluster Execution

For running on DelftBlue cluster, see [CLUSTER_SETUP.md](CLUSTER_SETUP.md) for detailed instructions on:
- Pre-downloading models and datasets
- Setting up the environment
- Submitting evaluation jobs

## Notes

- Uses QEMU emulation for multi-platform builds (automatic)
- Mac GPU (MPS) requires native execution: `python src/train.py`
- Image supports both `linux/amd64` and `linux/arm64` natively
