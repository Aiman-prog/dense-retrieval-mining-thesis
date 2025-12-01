# Build stage: Install dependencies and build tools
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime AS builder

WORKDIR /build

# Install system dependencies including Java 11 for PyTerrier
# DEBIAN_FRONTEND=noninteractive prevents interactive prompts (like timezone selection)
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    openjdk-11-jdk \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Java environment variables (architecture-agnostic for multi-platform builds)
# Create symlink to architecture-independent path
RUN JAVA_DIR=$(ls -d /usr/lib/jvm/java-11-openjdk-* | head -1) && \
    ln -sf ${JAVA_DIR} /usr/lib/jvm/java-11-openjdk
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage: Minimal image with only runtime dependencies
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install only Java 11 runtime (not full JDK) for PyTerrier
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    openjdk-11-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Set Java environment variables (architecture-agnostic for multi-platform builds)
# Create symlink to architecture-independent path
RUN JAVA_DIR=$(ls -d /usr/lib/jvm/java-11-openjdk-* | head -1) && \
    ln -sf ${JAVA_DIR} /usr/lib/jvm/java-11-openjdk
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Set Python path and ensure local packages are in PATH
ENV PYTHONPATH=/app
ENV PATH=/root/.local/bin:${PATH}

# Default command
CMD ["/bin/bash"]
