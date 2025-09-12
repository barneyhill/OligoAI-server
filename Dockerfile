# Use RunPod's PyTorch base image
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Update system packages and install vim
RUN apt-get update && \
    apt-get install -y vim wget curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    runpod \
    torch==2.1.0 \
    flash-attn==2.3.2 \
    pandas \
    scipy \
    lightning \
    wandb \
    ml_collections


# Uninstall torchvision as specified
RUN pip uninstall -y torchvision

# Clone the RiNALMo repository
RUN git clone https://github.com/barneyhill/RiNALMo

# Download the OligoAI model checkpoint
RUN wget https://huggingface.co/barneyhill/OligoAI/resolve/main/OligoAI_11_09_25.ckpt

# Set environment variables for better performance
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"

# can't find any other way to break cache on github
RUN rm -rf RiNALMo && git clone https://github.com/barneyhill/RiNALMo

COPY rp_handler.py .
CMD ["python3", "-u", "rp_handler.py"]
