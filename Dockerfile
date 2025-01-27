# FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
# RUN mkdir /job
# WORKDIR /job
# VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# # You should install any dependencies you need here.
# # RUN pip install tqdm

# Use a PyTorch Docker image with CUDA 12.1 support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set up the working directory inside the container
RUN mkdir /job
WORKDIR /job

# Define mountable volumes for data, source code, checkpoints, and output
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# Install system dependencies and Python libraries required for the project
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install required Python packages
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    transformers \
    tqdm \
    numpy \
    pandas \
    scikit-learn

# Download the model during the build process
RUN python -c "from transformers import AutoTokenizer, AutoModelForMaskedLM; \
    model_name = 'bert-base-multilingual-cased'; \
    AutoTokenizer.from_pretrained(model_name); \
    AutoModelForMaskedLM.from_pretrained(model_name)"

# Copy the source code and work directory into the container
COPY src/ /job/src/
COPY work/ /job/work/

# Ensure the prediction script is executable
RUN chmod +x /job/src/predict.sh

# Set default command to run the prediction script
CMD ["bash", "/job/src/predict.sh", "/job/data/input.txt", "/job/output/pred.txt"]

