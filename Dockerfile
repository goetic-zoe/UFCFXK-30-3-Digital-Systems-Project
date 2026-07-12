FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3
LABEL authors="goetic_zoe"
WORKDIR /workspace
RUN pip install --no-cache-dir Augmentor matplotlib
ENV TF_GPU_THREAD_MODE="gpu_private"
ENV TF_GPU_THREAD_COUNT="1"
ENV TF_CPP_MIN_LOG_LEVEL="3"
ENV GLOG_minloglevel="3"
ENV NVCC_APPEND_FLAGS="-arch=native"
ENV TF_CUDA_COMPUTE_CAPABILITIES="12.0"
ENV XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=false --xla_gpu_cuda_data_dir=/usr/local/cuda"
ENTRYPOINT ["top", "-b"]