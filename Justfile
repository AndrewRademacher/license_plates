train:
    TORCH_CUDA_VERSION=cu117 cargo run --release -- train -m first.model

prepare:
    TORCH_CUDA_VERSION=cu117 cargo run --release -- prepare
