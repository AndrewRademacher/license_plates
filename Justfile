train:
    TORCH_CUDA_VERSION=cu117 cargo run --release -- train -m target/license.model

prepare:
    TORCH_CUDA_VERSION=cu117 cargo run --release -- prepare -n target/license.norm.json
