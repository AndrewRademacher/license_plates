[linux]
train:
    TORCH_CUDA_VERSION=cu117 cargo run --release -- train -m target/license.model

[macos]
train:
    cargo run --release -- train -m target/license.model

[linux]
prepare:
    TORCH_CUDA_VERSION=cu117 cargo run --release -- prepare -n target/license.norm.json -l target/license.label.json

[macos]
prepare:
    cargo run --release -- prepare -n target/license.norm.json -l target/license.label.json

[linux]
infer:
    TORCH_CUDA_VERSION=cu117 cargo run --release -- inference -m target/license.model -n target/license.norm.json -l target/license.label.json -i data/plates/valid/MISSOURI/2.jpg -i data/plates/valid/ALABAMA/2.jpg -i data/plates/valid/TEXAS/2.jpg

[macos]
infer:
    cargo run --release -- inference -m target/license.model -n target/license.norm.json -l target/license.label.json -i data/plates/valid/MISSOURI/2.jpg -i data/plates/valid/ALABAMA/2.jpg -i data/plates/valid/TEXAS/2.jpg
