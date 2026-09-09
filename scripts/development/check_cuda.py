import torch

print("Torch version:", torch.__version__)

print("Is CUDA enabled?", torch.cuda.is_available())

print(torch.cuda.is_bf16_supported())

