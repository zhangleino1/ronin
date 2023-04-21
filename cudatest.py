import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Move tensors to GPU device
    device = torch.device("cuda")
    x = torch.randn(3, 3).to(device)
    y = torch.randn(3, 3).to(device)

    # Perform some operations on the tensors
    z = torch.matmul(x, y)
    print(z)
else:
    print("CUDA is not available")
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.cudnn.version())