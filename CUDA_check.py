import torch

# Check for CUDA availability (GPU support)
if torch.cuda.is_available():
    print("CUDA is available!")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU instead.")
    device = torch.device("cpu")

# Create a tensor and move it to the selected device
x = torch.rand(5, 3).to(device)

# Do some operations
y = x * x
print(y)