import torch

def verify_pytorch():
    """
    Verifies the PyTorch installation.
    """
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available. GPU is ready for use.")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is not available. PyTorch will use CPU.")
        
    # Create a sample tensor
    x = torch.rand(5, 3)
    print("\nSample tensor:")
    print(x)

if __name__ == "__main__":
    verify_pytorch()
