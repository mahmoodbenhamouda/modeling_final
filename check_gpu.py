import torch
import sys

def check_gpu():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device properties:")
        props = torch.cuda.get_device_properties(0)
        for key, value in props.__dict__.items():
            print(f"  {key}: {value}")
    else:
        print("No CUDA device available. Running on CPU only.")

if __name__ == "__main__":
    check_gpu() 