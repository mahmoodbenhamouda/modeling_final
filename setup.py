import os
import subprocess
import sys
import platform

def run_command(command):
    """Run a command and print output"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Errors: {result.stderr}")
    return result.returncode == 0

def create_directories():
    """Create necessary directories"""
    directories = ["models", "generated_images", "logs"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"CUDA is available: {has_cuda}")
            print(f"Device count: {device_count}")
            print(f"Device name: {device_name}")
            return True
        else:
            print("CUDA is not available. Models will run on CPU which may be slow.")
            return False
    except ImportError:
        print("PyTorch is not installed, can't check GPU availability.")
        return False
    except Exception as e:
        print(f"Error checking GPU: {str(e)}")
        return False

def install_requirements():
    """Install requirements"""
    print("Installing requirements...")
    return run_command(f"{sys.executable} -m pip install -r requirements.txt")

def install_spacy_model():
    """Install SpaCy model"""
    print("Installing SpaCy model...")
    return run_command(f"{sys.executable} -m spacy download en_core_web_sm")

def setup_environment():
    """Set up the complete environment"""
    print("=== Setting up Digital Marketing AI Platform ===")
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements.")
        return False
    
    # Check GPU
    check_gpu()
    
    # Install SpaCy model
    if not install_spacy_model():
        print("Failed to install SpaCy model.")
        return False
    
    print("\n=== Setup completed successfully! ===")
    print("You can now run the application with:")
    print("  streamlit run app.py")
    return True

if __name__ == "__main__":
    setup_environment() 