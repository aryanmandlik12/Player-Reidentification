import subprocess
import sys
import os
import requests
from pathlib import Path

def install_requirements():
    """Install required packages"""
    requirements = [
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=10.0.0",
        "gdown>=4.7.0"
    ]
    
    print("Installing required packages...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
    
    print("Setup complete!")

def download_model():
    """Download YOLOv11 model from Google Drive"""
    model_path = "yolov11_players.pt"
    
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return
    
    # Google Drive file ID from the provided link
    file_id = "1-5tOSHOSB9UXYPenOoZNAMScrePVcMD"
    
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        print("Downloading YOLOv11 model...")
        gdown.download(url, model_path, quiet=False)
        print(f"✓ Model downloaded to {model_path}")
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        print("Please manually download the model from:")
        print("https://drive.google.com/file/d/1-5tOSHOSB9UXYPenOoZNAMScrePVcMD/view")
        print(f"And save it as '{model_path}' in the project root")

if __name__ == "__main__":
    install_requirements()
    download_model()
