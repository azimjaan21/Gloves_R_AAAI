import sys
import os
import torch

# Add YOWO directory to Python path
yowo_path = os.path.abspath("c:/Users/dalab/Desktop/azimjaan21/Gloves_R_AAAI/YOWO")
sys.path.append(yowo_path)

# FIX: Modify model.py imports before importing YOWO
from YOWO.core.model import YOWO 

# Define Path to YOWO Config File (Try YAML or Python Config)
cfg_path = "c:/Users/dalab/Desktop/azimjaan21/Gloves_R_AAAI/YOWO/cfg/custom_config.py"  # Change this if needed

# Load YOWO Model with Config
yowo_model = YOWO(cfg_path)  # Pass cfg as argument

# Load Pretrained Weights
resnet_weights = "resnet-50-kinetics.pth"
checkpoint = torch.load(resnet_weights, map_location="cuda")
yowo_model.load_state_dict(checkpoint["state_dict"], strict=False)

# Send Model to GPU
yowo_model = yowo_model.cuda().eval()

print("✅ YOWO Model Loaded with Pretrained Weights!")
