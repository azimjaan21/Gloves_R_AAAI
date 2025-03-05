import torch

model_path = "RealESRGAN_x2.pth"
state_dict = torch.load(model_path, map_location="cpu")

print("Available keys in the model:", state_dict.keys())
