import torch

# ✅ Load FSRCNN from torch.hub
fsrcnn_model = torch.hub.load("pytorch/super-resolution", "fsrcnn", pretrained=False)

# ✅ Load Pretrained Weights
fsrcnn_model.load_state_dict(torch.load("FSRCNN-x4.pt", map_location="cpu"))

# ✅ Set to Evaluation Mode
fsrcnn_model.eval()

print("✅ FSRCNN Model Loaded Successfully!")
