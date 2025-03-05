import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet  # Model itself
from realesrgan import RealESRGANer  # Framework that uses it

# Path to the pre-trained model
model_path = 'RealESRGAN_x4plus.pth'

# Load the model state dictionary
state_dict = torch.load(model_path, map_location=torch.device('cpu'))['params_ema']

# Define RRDBNet model architecture
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
model.load_state_dict(state_dict, strict=True)

# Initialize Real-ESRGAN upsampler
upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=0,
    pre_pad=0,
    half=True
)

# Load input image
img = Image.open('image.png').convert('RGB')
img = np.array(img)

# Run Real-ESRGAN for super-resolution
output, _ = upsampler.enhance(img, outscale=4)

# Save the output image
output_img = Image.fromarray(output)
output_img.save('output.png')
