import argparse
import torch
import torch.onnx
from basicsr.archs.rrdbnet_arch import RRDBNet

def main(args):
    # Load Real-ESRGAN x2 model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    
    # Load state_dict directly since no 'params' or 'params_ema' key exists
    state_dict = torch.load(args.input, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    # Example input tensor for ONNX conversion
    x = torch.rand(1, 3, 64, 64)
    
    # Export the model to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model, x, args.output, opset_version=11, export_params=True, 
            input_names=['input'], output_names=['output']
        )
    print(f"✅ ONNX model saved at {args.output}")

if __name__ == '__main__':
    """ Convert Real-ESRGAN x2 PyTorch model to ONNX """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='RealESRGAN_x2.pth', help='Input model path')
    parser.add_argument('--output', type=str, default='realesrgan_x2.onnx', help='Output ONNX path')
    args = parser.parse_args()

    main(args)
