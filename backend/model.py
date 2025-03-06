import torch
import cv2
import numpy as np

def load_segmentation_model():
    """ Load a pretrained DeepLabV3 model """
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    model.eval()
    return model

def segment_image(model, image_tensor):
    """ Apply segmentation model and return mask """
    with torch.no_grad():
        output = model(image_tensor)['out']
    output_predictions = output.argmax(1).squeeze(0).cpu().numpy()
    
    # Convert to binary masks (walls, ceiling, floor)
    mask = (output_predictions == 15).astype(np.uint8) * 255  # Example: Detect walls (COCO class 15)
    
    return mask
