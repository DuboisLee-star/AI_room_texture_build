from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
import torch
from torchvision import models, transforms
import cv2

import numpy as np
import matplotlib.pyplot as plt

app = FastAPI()

# Load the pre-trained segmentation model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()  # Set the model to evaluation mode


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """ Convert image to tensor and apply necessary transformations """
    image = image.resize((256, 256))  # Resize image to 256x256 for consistency
    
    # Apply standard transformations: convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def segment_image(model, image_tensor: torch.Tensor) -> torch.Tensor:
    """ Segment the image using the model """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(image_tensor)['out']  # Get segmentation output
    
    # Get the class with the highest probability for each pixel (shape: (height, width))
    predicted_class = output.argmax(1).squeeze(0)  # Remove batch dimension, shape: (height, width)
    
    return predicted_class  # Returns (height, width)


@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    """ Handle image segmentation request """
    # Read the uploaded image
    image_data = await file.read()
    with open("tmp/received_image.jpg", "wb") as f:
        f.write(image_data)  # Saving the raw image data to disk
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    # image.show()
    
    input_tensor = preprocess_image(image)

    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input tensor min: {input_tensor.min()}, max: {input_tensor.max()}")

    segmented_mask = segment_image(model, input_tensor)
    print(f"Segmented mask shape: {segmented_mask.shape}")
   
    segmented_mask_np = segmented_mask.cpu().numpy()  # Shape: (height, width)
    plt.imshow(segmented_mask_np)
    plt.title("Segmented Mask")
    plt.axis('off')
    plt.show()

    return JSONResponse(content={"processed_image": True})
    # Perform segmentation
    
    # Convert the segmented mask to a format suitable for display
    # Convert the tensor back to an image (e.g., use torchvision.utils to save the tensor as an image)
    segmented_mask = segmented_mask.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
    _, buffer = cv2.imencode('.png', segmented_mask)
    
    # Convert the image to base64 for frontend display
    processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse(content={"processed_image": processed_image_base64})

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)
