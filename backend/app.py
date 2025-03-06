from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import numpy as np
import io
import base64
from model import load_segmentation_model, segment_image

app = FastAPI()

# Load the segmentation model
model = load_segmentation_model()

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    """ Handle image segmentation request """
    # Read the uploaded image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # Preprocess image (you can add the code later)
    input_tensor = preprocess_image(image)  # Here you can implement the gray conversion
    segmented_mask = segment_image(model, input_tensor)
    
    # Convert the processed mask to base64 for display on frontend
    _, buffer = cv2.imencode('.png', segmented_mask)
    processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse(content={"processed_image": processed_image_base64})


def preprocess_image(image: Image.Image):
    """ A placeholder for preprocessing logic """
    # This will return the image as is for now, you can add gray conversion here
    return image


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
