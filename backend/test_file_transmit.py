from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image
import pytest
import os
from app import app

# Assuming your app instance is named 'app'
client = TestClient(app)

# Test the file upload and image reading part
def test_image_processing():
    # Open a sample image to simulate uploading a file
    image_path = "../rooms/francesca-tosolini-hCU4fimRW-c-unsplash.jpg"  # Replace with an actual image path for testing
    with open(image_path, "rb") as f:
        image_data = f.read()

    # Create a fake file object for testing purposes
    fake_file = BytesIO(image_data)
    fake_file.name = "test_image.jpg"  # Fake the name of the file for FastAPI's file handling

    # Test the image reading and conversion code
    image = Image.open(fake_file).convert("RGB")

    # Verify that the image was opened correctly
    assert image is not None  # Check if image is loaded
    assert image.mode == "RGB"  # Ensure the image is in RGB mode

    # You can add more assertions based on the image content if necessary
    print(f"Image size: {image.size}")  # Print image size for confirmation

    # Now proceed with the actual backend image processing if needed...
    # input_tensor = preprocess_image(image)  # Assuming preprocess_image() function is defined
    # print(f"Processed image tensor shape: {input_tensor.shape}")  # Print tensor shape for verification

# Run the test
if __name__ == "__main__":
    pytest.main(["-v", "test_file_transmit.py"])  # Save this test as 'test_file_transmit.py'
