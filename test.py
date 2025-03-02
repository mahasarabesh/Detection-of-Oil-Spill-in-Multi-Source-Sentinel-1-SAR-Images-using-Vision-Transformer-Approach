import os
import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, Normalize
from torchvision import models
from collections import OrderedDict
import argparse

# Define a separate preprocessing function
def preprocess_image(image_path, transform):
    """
    Loads a TIFF image from `image_path`, fixes NaNs, ensures a 3-channel tensor,
    applies the provided transform, and returns both the transformed image tensor (with batch dim)
    and a copy for display.
    """
    # Load the image using rasterio
    with rasterio.open(image_path) as src:
        image = src.read()  # shape: [bands, height, width]

    # Convert to a float32 torch tensor
    image = torch.tensor(image, dtype=torch.float32)

    # Fix NaN values for each band
    for band in range(image.shape[0]):
        band_data = image[band]
        nan_mask = torch.isnan(band_data)
        if nan_mask.any():
            mean_value = band_data[~nan_mask].mean()
            band_data[nan_mask] = mean_value

    # Ensure the image has exactly 3 channels
    if image.shape[0] == 3:
        pass
    elif image.shape[0] > 3:
        image = image[:3, :, :]
    else:
        image = image.expand(3, image.shape[1], image.shape[2])

    # Save a copy for display (before applying normalization)
    display_image = image.clone()

    # Apply the provided transformations (e.g., resizing and normalization)
    image = transform(image)

    # Add a batch dimension
    image = image.unsqueeze(0)
    
    return image, display_image

def main(image_path, model_weights_path):
    # Define the transformation (same as used during training)
    input_size = 224
    transform = Compose([
        Resize((input_size, input_size)),  # Resize image to model input size
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # ImageNet normalization
    ])

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained ViT model and adjust the classification head for 2 classes
    model = models.vit_l_16(weights=None)
    num_classes = 2
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    model.to(device)
    model.eval()

    # Load the saved model checkpoint
    checkpoint = torch.load(model_weights_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # Remove 'module.' prefix from state dict keys if present
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value

    # Load the modified state dict into the model
    model.load_state_dict(new_state_dict)

    # Preprocess the image
    input_tensor, display_tensor = preprocess_image(image_path, transform)
    input_tensor = input_tensor.to(device)

    # Make a prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    print(f"Prediction: Class {predicted_class} with confidence {confidence:.2f}")

    # Prepare the image for display
    # Convert tensor (3, H, W) to numpy array (H, W, 3) and normalize for display purposes
    display_image = display_tensor.cpu().numpy()
    display_image = np.transpose(display_image, (1, 2, 0))
    # Normalize values to [0,1] for plotting (this is only for visualization)
    display_image = (display_image - display_image.min()) / (display_image.max() - display_image.min() + 1e-5)

    # Display the image along with the prediction
    plt.figure(figsize=(6,6))
    plt.imshow(display_image)
    plt.title(f"Predicted Class: {predicted_class}\nConfidence: {confidence:.2f}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict image class using a pre-trained ViT model.')
    parser.add_argument('image_path', type=str, help='Path to the input TIFF image.')
    parser.add_argument('model_weights_path', type=str, help='Path to the model weights file.')

    args = parser.parse_args()
    main(args.image_path, args.model_weights_path)
