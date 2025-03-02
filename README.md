# Detection-of-Oil-Spill-in-Multi-Source-Sentinel-1-SAR-Images-using-Vision-Transformer-Approach
This repository contains code for detecting oil spills in multi-source Sentinel-1 SAR images using a Vision Transformer (ViT) model. The project leverages PyTorch and the timm library to fine-tune a pre-trained ViT_L_16 model for binary classification (oil spill vs. no oil spill). In addition, an alternative transformer encoder model is provided to explore different architectures.

## Training the Model
The training process is implemented in the training and testing ViT.ipynb notebook. Key steps include:

### Data Preprocessing & Augmentation:
Images are resized to 224×224 and normalized using ImageNet statistics. Augmentations can be applied as needed.

### Dataset Splitting:
The dataset is split into training and validation sets by separating images based on their labels and using an 80/20 split for each class.

### Model Setup:
A pre-trained ViT_L_16 model (from torchvision) is loaded, and its classification head is modified to output 2 classes. Multi-GPU support is enabled with DataParallel if available.

### Training Loop:
The model is trained using CrossEntropyLoss with class weights to handle class imbalance. Metrics (loss, accuracy, precision, recall, and F1 score) are computed and logged per epoch, and checkpoints are saved based on the best validation and training accuracies.

### Metrics Logging:
Training history is saved as a CSV file for later analysis.

# Testing the Model
Two testing approaches are provided:

## 1. Notebook Testing
Within the training and testing ViT.ipynb notebook, sample images are processed and predictions are visualized directly using matplotlib.

## 2. Script Testing
The test.py script allows for command-line testing of the trained model. To test an image, run:

bash python test.py test_images/one.tif model_pretrained_weights/Vision_Transformer.pth
### The script will:
Preprocess the input TIFF image (handle NaNs, adjust channels, resize, and normalize).
Load the pre-trained ViT model with the provided checkpoint.
Output the predicted class and its confidence score.
Display the image with the prediction overlay.
## Conclusion

In summary, this project demonstrates a robust approach to detecting oil spills in Sentinel-1 SAR images using a Vision Transformer architecture. By leveraging advanced deep learning techniques and thorough preprocessing of multi-source data, we achieved effective binary classification between oil spill and non-oil spill scenarios. This repository not only provides a complete training and testing pipeline but also explores alternative transformer-based architectures, underscoring the flexibility and power of these models in remote sensing applications.

We hope this work serves as a valuable resource for researchers and practitioners aiming to advance environmental monitoring using cutting-edge deep learning methods. Contributions, feedback, and further enhancements are highly encouraged to improve the system and adapt it to broader applications.
