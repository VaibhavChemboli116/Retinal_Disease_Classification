import streamlit as st
import torch
import numpy as np
import cv2
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import torch.nn as nn
from monai.networks.nets import DenseNet121

import os
import gdown

# File ID and output path
file_id = "1XiPwUNIAGgkJxOIrq7vijEyInnBCTE-3"
output_path = "best_densenet_model.pth"

# Check if the file already exists
if not os.path.exists(output_path):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
    print("Download completed.")


class DenseNetClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(DenseNetClassifier, self).__init__()
        self.model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=num_classes)

    def forward(self, x):
        return self.model(x)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        output = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        self.model.zero_grad()
        target = output[:, class_idx]
        target.backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        cam = cv2.resize(cam, (x.size(2), x.size(3)))
        return cam, class_idx

# Initialize model
device = torch.device("mps" if torch.mps.is_available() else "cpu")
model = DenseNetClassifier(num_classes=8).to(device)
model.load_state_dict(torch.load("best_densenet_model.pth", map_location=torch.device('cpu')))
model.eval()

def get_last_conv_layer(model):
    for layer_name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return module
    raise ValueError("No Conv2D layer found in the model.")

last_conv_layer = get_last_conv_layer(model.model)
grad_cam = GradCAM(model, last_conv_layer)

# Define image transformations
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Streamlit app
st.title("Explainable AI: Retinal Disease Classification and Grad-CAM Visualization")
st.write("Upload a retinal OCT image to see the predicted class and Grad-CAM visualization.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Get Grad-CAM heatmap and predicted class
    heatmap, predicted_class = grad_cam(input_tensor)

    # Load class names
    class_names = ["AMD", "CNV", "CSR", "DME", "DR", "DRUSEN", "MH", "NORMAL"]
    predicted_label = class_names[predicted_class]

    st.write(f"Predicted Class: {predicted_label}")

    # Generate heatmap overlay
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    image_np = np.array(image)
    image_resized = cv2.resize(image_np, (heatmap.shape[1], heatmap.shape[0]))
    overlay = cv2.addWeighted(image_resized, 0.5, heatmap, 0.5, 0)

    # Convert from BGR to RGB for Streamlit
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Display using Streamlit
    st.image(overlay_rgb, caption="Grad-CAM Heatmap", use_container_width=True)
