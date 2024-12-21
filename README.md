# RDC

This project implements a deep learning-based solution for classifying retinal OCT (Optical Coherence Tomography) images into one of eight categories of retinal diseases. It includes an explainable AI component that generates Grad-CAM heatmaps to visualize the areas in the image that the model focuses on for its predictions. The project is built using PyTorch, Streamlit, and MONAI, with support for GPU acceleration.

Features

Retinal Disease Classification: Uses a pre-trained DenseNet121 model for multi-class classification of retinal diseases.

Explainable AI with Grad-CAM: Generates Grad-CAM heatmaps to highlight the areas of the image most relevant to the model's decision.

Streamlit Frontend: An interactive web app for uploading OCT images, viewing classification results, and visualizing Grad-CAM heatmaps.

The DenseNet121 model is trained using PyTorch and MONAI. The Grad-CAM explanation is implemented using hooks on the last convolutional layer of the model. For detailed training procedures, refer to the Jupyter notebooks provided in the repository (ai_project_v1.ipynb and ai_project_v2.ipynb).
