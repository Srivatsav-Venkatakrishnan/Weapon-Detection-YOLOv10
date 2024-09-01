# -*- coding: utf-8 -*-
"""
Script for training a YOLOv10n model on a custom dataset and exporting the trained model.
@author: SRIVATSAV
"""

from ultralytics import YOLO  # Import the YOLO class from the Ultralytics library
import torch  # Import PyTorch for deep learning operations

if __name__ == '__main__':

    # Optional: Set the CUDA device if using multiple GPUs
    # torch.cuda.set_device(0)
    # device = torch.device("cuda:0")

    # Step 1: Load the YOLO Model
    # Option 1: Build a new YOLOv10n model from a configuration file
    model = YOLO("yolov10n.yaml")

    # Option 2: Load a pre-trained YOLOv10n model (recommended for fine-tuning)
    # model = YOLO('yolov10n.pt')

    # For this example, we're using a custom pre-trained YOLOv10n model located on disk
    model = YOLO('C:/Users/sriva/Downloads/weapons_detection/weights_yolov10n/best.pt')

    # Step 2: Train the Model
    # Train the model on a custom dataset with the following parameters:
    # - `data`: Path to the dataset configuration file
    # - `device`: GPU device ID (0 for the first GPU)
    # - `epochs`: Number of training epochs
    # - `imgsz`: Image size for training (320x320)
    # - `optimizer`: Optimizer to use (AdamW in this case)
    # - `name`: Name of the training run (useful for organizing training results)
    # - `augment`: Whether to use data augmentation
    # - `lr0`: Initial learning rate
    # - `patience`: Number of epochs with no improvement after which learning rate will be reduced
    # - `cos_lr`: Whether to use cosine learning rate scheduling
    # - `workers`: Number of data loader workers (for parallel data loading)
    # - `batch`: Batch size
    results = model.train(
        data="C:/Users/sriva/Downloads/weapons_detection/config.yaml",
        device='0',
        epochs=650,
        imgsz=320,
        optimizer="AdamW",
        name="yolov10n_new_god2",
        augment=True,
        lr0=0.0001,
        patience=None,
        cos_lr=False,
        workers=8,
        batch=32
    )

    # Optional: Fine-tune the model using additional parameters
    # results = model.tune(
    #     data="/home/ubuntu/sabari/Yolo/config.yaml",
    #     epochs=100,
    #     imgsz=300,
    #     name="yolov8n_face_person_white_fine_tune",
    #     use_ray=True
    # )

    # Step 3: Export the Model
    # Export the trained model to ONNX format for deployment or further use
    model.export(format="onnx", imgsz=320)
    
    # Optional: Export the model to OpenVINO format
    # model.export(format="openvino", imgsz=320)
