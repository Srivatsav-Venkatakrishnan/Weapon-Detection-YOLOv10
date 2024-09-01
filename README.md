# Weapon-Detection-YOLOv10
The system is designed to accurately identify and classify weapons in live video feeds or images, enabling rapid response in security-sensitive environments. Leveraging the power of YOLOv10s, this solution provides a balance between speed and accuracy, making it suitable for deployment in real-world applications.
# Real-Time Weapon Detection using YOLOv10s

This repository contains the implementation of a real-time weapon detection system using the YOLOv10s model. The system is designed to detect and identify weapons in live video feeds or static images, providing a powerful tool for enhancing security in various environments.

## Features

- **Real-Time Processing**: Efficiently processes video streams with minimal delay.
- **High Accuracy**: Leverages the YOLOv10s model for precise weapon detection.
- **OpenVINO Optimization**: Utilizes the OpenVINO toolkit for optimized model inference on Intel hardware.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x
- Required Python libraries (specified in `requirements.txt`)
- OpenVINO toolkit installed and configured (if using OpenVINO optimized model)

## Setup and Installation

1. **Clone the Repository:**

   git clone https://github.com/Srivatsav-Venkatakrishnan/Weapon-Detection-YOLOv10.git
   cd weapon-detection-YOLOv10
   
2. **Install Dependencies:**
pip install -r requirements.txt

3. **Prepare Your Model:**

Ensure that your YOLOv10s model is converted to the OpenVINO format and that the .xml and .bin files are in place.

4. **Usage:**
To run weapon detection on a video, use the following command:
python test_video.py --video_path="path/to/your/video.mp4" --object_model_xml="path/to/your/model.xml"

5. **Example Commands:**
Run on a specific video:
python test_video.py --video_path="C:/Users/sriva/Downloads/weapons_detection/IMG_4706.mov" --object_model_xml="C:/Users/sriva/Downloads/weapons_detection/weights_yolov10n/best_openvino_model/best.xml"

Run on another test video:
python test_video.py --video_path="C:/Users/sriva/Downloads/weapons_detection/Knife_test.mp4" --object_model_xml="C:/Users/sriva/Downloads/weapons_detection/weights_yolov10n/best_openvino_model/best.xml"

6. **Model and Weights:**
The model used for detection is based on YOLOv10s, optimized with the OpenVINO toolkit for better performance. Ensure the weights file (best.xml) is correctly linked in the commands above.

7. **Contributing**
If you'd like to contribute to this project, please fork the repository and create a pull request with your proposed changes. We appreciate all contributions!

8. **License**
This project is licensed under the MIT License. See the LICENSE file for details.







