import cv2
import numpy as np
import os
import openvino as ov
from tqdm import tqdm
import argparse

def process_video(video_path, model_xml_path):
    """
    Process a video file to detect specific objects using an OpenVINO optimized model.

    Args:
        video_path (str): Path to the input video file.
        model_xml_path (str): Path to the OpenVINO model XML file.
    """

    # Initialize OpenVINO core and load the model
    core = ov.Core()
    config = {ov.properties.inference_num_threads(): 2, ov.properties.hint.enable_cpu_pinning(): True}
    model = core.read_model(model=model_xml_path)
    compiled_model = core.compile_model(model=model, device_name="CPU", config=config)

    # Set up video capture and get video properties
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Downsample 4K resolution videos by 50% for faster processing
    if frame_height == 2160 and frame_width == 3840:
        frame_width //= 2
        frame_height //= 2

    # Prepare output directory and video writer
    output_dir = "./output_predicted_video/"
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, os.path.basename(video_path))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Initialize lists for storing detection results
    image_names = []
    image_labels = []
    bounding_boxes = {'xmin': [], 'ymin': [], 'xmax': [], 'ymax': []}
    dimensions = {'width': [], 'height': []}
    probabilities = []

    # Define the list of labels that the model can detect
    labels = ['money', 'knife', 'monedero', 'pistol', 'smartphone', 'tarjeta']

    # Process each frame of the video
    for frame_idx in tqdm(range(frame_count), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB (model expects RGB input)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Downsample the frame if needed
        if frame_height == 2160 and frame_width == 3840:
            frame = cv2.resize(frame, (frame_width, frame_height))

        # Preprocess the frame: resize, normalize, and transpose for model input
        img_size = 320
        resized_frame = cv2.resize(frame, (img_size, img_size)) / 255.0
        input_tensor = resized_frame.transpose(2, 0, 1).astype(np.float32)[np.newaxis, ...]

        # Run inference on the preprocessed frame
        output = compiled_model(input_tensor)[0]
        detections = output[0]

        # Process detection results
        for detection in detections:
            x1, y1, x2, y2 = (detection[0:4] / img_size) * [frame_width, frame_height, frame_width, frame_height]
            prob = detection[4]
            class_id = int(detection[5])

            if prob > 0.2 and labels[class_id] in ['knife', 'money', 'smartphone']:
                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
                cv2.putText(frame, f"{labels[class_id]} {prob:.2f}", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Save detection details
                image_names.append(f"frame_{frame_idx:05d}")
                image_labels.append(labels[class_id])
                bounding_boxes['xmin'].append(int(x1))
                bounding_boxes['ymin'].append(int(y1))
                bounding_boxes['xmax'].append(int(x2))
                bounding_boxes['ymax'].append(int(y2))
                dimensions['width'].append(int(x2 - x1))
                dimensions['height'].append(int(y2 - y1))
                probabilities.append(prob)

        # Write the annotated frame to the output video
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Release video capture and writer resources
    cap.release()
    out.release()
    print(f"Output video saved at {output_video_path}")

def main():
    """
    Main function to parse arguments and process the video.
    """
    parser = argparse.ArgumentParser(description='Real-Time Video Object Detection using OpenVINO')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--object_model_xml', type=str, required=True, help='Path to the OpenVINO model XML file')
    
    args = parser.parse_args()
    process_video(args.video_path, args.object_model_xml)

if __name__ == "__main__":
    main()
