import cv2
import numpy as np
import tensorflow as tf

# Load the frozen inference graph
def load_model(model_path):
    detection_model = tf.saved_model.load(model_path)
    return detection_model

# Perform detection on the image
def detect_objects(image, detection_model):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]

    detections = detection_model(input_tensor)

    # All outputs are batches tensors
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    return detections

# Visualize the detection results on the image
def visualize_detection(image, detections, threshold=0.7):
    class_names = {1: 'Car', 2: 'Truck', 3: 'Bus'}
    h, w, _ = image.shape
    for i in range(detections['num_detections']):
        if detections['detection_scores'][i] >= threshold:
            box = detections['detection_boxes'][i]
            y1, x1, y2, x2 = box[0] * h, box[1] * w, box[2] * h, box[3] * w
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            class_id = detections["detection_classes"][i]
            label = f'{class_names.get(class_id, "Unknown")}: {detections["detection_scores"][i]:.2f}'
            cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Main function to perform object detection on video and write results to a video file
def main(video_path, model_path, output_path):
    # Load model
    detection_model = load_model(model_path)

    # Open video file or capture device
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection
        detections = detect_objects(frame_rgb, detection_model)

        # Visualize the results
        result_frame = visualize_detection(frame, detections)

        # Write the frame with detections to the output video
        out.write(result_frame)

        # Display the frame
        cv2.imshow('Object Detection', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'Test_LA.mp4'  # Replace with your video file path or use 0 for webcam
    model_path = './saved_model'  # Path to your saved model directory
    output_path = 'mobilenet_v2_LA.mp4'  # Output video file path
    main(video_path, model_path, output_path)
