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
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detection_model(input_tensor)

    # All outputs are batches tensors
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    return detections

# Mouse callback function to select a bounding box
def select_box(event, x, y, flags, param):
    global selected_box, box_selected, tracker, frame, tracking_enabled, detection_mode
    if event == cv2.EVENT_LBUTTONDOWN:
        if tracking_enabled:
            detection_mode = True
            tracking_enabled = False
            box_selected = False
        else:
            for i, bbox in enumerate(boxes):
                x_min, y_min, w, h = [int(v) for v in bbox]
                x_max = x_min + w
                y_max = y_min + h
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    selected_box = bbox
                    box_selected = True
                    tracking_enabled = True
                    detection_mode = False
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, tuple(selected_box))
                    break

# Path to video
video_path = "Test_Data/ML_Highway.mp4"
model_path = './saved_model'
output_path = 'Kalman_Results/output_tracked_hopeful.mp4'

# Load model
detection_model = load_model(model_path)

# Open video file or capture device
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Periodic detection update interval
detection_interval = 5  # Run detection every 10 frames
frame_count = 0
boxes = []
selected_box = None
box_selected = False
tracking_enabled = False
detection_mode = True

# Set mouse callback
cv2.namedWindow("Object Tracking")
cv2.setMouseCallback("Object Tracking", select_box)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % detection_interval == 0 or frame_count == 0 or detection_mode:
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection
        detections = detect_objects(frame_rgb, detection_model)

        # Clear previous boxes
        boxes = []
        for i in range(detections['num_detections']):
            if detections['detection_scores'][i] >= 0.8:
                box = detections['detection_boxes'][i]
                bbox = (int(box[1] * width), int(box[0] * height), int((box[3] - box[1]) * width), int((box[2] - box[0]) * height))
                boxes.append(bbox)
        detection_mode = False

    if tracking_enabled:
        # Update the selected tracker
        success, tracked_box = tracker.update(frame)
        if success:
            selected_box = (int(tracked_box[0]), int(tracked_box[1]), int(tracked_box[2]), int(tracked_box[3]))
        else:
            print("Tracking failed. Re-initializing tracker.")
            tracking_enabled = False
            detection_mode = True
            frame_count = 0  # Reset frame count to force re-detection

    # Draw the bounding boxes on the frame
    for bbox in boxes:
        if tracking_enabled and bbox == selected_box:
            color = (0, 255, 0)  # Green for selected box
        else:
            color = (255, 0, 0)  # Blue for others
        x_min, y_min, w, h = [int(v) for v in bbox]
        x_max = x_min + w
        y_max = y_min + h
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

    if tracking_enabled:
        x_min, y_min, w, h = [int(v) for v in selected_box]
        x_max = x_min + w
        y_max = y_min + h
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Write the frame with detections to the output video
    # out.write(frame)

    # Show frame to screen
    cv2.imshow("Object Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
# out.release()
cv2.destroyAllWindows()
