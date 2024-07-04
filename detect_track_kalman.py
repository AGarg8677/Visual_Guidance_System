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

# Initialize Kalman filter
def initialize_kalman():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]], np.float32) * 0.03
    return kalman

# Update Kalman filter
def update_kalman(kalman, bbox):
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
    kalman.correct(measurement)
    prediction = kalman.predict()
    predicted_bbox = (int(prediction[0] - w / 2), int(prediction[1] - h / 2), w, h)
    print(predicted_bbox)
    return predicted_bbox

# Path to video
video_path = "Test_Data/Multilane_Test.mp4"
model_path = './saved_model'
output_path = 'Kalman_Results/output_tracked_kalman.mp4'

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
detection_interval = 10  # Run detection every 10 frames
frame_count = 0
boxes = []
kalman_filters = []
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

        # Clear previous boxes and Kalman filters
        boxes = []
        kalman_filters = []
        for i in range(detections['num_detections']):
            if detections['detection_scores'][i] >= 0.4:
                box = detections['detection_boxes'][i]
                bbox = (int(box[1] * width), int(box[0] * height), int((box[3] - box[1]) * width), int((box[2] - box[0]) * height))
                boxes.append(bbox)
                kalman_filters.append(initialize_kalman())
        detection_mode = False

    if tracking_enabled and selected_box is not None:
        # Update the selected tracker
        success, tracked_box = tracker.update(frame)
        if success:
            selected_box = tracked_box

    # Update all Kalman filters and draw boxes
    for i, bbox in enumerate(boxes):
        kalman_filters[i].correct(np.array([[np.float32(bbox[0] + bbox[2] / 2)], [np.float32(bbox[1] + bbox[3] / 2)]]))
        prediction = kalman_filters[i].predict()
        x, y, w, h = bbox
        pred_bbox = (int(prediction[0] - w / 2), int(prediction[1] - h / 2), w, h)
        boxes[i] = pred_bbox
        if not tracking_enabled or bbox != selected_box:
            cv2.rectangle(frame, (pred_bbox[0], pred_bbox[1]), (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (255, 0, 0), 2)

    if tracking_enabled and selected_box is not None:
        x, y, w, h = [int(v) for v in selected_box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show frame to screen
    cv2.imshow("Object Tracking", frame)
    # out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
# out.release()
cv2.destroyAllWindows()
