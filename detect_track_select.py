# import cv2
# import numpy as np
# import tensorflow as tf

# # Load the frozen inference graph
# def load_model(model_path):
#     detection_model = tf.saved_model.load(model_path)
#     return detection_model

# # Perform detection on the image
# def detect_objects(image, detection_model):
#     input_tensor = tf.convert_to_tensor(image)
#     input_tensor = input_tensor[tf.newaxis, ...]

#     detections = detection_model(input_tensor)

#     # All outputs are batches tensors
#     num_detections = int(detections.pop('num_detections'))
#     detections = {key: value[0, :num_detections].numpy()
#                   for key, value in detections.items()}
#     detections['num_detections'] = num_detections

#     # detection_classes should be ints.
#     detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
#     return detections

# # Mouse callback function to select a bounding box
# def select_box(event, x, y, flags, param):
#     global selected_box, box_selected
#     if event == cv2.EVENT_LBUTTONDOWN:
#         for i, bbox in enumerate(boxes):
#             x_min, y_min, w, h = [int(v) for v in bbox]
#             x_max = x_min + w
#             y_max = y_min + h
#             if x_min <= x <= x_max and y_min <= y <= y_max:
#                 selected_box = bbox
#                 box_selected = True
#                 break

# # Path to video
# video_path = "Multilane_Test.mp4"
# model_path = './saved_model'
# output_path = 'output_tracked_click.mp4'

# # Load model
# detection_model = load_model(model_path)

# # Open video file or capture device
# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     print("Error opening video stream or file")
#     exit()

# # Get video properties
# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# # Initialize MultiTracker
# multi_tracker = cv2.legacy.MultiTracker_create()

# # Periodic detection update interval
# detection_interval = 2  # Run detection every 30 frames
# frame_count = 0
# boxes = []
# selected_box = None
# box_selected = False

# # Set mouse callback
# cv2.namedWindow("Object Tracking")
# cv2.setMouseCallback("Object Tracking", select_box)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     if frame_count % detection_interval == 0 or frame_count == 0:
#         # Convert frame to RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Perform object detection
#         detections = detect_objects(frame_rgb, detection_model)

#         # Clear previous trackers if no box selected
#         if not box_selected:
#             multi_tracker = cv2.legacy.MultiTracker_create()

#         # Add new trackers for each detected object
#         boxes = []
#         for i in range(detections['num_detections']):
#             if detections['detection_scores'][i] >= 0.7:
#                 box = detections['detection_boxes'][i]
#                 bbox = (int(box[1] * width), int(box[0] * height), int(box[3] * width) - int(box[1] * width), int(box[2] * height) - int(box[0] * height))
#                 boxes.append(bbox)
#                 if not box_selected:
#                     tracker = cv2.legacy.TrackerKCF_create()
#                     multi_tracker.add(tracker, frame, bbox)
#     else:
#         # Update tracker
#         success, tracked_boxes = multi_tracker.update(frame)
#         if not success:
#             print("Tracking failed. Re-initializing tracker.")
#             frame_count = 0  # Reset frame count to force re-detection

#         # Keep only the selected box if one is selected
#         if box_selected and selected_box is not None:
#             boxes = [selected_box]
#             multi_tracker = cv2.legacy.MultiTracker_create()
#             tracker = cv2.legacy.TrackerKCF_create()
#             multi_tracker.add(tracker, frame, selected_box)

#     # Draw the bounding boxes on the frame
#     for bbox in boxes:
#         x_min, y_min, w, h = [int(v) for v in bbox]
#         x_max = x_min + w
#         y_max = y_min + h
#         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#     # Write the frame with detections to the output video
#     out.write(frame)

#     # Show frame to screen
#     cv2.imshow("Object Tracking", frame)
#     if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
#         break

#     frame_count += 1

# cap.release()
# out.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import tensorflow as tf

# # Load the frozen inference graph
# def load_model(model_path):
#     detection_model = tf.saved_model.load(model_path)
#     return detection_model

# # Perform detection on the image
# def detect_objects(image, detection_model):
#     input_tensor = tf.convert_to_tensor(image)
#     input_tensor = input_tensor[tf.newaxis, ...]

#     detections = detection_model(input_tensor)

#     # All outputs are batches tensors
#     num_detections = int(detections.pop('num_detections'))
#     detections = {key: value[0, :num_detections].numpy()
#                   for key, value in detections.items()}
#     detections['num_detections'] = num_detections

#     # detection_classes should be ints.
#     detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
#     return detections

# # Mouse callback function to select a bounding box
# def select_box(event, x, y, flags, param):
#     global selected_box, box_selected, multi_tracker, frame
#     if event == cv2.EVENT_LBUTTONDOWN:
#         for i, bbox in enumerate(boxes):
#             x_min, y_min, w, h = [int(v) for v in bbox]
#             x_max = x_min + w
#             y_max = y_min + h
#             if x_min <= x <= x_max and y_min <= y <= y_max:
#                 selected_box = bbox
#                 box_selected = True
#                 # Re-initialize tracker with the selected box
#                 multi_tracker = cv2.legacy.MultiTracker_create()
#                 tracker = cv2.legacy.TrackerKCF_create()
#                 multi_tracker.add(tracker, frame, selected_box)
#                 break

# # Path to video
# video_path = "Multilane_Test.mp4"
# model_path = './saved_model'
# output_path = 'output_tracked_selection.mp4'

# # Load model
# detection_model = load_model(model_path)

# # Open video file or capture device
# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     print("Error opening video stream or file")
#     exit()

# # Get video properties
# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# # Initialize MultiTracker
# multi_tracker = cv2.legacy.MultiTracker_create()

# # Periodic detection update interval
# detection_interval = 5  # Run detection every 10 frames
# frame_count = 0
# boxes = []
# selected_box = None
# box_selected = False

# # Set mouse callback
# cv2.namedWindow("Object Tracking")
# cv2.setMouseCallback("Object Tracking", select_box)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     if frame_count % detection_interval == 0 or frame_count == 0:
#         # Convert frame to RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Perform object detection
#         detections = detect_objects(frame_rgb, detection_model)

#         # Clear previous trackers if no box selected
#         if not box_selected:
#             multi_tracker = cv2.legacy.MultiTracker_create()

#         # Add new trackers for each detected object
#         boxes = []
#         for i in range(detections['num_detections']):
#             if detections['detection_scores'][i] >= 0.7:
#                 box = detections['detection_boxes'][i]
#                 bbox = (int(box[1] * width), int(box[0] * height), int(box[3] * width) - int(box[1] * width), int(box[2] * height) - int(box[0] * height))
#                 boxes.append(bbox)
#                 if not box_selected:
#                     tracker = cv2.legacy.TrackerKCF_create()
#                     multi_tracker.add(tracker, frame, bbox)
#     else:
#         # Update tracker
#         success, tracked_boxes = multi_tracker.update(frame)
#         if not success:
#             print("Tracking failed. Re-initializing tracker.")
#             frame_count = 0  # Reset frame count to force re-detection

#         # Keep only the selected box if one is selected
#         if box_selected and selected_box is not None:
#             boxes = [selected_box]
#             multi_tracker = cv2.legacy.MultiTracker_create()
#             tracker = cv2.legacy.TrackerKCF_create()
#             multi_tracker.add(tracker, frame, selected_box)

#     # Draw the bounding boxes on the frame
#     for bbox in boxes:
#         x_min, y_min, w, h = [int(v) for v in bbox]
#         x_max = x_min + w
#         y_max = y_min + h
#         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#     # Write the frame with detections to the output video
#     out.write(frame)

#     # Show frame to screen
#     cv2.imshow("Object Tracking", frame)
#     if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
#         break

#     frame_count += 1

# cap.release()
# out.release()
# cv2.destroyAllWindows()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

# This version is working fine, just after selection of objects cannot reselect them

# import cv2
# import numpy as np
# import tensorflow as tf

# # Load the frozen inference graph
# def load_model(model_path):
#     detection_model = tf.saved_model.load(model_path)
#     return detection_model

# # Perform detection on the image
# def detect_objects(image, detection_model):
#     input_tensor = tf.convert_to_tensor(image)
#     input_tensor = input_tensor[tf.newaxis, ...]

#     detections = detection_model(input_tensor)

#     # All outputs are batches tensors
#     num_detections = int(detections.pop('num_detections'))
#     detections = {key: value[0, :num_detections].numpy()
#                   for key, value in detections.items()}
#     detections['num_detections'] = num_detections

#     # detection_classes should be ints.
#     detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
#     return detections

# # Mouse callback function to select a bounding box
# def select_box(event, x, y, flags, param):
#     global selected_box, box_selected, tracker, frame
#     if event == cv2.EVENT_LBUTTONDOWN:
#         for i, bbox in enumerate(boxes):
#             x_min, y_min, w, h = [int(v) for v in bbox]
#             x_max = x_min + w
#             y_max = y_min + h
#             if x_min <= x <= x_max and y_min <= y <= y_max:
#                 selected_box = bbox
#                 box_selected = True
#                 # Re-initialize tracker with the selected box
#                 tracker = cv2.TrackerCSRT_create()
#                 tracker.init(frame, tuple(selected_box))
#                 break

# # Path to video
# video_path = "ML_Highway.mp4"
# model_path = './saved_model'
# output_path = 'output_tracked_ML_Highway.mp4'

# # Load model
# detection_model = load_model(model_path)

# # Open video file or capture device
# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     print("Error opening video stream or file")
#     exit()

# # Get video properties
# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(f"The value of fps:: {fps}")

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# # Initialize CSRT Tracker
# tracker = cv2.TrackerCSRT_create()

# # Periodic detection update interval
# detection_interval = 2  # Run detection every 2 frames
# frame_count = 0
# boxes = []
# selected_box = None
# box_selected = False

# # Set mouse callback
# cv2.namedWindow("Object Tracking")
# cv2.setMouseCallback("Object Tracking", select_box)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     if frame_count % detection_interval == 0 or frame_count == 0:
#         # Convert frame to RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Perform object detection
#         detections = detect_objects(frame_rgb, detection_model)

#         # Clear previous boxes
#         boxes = []
#         for i in range(detections['num_detections']):
#             if detections['detection_scores'][i] >= 0.7:
#                 box = detections['detection_boxes'][i]
#                 bbox = (int(box[1] * width), int(box[0] * height), int(box[3] * width) - int(box[1] * width), int(box[2] * height) - int(box[0] * height))
#                 boxes.append(bbox)
                
#     if box_selected and selected_box is not None:
#         # Update the tracker
#         success, tracked_box = tracker.update(frame)
#         if success:
#             x, y, w, h = [int(v) for v in tracked_box]
#             selected_box = (x, y, w, h)
#             boxes = [selected_box]
#         else:
#             print("Tracking failed. Re-initializing tracker.")
#             box_selected = False
#             frame_count = 0  # Reset frame count to force re-detection

#     # Draw the bounding boxes on the frame
#     for bbox in boxes:
#         x_min, y_min, w, h = [int(v) for v in bbox]
#         x_max = x_min + w
#         y_max = y_min + h
#         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#     # Write the frame with detections to the output video
#     out.write(frame)

#     # Show frame to screen
#     cv2.imshow("Object Tracking", frame)
#     if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
#         break

#     frame_count += 1

# cap.release()
# out.release()
# cv2.destroyAllWindows()

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
                    # Re-initialize tracker with the selected box
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, tuple(selected_box))
                    break

# Path to video
video_path = "Multilane_Test.mp4"
model_path = './saved_model'
output_path = 'output_tracked_Multilane.mp4'

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
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Initialize CSRT Tracker
tracker = cv2.TrackerCSRT_create()

# Periodic detection update interval
detection_interval = 2  # Run detection every 2 frames
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
            if detections['detection_scores'][i] >= 0.5:
                box = detections['detection_boxes'][i]
                bbox = (int(box[1] * width), int(box[0] * height), int(box[3] * width) - int(box[1] * width), int(box[2] * height) - int(box[0] * height))
                boxes.append(bbox)
        detection_mode = False
                
    if box_selected and selected_box is not None and tracking_enabled:
        # Update the tracker
        success, tracked_box = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in tracked_box]
            selected_box = (x, y, w, h)
            boxes = [selected_box]
        else:
            print("Tracking failed. Re-initializing tracker.")
            box_selected = False
            frame_count = 0  # Reset frame count to force re-detection

    # Draw the bounding boxes on the frame
    for bbox in boxes:
        x_min, y_min, w, h = [int(v) for v in bbox]
        x_max = x_min + w
        y_max = y_min + h
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Write the frame with detections to the output video
    out.write(frame)

    # Show frame to screen
    cv2.imshow("Object Tracking", frame)
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()
