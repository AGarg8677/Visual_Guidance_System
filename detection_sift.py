# Try 1

# import cv2
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt

# # Load the frozen inference graph
# def load_model(model_path):
#     detection_model = tf.saved_model.load(model_path)
#     return detection_model

# # Perform detection on the image
# def detect_objects(image, detection_model):
#     input_tensor = tf.convert_to_tensor(image)
#     input_tensor = input_tensor[tf.newaxis,...]

#     detections = detection_model(input_tensor)

#     # All outputs are batches tensors
#     num_detections = int(detections.pop('num_detections'))
#     detections = {key: value[0, :num_detections].numpy()
#                   for key, value in detections.items()}
#     detections['num_detections'] = num_detections

#     # detection_classes should be ints.
#     detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
#     return detections

# # Visualize the detection results on the image
# def visualize_detection(image, detections, threshold=0.7):
#     class_names = {1: 'Car', 2: 'Truck', 3: 'Bus'}
#     h, w, _ = image.shape
#     for i in range(detections['num_detections']):
#         if detections['detection_scores'][i] >= threshold:
#             box = detections['detection_boxes'][i]
#             y1, x1, y2, x2 = box[0] * h, box[1] * w, box[2] * h, box[3] * w
#             cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             class_id = detections["detection_classes"][i]
#             label = f'{class_names.get(class_id, "Unknown")}: {detections["detection_scores"][i]:.2f}'
#             cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     return image

# # Path to video
# video_path = "Multilane_Test.mp4"
# model_path = './saved_model'
# # output_path = 'output_tracked.mp4'

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
# # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# # out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# # Read only the first frame for object detection
# ret, frame = cap.read()
# if not ret:
#     print("Failed to read the video")
#     cap.release()
#     # out.release()
#     exit()

# # Convert frame to RGB
# frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# # Perform object detection
# detections = detect_objects(frame_rgb, detection_model)

# # Get bounding box of the first detected object
# bbox = None
# for i in range(detections['num_detections']):
#     if detections['detection_scores'][i] >= 0.7:
#         box = detections['detection_boxes'][i]
#         bbox = (int(box[1] * width), int(box[0] * height), int(box[3] * width), int(box[2] * height))
#         break

# if bbox is None:
#     print("No object detected.")
#     cap.release()
#     # out.release()
#     exit()

# x_min, y_min, x_max, y_max = bbox

# # Draw the bounding box on the first frame
# cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# # Take region of interest (inside of rectangle)
# roi_image = frame[y_min:y_max, x_min:x_max]

# # Convert ROI to gray scale
# roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

# # Create SIFT algorithm object
# sift = cv2.SIFT_create()

# # Find ROI's keypoints and descriptors
# keypoints_1, descriptors_1 = sift.detectAndCompute(roi_gray, None)

# # Draw keypoints on the ROI image
# roi_keypoint_image = cv2.drawKeypoints(roi_gray, keypoints_1, None)

# # Visualize keypoints
# plt.subplot(121)
# plt.imshow(roi_gray, cmap="gray")
# plt.subplot(122)
# plt.imshow(roi_keypoint_image, cmap="gray")
# plt.show()

# # Matcher object
# bf = cv2.BFMatcher()

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert frame to gray scale
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Find current frame's keypoints and descriptors
#     keypoints_2, descriptors_2 = sift.detectAndCompute(frame_gray, None)

#     # Compare the keypoints/descriptors extracted from the first frame with those from the current frame
#     matches = bf.match(descriptors_1, descriptors_2)

#     for match in matches:
#         # .queryIdx and .trainIdx give index for keypoints
#         query_idx = match.queryIdx  # keypoint index from target image
#         train_idx = match.trainIdx  # keypoint index from current frame

#         # Take coordinates that match
#         pt1 = keypoints_1[query_idx].pt
#         pt2 = keypoints_2[train_idx].pt

#         # Draw circle to pt2 coordinates (current frame coordinates)
#         cv2.circle(frame, (int(pt2[0]), int(pt2[1])), 2, (255, 0, 0), 2)

#     # Visualize the detection results on the image
#     result_frame = visualize_detection(frame, detections)

#     # Write the frame with detections to the output video
#     # out.write(result_frame)

#     # Show frame to screen
#     cv2.imshow("Object Tracking", result_frame)
#     if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
#         break

# cap.release()
# # out.release()
# cv2.destroyAllWindows()

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Try 2
# import cv2
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt

# # Load the frozen inference graph
# def load_model(model_path):
#     detection_model = tf.saved_model.load(model_path)
#     return detection_model

# # Perform detection on the image
# def detect_objects(image, detection_model):
#     input_tensor = tf.convert_to_tensor(image)
#     input_tensor = input_tensor[tf.newaxis,...]

#     detections = detection_model(input_tensor)

#     # All outputs are batches tensors
#     num_detections = int(detections.pop('num_detections'))
#     detections = {key: value[0, :num_detections].numpy()
#                   for key, value in detections.items()}
#     detections['num_detections'] = num_detections

#     # detection_classes should be ints.
#     detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
#     return detections

# # Visualize the detection results on the image
# def visualize_detection(image, detections, threshold=0.7):
#     class_names = {1: 'Car', 2: 'Truck', 3: 'Bus'}
#     h, w, _ = image.shape
#     for i in range(detections['num_detections']):
#         if detections['detection_scores'][i] >= threshold:
#             box = detections['detection_boxes'][i]
#             y1, x1, y2, x2 = box[0] * h, box[1] * w, box[2] * h, box[3] * w
#             cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             class_id = detections["detection_classes"][i]
#             label = f'{class_names.get(class_id, "Unknown")}: {detections["detection_scores"][i]:.2f}'
#             cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     return image

# # Path to video
# video_path = "Multilane_Test.mp4"
# model_path = './saved_model'
# # output_path = 'output_tracked.mp4'

# # Load model
# detection_model = load_model(model_path)

# # Open video file or capture device
# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     print("Error opening video stream or file")
#     exit()

# # Get video properties
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Define the codec and create VideoWriter object
# # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# # out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# # Read only the first frame for object detection
# ret, frame = cap.read()
# if not ret:
#     print("Failed to read the video")
#     cap.release()
#     # out.release()
#     exit()

# # Convert frame to RGB
# frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# # Perform object detection
# detections = detect_objects(frame_rgb, detection_model)

# # Get bounding box of the first detected object
# bbox = None
# for i in range(detections['num_detections']):
#     if detections['detection_scores'][i] >= 0.7:
#         box = detections['detection_boxes'][i]
#         bbox = (int(box[1] * width), int(box[0] * height), int(box[3] * width), int(box[2] * height))
#         break

# if bbox is None:
#     print("No object detected.")
#     cap.release()
#     # out.release()
#     exit()

# x_min, y_min, x_max, y_max = bbox

# # Draw the bounding box on the first frame
# cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# # # Initialize the tracker with the first frame and bounding box
# # tracker = cv2.legacy.TrackerCSRT_create()
# # tracker.init(frame, (x_min, y_min, x_max - x_min, y_max - y_min))

# # Initialize the tracker with the first frame and bounding box
# tracker = cv2.legacy.TrackerMedianFlow_create()
# tracker.init(frame, (x_min, y_min, x_max - x_min, y_max - y_min))

# # SIFT feature matching
# # Convert ROI to gray scale
# roi_image = frame[y_min:y_max, x_min:x_max]
# roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
# sift = cv2.SIFT_create()
# keypoints_1, descriptors_1 = sift.detectAndCompute(roi_gray, None)
# roi_keypoint_image = cv2.drawKeypoints(roi_gray, keypoints_1, None)
# plt.subplot(121)
# plt.imshow(roi_gray, cmap="gray")
# plt.subplot(122)
# plt.imshow(roi_keypoint_image, cmap="gray")
# plt.show()
# bf = cv2.BFMatcher()

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Update tracker
#     success, bbox = tracker.update(frame)
#     if success:
#         x_min, y_min, w, h = [int(v) for v in bbox]
#         x_max = x_min + w
#         y_max = y_min + h
#         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#         # Convert ROI to gray scale
#         roi_image = frame[y_min:y_max, x_min:x_max]
#         roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

#         # SIFT feature matching
#         keypoints_2, descriptors_2 = sift.detectAndCompute(roi_gray, None)
#         matches = bf.match(descriptors_1, descriptors_2)
#         for match in matches:
#             pt1 = keypoints_1[match.queryIdx].pt
#             pt2 = keypoints_2[match.trainIdx].pt
#             cv2.circle(frame, (int(pt2[0] + x_min), int(pt2[1] + y_min)), 2, (255, 0, 0), 2)

#     # Visualize the detection results on the image
#     result_frame = visualize_detection(frame, detections)

#     # Write the frame with detections to the output video
#     # out.write(result_frame)

#     # Show frame to screen
#     cv2.imshow("Object Tracking", result_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# # out.release()
# cv2.destroyAllWindows()

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Try 3

# import cv2
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt

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

# # Visualize the detection results on the image
# def visualize_detection(image, detections, threshold=0.7):
#     class_names = {1: 'Car', 2: 'Truck', 3: 'Bus'}
#     h, w, _ = image.shape
#     for i in range(detections['num_detections']):
#         if detections['detection_scores'][i] >= threshold:
#             box = detections['detection_boxes'][i]
#             y1, x1, y2, x2 = box[0] * h, box[1] * w, box[2] * h, box[3] * w
#             cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             class_id = detections["detection_classes"][i]
#             label = f'{class_names.get(class_id, "Unknown")}: {detections["detection_scores"][i]:.2f}'
#             cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     return image

# # Path to video
# video_path = "Multilane_Test.mp4"
# model_path = './saved_model'
# # output_path = 'output_tracked.mp4'

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
# # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# # out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# # Read only the first frame for object detection
# ret, frame = cap.read()
# if not ret:
#     print("Failed to read the video")
#     cap.release()
#     # out.release()
#     exit()

# # Convert frame to RGB
# frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# # Perform object detection
# detections = detect_objects(frame_rgb, detection_model)

# # Get bounding box of the first detected object
# bbox = None
# for i in range(detections['num_detections']):
#     if detections['detection_scores'][i] >= 0.7:
#         box = detections['detection_boxes'][i]
#         bbox = (int(box[1] * width), int(box[0] * height), int(box[3] * width), int(box[2] * height))
#         break

# if bbox is None:
#     print("No object detected.")
#     cap.release()
#     # out.release()
#     exit()

# x_min, y_min, x_max, y_max = bbox

# # Draw the bounding box on the first frame
# cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# # Initialize the tracker with the first frame and bounding box
# tracker = cv2.legacy.TrackerKCF_create()
# tracker.init(frame, (x_min, y_min, x_max - x_min, y_max - y_min))

# # Periodic detection update interval
# detection_interval = 30  # Run detection every 30 frames
# frame_count = 0

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Update tracker
#     success, bbox = tracker.update(frame)
#     if success:
#         x_min, y_min, w, h = [int(v) for v in bbox]
#         x_max = x_min + w
#         y_max = y_min + h
#         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#     else:
#         print("Tracking failed.")

#     # Periodically run object detection
#     if frame_count % detection_interval == 0:
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         detections = detect_objects(frame_rgb, detection_model)
#         # Get bounding box of the first detected object
#         bbox = None
#         for i in range(detections['num_detections']):
#             if detections['detection_scores'][i] >= 0.7:
#                 box = detections['detection_boxes'][i]
#                 bbox = (int(box[1] * width), int(box[0] * height), int(box[3] * width), int(box[2] * height))
#                 break
#         if bbox:
#             x_min, y_min, x_max, y_max = bbox
#             tracker.init(frame, (x_min, y_min, x_max - x_min, y_max - y_min))

#     # Visualize the detection results on the image
#     result_frame = visualize_detection(frame, detections)

#     # Write the frame with detections to the output video
#     # out.write(result_frame)

#     # Show frame to screen
#     cv2.imshow("Object Tracking", result_frame)
#     if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
#         break

#     frame_count += 1

# cap.release()
# # out.release()
# cv2.destroyAllWindows()

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Try 4

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

# # Visualize the detection results on the image
# def visualize_detection(image, detections, threshold=0.7):
#     class_names = {1: 'Car', 2: 'Truck', 3: 'Bus'}
#     h, w, _ = image.shape
#     for i in range(detections['num_detections']):
#         if detections['detection_scores'][i] >= threshold:
#             box = detections['detection_boxes'][i]
#             y1, x1, y2, x2 = box[0] * h, box[1] * w, box[2] * h, box[3] * w
#             cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             class_id = detections["detection_classes"][i]
#             label = f'{class_names.get(class_id, "Unknown")}: {detections["detection_scores"][i]:.2f}'
#             cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     return image

# # Path to video
# video_path = "Multilane_Test.mp4"
# model_path = './saved_model'

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

# # Read the first frame for object detection
# ret, frame = cap.read()
# if not ret:
#     print("Failed to read the video")
#     cap.release()
#     exit()

# # Convert frame to RGB
# frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# # Perform object detection
# detections = detect_objects(frame_rgb, detection_model)

# # Get bounding box of the first detected object
# bbox = None
# for i in range(detections['num_detections']):
#     if detections['detection_scores'][i] >= 0.7:
#         box = detections['detection_boxes'][i]
#         bbox = (int(box[1] * width), int(box[0] * height), int(box[3] * width), int(box[2] * height))
#         break

# if bbox is None:
#     print("No object detected.")
#     cap.release()
#     exit()

# # Initialize the tracker with the first frame and bounding box
# tracker = cv2.legacy.TrackerCSRT_create()
# tracker.init(frame, bbox)

# # Periodic detection update interval
# detection_interval = 30  # Run detection every 30 frames
# frame_count = 0

# # Process the video frames
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Update tracker
#     success, bbox = tracker.update(frame)
#     if success:
#         x_min, y_min, w, h = [int(v) for v in bbox]
#         x_max = x_min + w
#         y_max = y_min + h
#         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#     else:
#         print("Tracking failed. Re-initializing tracker.")
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         detections = detect_objects(frame_rgb, detection_model)
#         bbox = None
#         for i in range(detections['num_detections']):
#             if detections['detection_scores'][i] >= 0.7:
#                 box = detections['detection_boxes'][i]
#                 bbox = (int(box[1] * width), int(box[0] * height), int(box[3] * width), int(box[2] * height))
#                 break
#         if bbox:
#             tracker = cv2.legacy.TrackerCSRT_create()
#             tracker.init(frame, bbox)
#             frame_count = 0  # Reset frame count after re-initialization

#     # Periodically run object detection
#     if frame_count % detection_interval == 0:
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         detections = detect_objects(frame_rgb, detection_model)
#         # Get bounding box of the first detected object
#         bbox = None
#         for i in range(detections['num_detections']):
#             if detections['detection_scores'][i] >= 0.4:
#                 box = detections['detection_boxes'][i]
#                 bbox = (int(box[1] * width), int(box[0] * height), int(box[3] * width), int(box[2] * height))
#                 break
#         if bbox:
#             tracker.init(frame, bbox)

#     # Visualize the detection results on the image
#     result_frame = visualize_detection(frame, detections)

#     # Show frame to screen
#     cv2.imshow("Object Tracking", result_frame)
#     if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
#         break

#     frame_count += 1

# cap.release()
# cv2.destroyAllWindows()

# Try 5
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

# Path to video
video_path = "Yellow_Taxi.mp4"
model_path = './saved_model'
output_path = 'output_track_latest_3.mp4'

# Load model
detection_model = load_model(model_path)

# Open video file or capture device
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Initialize MultiTracker
multi_tracker = cv2.legacy.MultiTracker_create()

# Periodic detection update interval
detection_interval = 20  # Run detection every 30 frames
frame_count = 0
boxes = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % detection_interval == 0 or frame_count == 0:
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection
        detections = detect_objects(frame_rgb, detection_model)

        # Clear previous trackers
        multi_tracker = cv2.legacy.MultiTracker_create()

        # Add new trackers for each detected object
        for i in range(detections['num_detections']):
            if detections['detection_scores'][i] >= 0.5:
                box = detections['detection_boxes'][i]
                bbox = (int(box[1] * width), int(box[0] * height), int(box[3] * width) - int(box[1] * width), int(box[2] * height) - int(box[0] * height))
                tracker = cv2.legacy.TrackerCSRT_create()
                multi_tracker.add(tracker, frame, bbox)
    else:
        # Update tracker
        success, boxes = multi_tracker.update(frame)
        if not success:
            print("Tracking failed. Re-initializing tracker.")
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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()

