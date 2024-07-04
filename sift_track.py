# # import necessary libraries
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # path to video
# video_path = "Multilane_Test.mp4"
# video = cv2.VideoCapture(video_path)

# # read only first frame for drawing rectangle for desired object
# ret, frame = video.read()

# # initialize the coordinates with big random numbers
# x_min, y_min, x_max, y_max = 36000, 36000, 0, 0


# def coordinate_chooser(event, x, y, flags, param):
#     global x_min, y_min, x_max, y_max

#     # when you click the right button, it will provide coordinates for variables
#     if event == cv2.EVENT_RBUTTONDOWN:
#         # update x_min and y_min
#         x_min = min(x, x_min)
#         y_min = min(y, y_min)
#         # update x_max and y_max
#         x_max = max(x, x_max)
#         y_max = max(y, y_max)
#         # draw rectangle
#         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

#     # reset coordinates with middle button of your mouse
#     if event == cv2.EVENT_MBUTTONDOWN:
#         print("reset coordinate data")
#         x_min, y_min, x_max, y_max = 36000, 36000, 0, 0


# cv2.namedWindow('coordinate_screen')
# # Set mouse handler for the specified window
# cv2.setMouseCallback('coordinate_screen', coordinate_chooser)

# while True:
#     cv2.imshow("coordinate_screen", frame)  # show only the first frame
#     k = cv2.waitKey(5) & 0xFF  # press ESC to exit
#     if k == 27:
#         cv2.destroyAllWindows()
#         break

# # take region of interest (inside of rectangle)
# roi_image = frame[y_min:y_max, x_min:x_max]

# # convert roi to gray scale
# roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

# # create SIFT algorithm object
# sift = cv2.SIFT_create()

# # find roi's keypoints and descriptors
# keypoints_1, descriptors_1 = sift.detectAndCompute(roi_gray, None)

# # draw keypoints on the roi image
# roi_keypoint_image = cv2.drawKeypoints(roi_gray, keypoints_1, roi_gray)

# # visualize keypoints
# plt.subplot(121)
# plt.imshow(roi_gray, cmap="gray")
# plt.subplot(122)
# plt.imshow(roi_keypoint_image, cmap="gray")
# plt.show()

# # path to video
# video = cv2.VideoCapture(video_path)

# # matcher object
# bf = cv2.BFMatcher()

# while True:
#     # reading video
#     ret, frame = video.read()
#     if not ret:
#         break

#     # convert frame to gray scale
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # find current frame's keypoints and descriptors
#     keypoints_2, descriptors_2 = sift.detectAndCompute(frame_gray, None)

#     # compare the keypoints/descriptors extracted from the first frame with those from the current frame
#     matches = bf.match(descriptors_1, descriptors_2)

#     for match in matches:
#         # .queryIdx and .trainIdx give index for keypoints
#         query_idx = match.queryIdx  # keypoint index from target image
#         train_idx = match.trainIdx  # keypoint index from current frame

#         # take coordinates that match
#         pt1 = keypoints_1[query_idx].pt
#         pt2 = keypoints_2[train_idx].pt

#         # draw circle to pt2 coordinates (current frame coordinates)
#         cv2.circle(frame, (int(pt2[0]), int(pt2[1])), 2, (255, 0, 0), 2)

#     # show frame to screen
#     cv2.imshow("coordinate_screen", frame)
#     k = cv2.waitKey(5) & 0xFF  # press ESC to exit
#     if k == 27:
#         break

# cv2.destroyAllWindows()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to video
video_path = "Multilane_Test.mp4"
video = cv2.VideoCapture(video_path)

# Read only the first frame for drawing rectangles for desired objects
ret, frame = video.read()
if not ret:
    print("Failed to read the video.")
    exit()

# List to store the coordinates of multiple ROIs
rois = []

# Temporary variables for a single ROI
x_min, y_min, x_max, y_max = 36000, 36000, 0, 0

def coordinate_chooser(event, x, y, flags, param):
    global x_min, y_min, x_max, y_max, frame, rois

    # When you click the right button, it will provide coordinates for variables
    if event == cv2.EVENT_RBUTTONDOWN:
        # Update x_min and y_min
        x_min = min(x, x_min)
        y_min = min(y, y_min)
        # Update x_max and y_max
        x_max = max(x, x_max)
        y_max = max(y, y_max)
        # Draw rectangle
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

    # Reset coordinates with middle button of your mouse
    if event == cv2.EVENT_MBUTTONDOWN:
        print("Reset coordinate data")
        x_min, y_min, x_max, y_max = 36000, 36000, 0, 0

    # Finalize the current ROI and reset temporary variables with left button of your mouse
    if event == cv2.EVENT_LBUTTONDOWN:
        if x_min < x_max and y_min < y_max:
            rois.append((x_min, y_min, x_max, y_max))
        print(f"ROIs: {rois}")
        x_min, y_min, x_max, y_max = 36000, 36000, 0, 0

cv2.namedWindow('coordinate_screen')
# Set mouse handler for the specified window
cv2.setMouseCallback('coordinate_screen', coordinate_chooser)

while True:
    cv2.imshow("coordinate_screen", frame)  # show only the first frame
    k = cv2.waitKey(5) & 0xFF  # press ESC to exit
    if k == 27:
        break

cv2.destroyAllWindows()

# Initialize SIFT algorithm object
sift = cv2.SIFT_create()

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame.shape[1], frame.shape[0]))

# List to store keypoints and descriptors of multiple ROIs
keypoints_descriptors = []

# Process each ROI
for (x_min, y_min, x_max, y_max) in rois:
    # Take region of interest (inside of rectangle)
    roi_image = frame[y_min:y_max, x_min:x_max]
    # Convert ROI to gray scale
    roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    # Find ROI's keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(roi_gray, None)
    keypoints_descriptors.append((keypoints, descriptors))

    # Draw keypoints on the ROI image
    roi_keypoint_image = cv2.drawKeypoints(roi_gray, keypoints, roi_gray)

    # Visualize keypoints
    plt.subplot(121)
    plt.imshow(roi_gray, cmap="gray")
    plt.subplot(122)
    plt.imshow(roi_keypoint_image, cmap="gray")
    plt.show()

# Matcher object
bf = cv2.BFMatcher()

# Reset video to start
video = cv2.VideoCapture(video_path)

while True:
    # Read video frame
    ret, frame = video.read()
    if not ret:
        break

    # Convert frame to gray scale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Process each ROI
    for keypoints_1, descriptors_1 in keypoints_descriptors:
        # Find current frame's keypoints and descriptors
        keypoints_2, descriptors_2 = sift.detectAndCompute(frame_gray, None)

        # Compare the keypoints/descriptors extracted from the first frame with those from the current frame
        matches = bf.match(descriptors_1, descriptors_2)

        for match in matches:
            # .queryIdx and .trainIdx give index for keypoints
            query_idx = match.queryIdx  # keypoint index from target image
            train_idx = match.trainIdx  # keypoint index from current frame

            # Take coordinates that match
            pt1 = keypoints_1[query_idx].pt
            pt2 = keypoints_2[train_idx].pt

            # Draw circle to pt2 coordinates (current frame coordinates)
            cv2.circle(frame, (int(pt2[0]), int(pt2[1])), 2, (255, 0, 0), 2)

    # Show frame to screen
    cv2.imshow("coordinate_screen", frame)
    # Write frame to output video
    out.write(frame)

    k = cv2.waitKey(5) & 0xFF  # press ESC to exit
    if k == 27:
        break

cv2.destroyAllWindows()
video.release()
out.release()
