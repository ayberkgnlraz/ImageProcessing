import cv2
import numpy as np

# Loading our image
image = cv2.imread('python_ayberk\image.jpg')

# Converting the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Applying GaussianBlur to reduce noise and improve contour detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# edge detection (Canny) to find contours
edges = cv2.Canny(blurred, 50, 150)

# HoughLines method to detect lines in the image
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

# Creating a separate image to draw rectangles and text
output_frame = image.copy()

# Keeping track of detected object dimensions
detected_objects = []

# Storing lines for distance calculation
all_lines = []

# Iterating over all detected lines
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    # Drawing lines on the separate image
    cv2.line(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Storing lines for distance calculation
    all_lines.append(((x1, y1), (x2, y2)))

# Using HoughLinesP method to detect more accurate lines
lines_p = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Iterating over all detected lines
for line in lines_p:
    x1, y1, x2, y2 = line[0]
    cv2.line(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Storing lines for distance calculation
    all_lines.append(((x1, y1), (x2, y2)))

# Seting a threshold for the minimum distance between lines
min_distance_threshold = 50

# Seting a threshold for checking if lines are approximately parallel
parallel_threshold = 0.1  

# Calculating the distance between parallel lines
distances = []
for i in range(len(all_lines)):
    for j in range(i + 1, len(all_lines)):
        # Extract endpoints of lines
        (x1, y1), (x2, y2) = all_lines[i]
        (x3, y3), (x4, y4) = all_lines[j]

        # Calculating the angle between the two lines
        angle_diff = np.abs(np.arctan2(y2 - y1, x2 - x1) - np.arctan2(y4 - y3, x4 - x3))

        # Checking if the lines are approximately parallel
        if angle_diff < parallel_threshold:
            # Calculating the distance between lines using the distance formula
            distance = np.abs((y2 - y1) * x3 - (x2 - x1) * y3 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

            # Checking if the distance is above the threshold
            if distance > min_distance_threshold:
                distances.append(distance)

# Displaying the distances on the image and in the terminal
for i, distance in enumerate(distances):
    cv2.putText(output_frame, f"Distance {i + 1}: {distance:.2f} pixels", (10, 30 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    print(f"Distance {i + 1}: {distance:.2f} pixels")

# Displaying the combined image with detected objects
cv2.imshow('Detected Objects', output_frame)

# saving the new image when we press "s"
key = cv2.waitKey(0)
if key == ord('s'):
   
    cv2.imwrite('output_image.jpg', output_frame)

cv2.destroyAllWindows()






























