import cv2
import numpy as np

# Load the image
image = cv2.imread('python_ayberk\shapess.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and improve contour detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use edge detection (Canny) to find contours
edges = cv2.Canny(blurred, 50, 150)

# Use HoughLines method to detect lines in the image
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

# Create a separate image to draw rectangles and text
output_frame = image.copy()

# Keep track of detected object dimensions
detected_objects = []

# Store lines for distance calculation
all_lines = []

# Iterate over all detected lines
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

    # Draw lines on the separate image
    cv2.line(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Store lines for distance calculation
    all_lines.append(((x1, y1), (x2, y2)))

# Use HoughLinesP method to detect more accurate lines
lines_p = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Iterate over all detected lines
for line in lines_p:
    x1, y1, x2, y2 = line[0]
    cv2.line(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Store lines for distance calculation
    all_lines.append(((x1, y1), (x2, y2)))

# Set a threshold for the minimum distance between lines
min_distance_threshold = 50

# Set a threshold for checking if lines are approximately parallel
parallel_threshold = 0.1  # You can adjust this value based on your needs

# Calculate the distance between parallel lines
distances = []
for i in range(len(all_lines)):
    for j in range(i + 1, len(all_lines)):
        # Extract endpoints of lines
        (x1, y1), (x2, y2) = all_lines[i]
        (x3, y3), (x4, y4) = all_lines[j]

        # Calculate the angle between the two lines
        angle_diff = np.abs(np.arctan2(y2 - y1, x2 - x1) - np.arctan2(y4 - y3, x4 - x3))

        # Check if the lines are approximately parallel
        if angle_diff < parallel_threshold:
            # Calculate the distance between lines using the distance formula
            distance = np.abs((y2 - y1) * x3 - (x2 - x1) * y3 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

            # Check if the distance is above the threshold
            if distance > min_distance_threshold:
                distances.append(distance)

# Display the distances on the image and in the terminal
for i, distance in enumerate(distances):
    cv2.putText(output_frame, f"Distance {i + 1}: {distance:.2f} pixels", (10, 30 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    print(f"Distance {i + 1}: {distance:.2f} pixels")

# Display the combined image with detected objects
cv2.imshow('Detected Objects', output_frame)

# Check if the "s" key is pressed (wait for key event for 0 milliseconds)
key = cv2.waitKey(0)
if key == ord('s'):
    # Save the displayed window as another JPG image
    cv2.imwrite('output_image.jpg', output_frame)

cv2.destroyAllWindows()






























