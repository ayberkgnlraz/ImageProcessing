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

# Use HoughLinesP method to detect more accurate lines
lines_p = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Iterate over all detected lines
for line in lines_p:
    x1, y1, x2, y2 = line[0]
    cv2.line(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Use probabilistic Hough transform to detect rectangles
rectangles = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100, min_theta=np.pi/4, max_theta=3*np.pi/4)

# Iterate over all detected rectangles
for rectangle in rectangles:
    rho, theta = rectangle[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    # Draw rectangles on the separate image
    cv2.line(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Display the combined image with detected objects
cv2.imshow('Detected Objects', output_frame)

# Check if the "s" key is pressed (wait for key event for 0 milliseconds)
key = cv2.waitKey(0)
if key == ord('s'):
    # Save the displayed window as another JPG image
    cv2.imwrite('output_image.jpg', output_frame)

cv2.destroyAllWindows()