import cv2
import numpy as np

# Loading our image
image = cv2.imread('python_ayberk\image.jpg')

# Converting the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Applying GaussianBlur to reduce noise and improve contour detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Using edge detection (Canny) to find contours
edges = cv2.Canny(blurred, 50, 150)

# Using HoughLines method to detect lines in the image
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

# Creating a separate image to draw rectangles and text
output_frame = image.copy()

# Keeping track of detected object dimensions
detected_objects = []

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

# Using HoughLinesP method to detect more accurate lines
lines_p = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Iterating over all detected lines
for line in lines_p:
    x1, y1, x2, y2 = line[0]
    cv2.line(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Using probabilistic Hough transform to detect rectangles
rectangles = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100, min_theta=np.pi/4, max_theta=3*np.pi/4)

# Iterating over all detected rectangles
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

    # Drawing rectangles on the separate image
    cv2.line(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Displaying the combined image with detected objects
cv2.imshow('Detected Objects', output_frame)

# saving new image when we press "s"
key = cv2.waitKey(0)
if key == ord('s'):
    
    cv2.imwrite('output_image.jpg', output_frame)

cv2.destroyAllWindows()