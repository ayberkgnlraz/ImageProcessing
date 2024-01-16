import cv2
import numpy as np

# loading our image
image = cv2.imread('python_ayberk\shapess.jpg')

# Converting the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Applying GaussianBlur to reduce noise and improve contour detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Using edge detection (Canny) to find contours
edges = cv2.Canny(blurred, 50, 150)

# Finding contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Creating a separate image to draw rectangles and text
output_frame = image.copy()

# Keeping track of detected object dimensions
detected_objects = []

# Iterating over all detected contours
for contour in contours:
    # Calculating the approximate polygonal curve representing the contour
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Checking if the contour is rectangular and has sufficient area
    if len(approx) == 4 and cv2.contourArea(contour) > 1000:  # Adjust the area threshold based on your needs
        # Drawing the bounding rectangle on the separate image
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(output_frame, [box], 0, (0, 255, 0), 2)

        # Calculating the size of the object in millimeters based on the A4 paper dimensions
        a4_width_mm = 210
        a4_height_mm = 297
        object_width_mm = (rect[1][0] / image.shape[1]) * a4_width_mm
        object_height_mm = (rect[1][1] / image.shape[0]) * a4_height_mm

        # Displaying the dimensions on the image
        text = f"Object Size: {object_width_mm:.2f} mm x {object_height_mm:.2f} mm"
        cv2.putText(output_frame, text, (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Displaying the dimensions in the terminal
        print(f"Object Size: {object_width_mm:.2f} mm x {object_height_mm:.2f} mm")

        # Keeping track of detected object dimensions
        detected_objects.append((object_width_mm, object_height_mm))

# Displaying the combined image with detected objects
cv2.imshow('Detected Objects', output_frame)

# Printing all detected object dimensions
print("All Detected Object Dimensions:")
for i, (width, height) in enumerate(detected_objects, start=1):
    print(f"Object {i}: {width:.2f} mm x {height:.2f} mm")

# saving new image when we press "s"
key = cv2.waitKey(0)
if key == ord('s'):
    # Save the displayed window as another JPG image
    cv2.imwrite('output_image.jpg', output_frame)

cv2.destroyAllWindows()
