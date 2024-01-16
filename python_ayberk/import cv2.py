import cv2
import numpy as np
import time

# pixels to millimeters (just for one i will count pixels as mm)
pixels_to_mm = 1

# webcam opening code
cap = cv2.VideoCapture(0)

# Checking if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Time interval for printing results to the terminal (in seconds) (for the sake of readibility of results)
print_interval = 2
last_print_time = time.time()

# Minimum contour area to filter out small details
min_contour_area = 500

# background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    # Reading a frame from the camera
    ret, frame = cap.read()

    # Checking if the frame was successfully read
    if not ret:
        print("Error: Could not read frame.")
        break

    # background subtraction
    fgmask = fgbg.apply(frame)

    # Using Contour detection to find rectangular objects
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Creating a separate image to draw rectangles and text
    output_frame = frame.copy()

    # Iterating over detected contours
    for contour in contours:
        # Filtering out small contours
        if cv2.contourArea(contour) < min_contour_area:
            continue

        # Finding the bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Drawing the bounding rectangle on the separate image
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculating the size of the object in millimeters based on the reference object
        object_width_mm = w * pixels_to_mm
        object_height_mm = h * pixels_to_mm

        # Calculating the area of the object in square millimeters
        object_area_mm2 = object_width_mm * object_height_mm

        # Displaying the dimensions in the terminal every 2 seconds
        current_time = time.time()
        if current_time - last_print_time >= print_interval:
            print(f"Width: {object_width_mm:.2f} mm, Height: {object_height_mm:.2f} mm, Area: {object_area_mm2:.2f} mm^2")
            last_print_time = current_time

    # Displaying the combined frame with detected rectangles
    cv2.imshow('Live Stream', output_frame)

    # Breaking the loop if 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Releasing the camera and close the window
cap.release()
cv2.destroyAllWindows()
















