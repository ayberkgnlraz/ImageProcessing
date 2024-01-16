import cv2
import numpy as np
import time

# Known dimensions of the reference object (cigarette pack) in millimeters
reference_width_mm = 90
reference_height_mm = 60

# Known dimensions of the reference object in square millimeters
reference_area_mm2 = 63 * 10  # 63cm^2 to mm^2, adjust the factor based on your actual conversion

# Known conversion factor: pixels to millimeters
pixels_to_mm = 10  # Adjust this based on your actual conversion factor

# Open a connection to the camera 
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Time interval for printing results to the terminal (in seconds)
print_interval = 2
last_print_time = time.time()

# Minimum contour area to filter out small details
min_contour_area = 500

# Create a background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if not ret:
        print("Error: Could not read frame.")
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Use Contour detection to find rectangular objects
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a separate image to draw rectangles and text
    output_frame = frame.copy()

    # Iterate over detected contours
    for contour in contours:
        # Filter out small contours
        if cv2.contourArea(contour) < min_contour_area:
            continue

        # Find the bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Draw the bounding rectangle on the separate image
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate the size of the object in millimeters based on the reference object
        object_width_mm = w * pixels_to_mm
        object_height_mm = h * pixels_to_mm

        # Calculate the area of the object in square millimeters
        object_area_mm2 = object_width_mm * object_height_mm

        # Display the dimensions in the terminal every 2 seconds
        current_time = time.time()
        if current_time - last_print_time >= print_interval:
            print(f"Width: {object_width_mm:.2f} mm, Height: {object_height_mm:.2f} mm, Area: {object_area_mm2:.2f} mm^2")
            last_print_time = current_time

    # Display the combined frame with detected rectangles
    cv2.imshow('Live Stream', output_frame)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
















