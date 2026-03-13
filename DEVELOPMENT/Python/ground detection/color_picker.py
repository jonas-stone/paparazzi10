import cv2
import numpy as np

# Load your reference image
image_path = 'TEAM-10-PROTOTYPING/downloads from drone/20260306-095826/733439027.jpg' 
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not find image. Check your path!")
    exit()

# Convert to HSV once
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the HSV value at the clicked coordinates (y, x)
        hsv_value = hsv_image[y, x]
        print(f"Clicked at ({x}, {y}) - HSV Value: {hsv_value}")
        print(f"Suggested Lower: [{hsv_value[0]-10}, 50, 50]")
        print(f"Suggested Upper: [{hsv_value[0]+10}, 255, 255]\n")

# Create a window and bind the function to it
cv2.namedWindow('Calibration')
cv2.setMouseCallback('Calibration', mouse_callback)

print("INSTRUCTIONS:")
print("1. Click on different green parts of the image (dark green, light green).")
print("2. Note the 'Hue' (the first number).")
print("3. Press any key to close the window.")

cv2.imshow('Calibration', image)
cv2.waitKey(0)
cv2.destroyAllWindows()