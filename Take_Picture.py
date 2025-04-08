import time
import cv2

def capture_image(save_path="captured_image.jpg"):
    # Initialize the webcam (webcam ID=0 is typically the default device)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Give the camera a moment to warm up (optional)
    time.sleep(2)

    # Read a frame from the camera
    ret, frame = cap.read()

    if ret:
        # Save the captured frame to a file
        cv2.imwrite(save_path, frame)
        print(f"Image saved as {save_path}")
    else:
        print("Failed to capture image")

    # Release the camera
    cap.release()

if __name__ == '__main__':
    capture_image()
