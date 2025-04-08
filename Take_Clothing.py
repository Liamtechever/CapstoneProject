import time
import os
import cv2

def capture_image(save_path):
    """
    Captures an image using the webcam and saves it to the provided save_path.
    Ensures the directory exists before saving.
    """
    # Ensure the directory for the save_path exists
    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Initialize the webcam (webcam ID=0 is typically the default device)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    # Give the camera a moment to warm up
    time.sleep(2)

    # Read a frame from the camera
    ret, frame = cap.read()

    if ret:
        # Save the captured frame to the file
        cv2.imwrite(save_path, frame)
        print(f"Image saved as {save_path}")
    else:
        print("Failed to capture image")
        save_path = None

    # Release the camera
    cap.release()
    return save_path

def TakePicture():
    """
    Captures a new clothing image using the webcam and saves it with a unique filename in the 'images' folder.
    Returns the file path of the saved image.
    """
    # Generate a unique filename using the current timestamp
    unique_filename = f"clothing_{int(time.time())}.jpg"
    save_path = os.path.join("images", unique_filename)
    return capture_image(save_path)

if __name__ == "__main__":
    TakePicture()
