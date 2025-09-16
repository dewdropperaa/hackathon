import cv2
import os
import time
from datetime import datetime

def create_captures_folder():
    """Create the captures folder if it doesn't exist."""
    if not os.path.exists('captures'):
        os.makedirs('captures')
        print("Created 'captures' folder")

def capture_image():
    """Capture a single image from the webcam."""
    # Initialize the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return False
    
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    if ret:
        # Generate timestamp filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plant_{timestamp}.jpg"
        filepath = os.path.join('captures', filename)
        
        # Save the image
        cv2.imwrite(filepath, frame)
        print(f"✓ Captured: {filename}")
        
        # Release the webcam
        cap.release()
        return True
    else:
        print("Error: Failed to capture image")
        cap.release()
        return False

def main():
    """Main function to run the auto-capture loop."""
    # Configuration - change this for testing
    CAPTURE_INTERVAL = 300  # 5 minutes (300 seconds)
    # For testing, uncomment the line below:
    # CAPTURE_INTERVAL = 10  # 10 seconds for testing
    
    print("Webcam Auto-Capture Script")
    print(f"Capturing every {CAPTURE_INTERVAL} seconds")
    print("Press Ctrl+C to stop")
    print("-" * 40)
    
    # Create the captures folder
    create_captures_folder()
    
    try:
        while True:
            # Capture an image
            success = capture_image()
            
            if not success:
                print("Retrying in 30 seconds...")
                time.sleep(30)
                continue
            
            # Wait for the specified interval
            print(f"Next capture in {CAPTURE_INTERVAL} seconds...")
            time.sleep(CAPTURE_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\nCapture stopped by user")
        print("Script terminated gracefully")

if __name__ == "__main__":
    main()