import cv2
import numpy as np
import easyocr
import imutils
import time
import os

def main():
    # Initialize the OCR reader
    print("Initializing EasyOCR... This may take a moment.")
    reader = easyocr.Reader(['en'], gpu=False)
    print("EasyOCR initialized successfully!")
    
    # File paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cascade_path = os.path.join(current_dir, 'resources', 'haarcascade_russian_plate_number.xml')
    
    # Check if cascade file exists
    if not os.path.exists(cascade_path):
        print(f"Error: Cascade file not found at {cascade_path}")
        return
    
    # Initialize the cascade classifier for license plate detection
    plate_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Initialize webcam (0 is usually the built-in webcam)
    # To use a video file instead, replace 0 with the file path like:
    # video_path = os.path.join(current_dir, 'resources', 'your_video.mp4')
    # cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # Convert frame to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect license plates
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.2, 
                                              minNeighbors=5, 
                                              minSize=(25, 25))
        
        # Process each detected plate
        for (x, y, w, h) in plates:
            # Extract the plate region
            plate_img = frame[y:y+h, x:x+w]
            
            # Draw rectangle around the plate
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Apply preprocessing to the plate image
            plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            plate_blur = cv2.GaussianBlur(plate_gray, (5, 5), 0)
            
            # Use OCR to read the license plate
            try:
                results = reader.readtext(plate_blur)
                
                # Check if any text was found
                if results:
                    # Get the text with highest confidence
                    best_result = max(results, key=lambda x: x[2])
                    text, confidence = best_result[1], best_result[2]
                    
                    # Only show if confidence is high enough
                    if confidence > 0.4:
                        # Display the recognized text above the plate
                        cv2.putText(frame, f"Plate: {text}", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                   (0, 255, 0), 2)
                        print(f"Detected plate: {text} (confidence: {confidence:.2f})")
            except Exception as e:
                print(f"OCR Error: {e}")
        
        # Display the resulting frame
        cv2.imshow('License Plate Detection', frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()