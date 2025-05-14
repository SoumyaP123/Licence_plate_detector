# import cv2
# import numpy as np
# import easyocr
# import imutils
# import time
# import os
# import argparse

# def preprocess_plate_image(plate_img):
#     """Apply image preprocessing techniques to improve OCR accuracy"""
#     # Convert to grayscale
#     gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
#     # Apply Gaussian blur
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # Apply adaptive thresholding
#     thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                   cv2.THRESH_BINARY_INV, 11, 2)
    
#     # Apply morphological operations to clean the image
#     kernel = np.ones((3,3), np.uint8)
#     morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
#     return morph

# def main():
#     # Parse command-line arguments
#     parser = argparse.ArgumentParser(description='License Plate Detection from Video')
#     parser.add_argument('video_path', help='Path to the video file')
#     parser.add_argument('--output', help='Path to save output video', default=None)
#     parser.add_argument('--save-plates', action='store_true', help='Save detected plate images')
#     args = parser.parse_args()
    
#     # Create output directory for plates if needed
#     if args.save_plates:
#         os.makedirs('detected_plates', exist_ok=True)
    
#     # Initialize the OCR reader
#     print("Initializing EasyOCR... This may take a moment.")
#     reader = easyocr.Reader(['en'], gpu=False)
#     print("EasyOCR initialized successfully!")
    
#     # File paths
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     cascade_path = os.path.join(current_dir, 'resources', 'haarcascade_russian_plate_number.xml')
    
#     # Check if cascade file exists
#     if not os.path.exists(cascade_path):
#         print(f"Error: Cascade file not found at {cascade_path}")
#         return
    
#     # Check if video file exists
#     if not os.path.exists(args.video_path):
#         print(f"Error: Video file not found at {args.video_path}")
#         return
    
#     # Initialize the cascade classifier for license plate detection
#     plate_cascade = cv2.CascadeClassifier(cascade_path)
    
#     # Initialize video capture
#     cap = cv2.VideoCapture(args.video_path)
    
#     if not cap.isOpened():
#         print("Error: Could not open video source")
#         return
    
#     # Get video properties for saving output
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     # Initialize video writer if output is specified
#     out = None
#     if args.output:
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
    
#     # Variables for tracking progress
#     plate_count = 0
#     start_time = time.time()
#     frame_count = 0
    
#     while True:
#         # Read a frame from the video
#         ret, frame = cap.read()
#         if not ret:
#             print("End of video stream. Exiting...")
#             break
        
#         frame_count += 1
        
#         # Calculate and print progress
#         if frame_count % 30 == 0:  # Update progress every 30 frames
#             elapsed = time.time() - start_time
#             progress = (frame_count / total_frames) * 100
#             fps_rate = frame_count / elapsed
#             remaining = (total_frames - frame_count) / fps_rate if fps_rate > 0 else 0
#             print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) | FPS: {fps_rate:.1f} | Time remaining: {remaining:.1f}s")
        
#         # Convert frame to grayscale for detection
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Detect license plates
#         plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.2, 
#                                               minNeighbors=5, 
#                                               minSize=(25, 25))
        
#         # Process each detected plate
#         for (x, y, w, h) in plates:
#             # Extract the plate region
#             plate_img = frame[y:y+h, x:x+w]
            
#             # Skip if plate is too small
#             if plate_img.size == 0 or plate_img.shape[0] < 15 or plate_img.shape[1] < 15:
#                 continue
            
#             # Draw rectangle around the plate
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
#             # Apply preprocessing to the plate image
#             processed_plate = preprocess_plate_image(plate_img)
            
#             # Use OCR to read the license plate
#             try:
#                 results = reader.readtext(processed_plate)
                
#                 # Check if any text was found
#                 if results:
#                     # Get the text with highest confidence
#                     best_result = max(results, key=lambda x: x[2])
#                     text, confidence = best_result[1], best_result[2]
                    
#                     # Only show if confidence is high enough
#                     if confidence > 0.4:
#                         # Display the recognized text above the plate
#                         cv2.putText(frame, f"{text} ({confidence:.2f})", (x, y-10), 
#                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
#                                    (0, 255, 0), 2)
                        
#                         # Save plate image if requested
#                         if args.save_plates:
#                             plate_filename = f"detected_plates/plate_{plate_count}_{text}.jpg"
#                             cv2.imwrite(plate_filename, plate_img)
#                             print(f"Saved plate image to {plate_filename}")
#                             plate_count += 1
#             except Exception as e:
#                 print(f"OCR Error: {e}")
        
#         # Display the resulting frame
#         cv2.imshow('License Plate Detection', frame)
        
#         # Write frame to output video if specified
#         if out is not None:
#             out.write(frame)
        
#         # Break the loop when 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # Release resources when done
#     cap.release()
#     if out is not None:
#         out.write(frame)
#         out.release()
#     cv2.destroyAllWindows()
    
#     print(f"Processing complete! Detected {plate_count} license plates.")
#     print(f"Total time: {time.time() - start_time:.2f} seconds")

# if __name__ == "__main__":
#     main()


import cv2
import numpy as np
import easyocr
import imutils
import time
import os
import argparse

def preprocess_plate_image(plate_img):
    """Apply image preprocessing techniques to improve OCR accuracy"""
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply morphological operations to clean the image
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return morph

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='License Plate Detection from Video')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--output', help='Path to save output video', default=None)
    parser.add_argument('--save-plates', action='store_true', help='Save detected plate images')
    parser.add_argument('--resize-width', type=int, default=None, help='Resize video width (maintains aspect ratio)')
    parser.add_argument('--display-width', type=int, default=1280, help='Display window width')
    args = parser.parse_args()
    
    # Create output directory for plates if needed
    if args.save_plates:
        os.makedirs('detected_plates', exist_ok=True)
    
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
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
        return
    
    # Initialize the cascade classifier for license plate detection
    plate_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Initialize video capture
    cap = cv2.VideoCapture(args.video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate resize dimensions (maintaining aspect ratio)
    if args.resize_width:
        resize_width = args.resize_width
        resize_height = int(original_height * (resize_width / original_width))
    else:
        resize_width = original_width
        resize_height = original_height
    
    print(f"Original video dimensions: {original_width}x{original_height}")
    print(f"Processing dimensions: {resize_width}x{resize_height}")
    
    # Initialize video writer if output is specified
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.output, fourcc, fps, (resize_width, resize_height))
    
    # Variables for tracking progress
    plate_count = 0
    start_time = time.time()
    frame_count = 0
    
    # Create a resizable window
    cv2.namedWindow('License Plate Detection', cv2.WINDOW_NORMAL)
    
    # Calculate display dimensions (maintaining aspect ratio)
    display_width = min(args.display_width, resize_width)
    display_height = int(resize_height * (display_width / resize_width))
    cv2.resizeWindow('License Plate Detection', display_width, display_height)
    
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            print("End of video stream. Exiting...")
            break
        
        frame_count += 1
        
        # Resize frame for processing (if needed)
        if args.resize_width:
            frame = cv2.resize(frame, (resize_width, resize_height))
        
        # Calculate and print progress
        if frame_count % 30 == 0:  # Update progress every 30 frames
            elapsed = time.time() - start_time
            progress = (frame_count / total_frames) * 100
            fps_rate = frame_count / elapsed
            remaining = (total_frames - frame_count) / fps_rate if fps_rate > 0 else 0
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) | FPS: {fps_rate:.1f} | Time remaining: {remaining:.1f}s")
        
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
            
            # Skip if plate is too small
            if plate_img.size == 0 or plate_img.shape[0] < 15 or plate_img.shape[1] < 15:
                continue
            
            # Draw rectangle around the plate
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Apply preprocessing to the plate image
            processed_plate = preprocess_plate_image(plate_img)
            
            # Use OCR to read the license plate
            try:
                results = reader.readtext(processed_plate)
                
                # Check if any text was found
                if results:
                    # Get the text with highest confidence
                    best_result = max(results, key=lambda x: x[2])
                    text, confidence = best_result[1], best_result[2]
                    
                    # Only show if confidence is high enough
                    if confidence > 0.4:
                        # Display the recognized text above the plate
                        cv2.putText(frame, f"{text} ({confidence:.2f})", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                   (0, 255, 0), 2)
                        
                        # Save plate image if requested
                        if args.save_plates:
                            plate_filename = f"detected_plates/plate_{plate_count}_{text}.jpg"
                            cv2.imwrite(plate_filename, plate_img)
                            print(f"Saved plate image to {plate_filename}")
                            plate_count += 1
            except Exception as e:
                print(f"OCR Error: {e}")
        
        # Display the resulting frame
        cv2.imshow('License Plate Detection', frame)
        
        # Write frame to output video if specified
        if out is not None:
            out.write(frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources when done
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"Processing complete! Detected {plate_count} license plates.")
    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()