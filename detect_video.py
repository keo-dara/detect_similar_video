from ultralytics import YOLO
import cv2
                                                                                                              
# Load your trained model                                                                                                                        
model = YOLO('./runs/detect/video_training3/weights/best.pt')                                                                                     
                                                                                                                                                 
# Open video capture (0 for webcam, or path to video file)
cap = cv2.VideoCapture('videos/a.mp4')  # Use video file

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection on the frame
    results = model(frame)
    
    # Draw the results on the frame
    annotated_frame = results[0].plot()
    
    # Display the frame
    cv2.imshow('YOLO Detection', annotated_frame)
    
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
