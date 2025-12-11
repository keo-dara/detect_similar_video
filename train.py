import warnings                                                                                                                                  
warnings.filterwarnings('ignore')                                                                                                                
from ultralytics import YOLO                                                                                                                     
import os                                                                                                                                        
                                                                                                                                                 
# Check if dataset exists                                                                                                                        
if not os.path.exists('./yolo_dataset/dataset.yaml'):                                                                                            
    print("Error: yolo_dataset/dataset.yaml not found!")                                                                                         
    print("Please run video_title_annotate.py first to create the dataset.")                                                                     
    exit()                                                                                                                                       
                                                                                                                                                 
# Load a pretrained model                                                                                                                        
model = YOLO('yolov8n.pt')                                                                                                                       
                                                                                                                                                 
# Train the model on your video-based dataset                                                                                                    
results = model.train(                                                                                                                           
    data='./yolo_dataset/dataset.yaml',                                                                                                          
    epochs=100,                                                                                                                                  
    imgsz=640,                                                                                                                                   
    batch=16,           
    device='mps',                                                                                                                         
    name='video_training'                                                                                                                        
)                                                                                                                                                
                                                                                                                                                 
print(f"Training completed! Best model saved at: {results.save_dir}")                                                                            