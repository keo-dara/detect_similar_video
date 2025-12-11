                                                                                                                                                 
import cv2                                                                                                                                       
import os                                                                                                                                        
from pathlib import Path                                                                                                                         
import yaml                                                                                                                                      
                                                                                                                                                 
def annotate_by_video_title():                                                                                                                   
    video_dir = "./videos"                                                                                                                       
    dataset_dir = "./yolo_dataset"                                                                                                               
                                                                                                                                                 
    # Create directory structure                                                                                                                 
    dirs = [                                                                                                                                     
        f"{dataset_dir}/train/images",                                                                                                           
        f"{dataset_dir}/train/labels",                                                                                                           
        f"{dataset_dir}/val/images",                                                                                                             
        f"{dataset_dir}/val/labels"                                                                                                              
    ]                                                                                                                                            
                                                                                                                                                 
    for dir_path in dirs:                                                                                                                        
        os.makedirs(dir_path, exist_ok=True)                                                                                                     
                                                                                                                                                 
    # Get all video files and create class mapping                                                                                               
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]                                                                       
    video_files.sort()  # Sort to ensure consistent class IDs                                                                                    
                                                                                                                                                 
    # Create class mapping from video names                                                                                                      
    class_names = []                                                                                                                             
    class_mapping = {}                                                                                                                           
                                                                                                                                                 
    for i, video_file in enumerate(video_files):                                                                                                 
        class_name = Path(video_file).stem  # Get filename without extension                                                                     
        class_names.append(class_name)                                                                                                           
        class_mapping[video_file] = i                                                                                                            
                                                                                                                                                 
    print(f"Found classes: {class_names}")                                                                                                       
    print(f"Class mapping: {class_mapping}")                                                                                                     
                                                                                                                                                 
    all_frames = []                                                                                                                              
                                                                                                                                                 
    # Process each video                                                                                                                         
    for video_file in video_files:                                                                                                               
        video_path = os.path.join(video_dir, video_file)                                                                                         
        class_id = class_mapping[video_file]                                                                                                     
                                                                                                                                                 
        print(f"Processing {video_file} as class {class_id} ({Path(video_file).stem})...")                                                       
                                                                                                                                                 
        cap = cv2.VideoCapture(video_path)                                                                                                       
        frame_count = 0                                                                                                                          
        saved_count = 0                                                                                                                          
                                                                                                                                                 
        while True:                                                                                                                              
            ret, frame = cap.read()                                                                                                              
            if not ret:                                                                                                                          
                break                                                                                                                            
                                                                                                                                                 
            # Extract every 30th frame                                                                                                           
            if frame_count % 30 == 0:                                                                                                            
                # Save frame                                                                                                                     
                video_name = Path(video_file).stem                                                                                               
                frame_name = f"{video_name}_frame_{saved_count:06d}.jpg"                                                                         
                frame_path = os.path.join(dataset_dir, "temp", frame_name)                                                                       
                                                                                                                                                 
                # Create temp directory                                                                                                          
                os.makedirs(os.path.dirname(frame_path), exist_ok=True)                                                                          
                cv2.imwrite(frame_path, frame)                                                                                                   
                                                                                                                                                 
                # Create annotation for entire frame (whole image is this class)                                                                 
                # Using normalized coordinates: center at 0.5, 0.5 with full width/height                                                        
                annotation = f"{class_id} 0.5 0.5 1.0 1.0"  # Full frame annotation                                                              
                                                                                                                                                 
                all_frames.append((frame_path, [annotation]))                                                                                    
                saved_count += 1                                                                                                                 
                                                                                                                                                 
            frame_count += 1                                                                                                                     
                                                                                                                                                 
        cap.release()                                                                                                                            
        print(f"Extracted {saved_count} frames from {video_file}")                                                                               
                                                                                                                                                 
    # Split into train/val (80/20)                                                                                                               
    import random                                                                                                                                
    random.shuffle(all_frames)                                                                                                                   
    split_idx = int(0.8 * len(all_frames))                                                                                                       
    train_frames = all_frames[:split_idx]                                                                                                        
    val_frames = all_frames[split_idx:]                                                                                                          
                                                                                                                                                 
    # Move files to proper directories                                                                                                           
    def save_split(frames_list, split_name):                                                                                                     
        for i, (frame_path, annotations) in enumerate(frames_list):                                                                              
            # New paths                                                                                                                          
            img_name = f"{split_name}_{i:06d}.jpg"                                                                                               
            txt_name = f"{split_name}_{i:06d}.txt"                                                                                               
                                                                                                                                                 
            new_img_path = os.path.join(dataset_dir, split_name, "images", img_name)                                                             
            new_txt_path = os.path.join(dataset_dir, split_name, "labels", txt_name)                                                             
                                                                                                                                                 
            # Copy image                                                                                                                         
            import shutil                                                                                                                        
            shutil.move(frame_path, new_img_path)                                                                                                
                                                                                                                                                 
            # Save annotations                                                                                                                   
            with open(new_txt_path, 'w') as f:                                                                                                   
                for annotation in annotations:                                                                                                   
                    f.write(annotation + '\n')                                                                                                   
                                                                                                                                                 
    save_split(train_frames, "train")                                                                                                            
    save_split(val_frames, "val")                                                                                                                
                                                                                                                                                 
    # Clean up temp directory                                                                                                                    
    import shutil                                                                                                                                
    temp_dir = os.path.join(dataset_dir, "temp")                                                                                                 
    if os.path.exists(temp_dir):                                                                                                                 
        shutil.rmtree(temp_dir)                                                                                                                  
                                                                                                                                                 
    # Create dataset.yaml                                                                                                                        
    dataset_config = {                                                                                                                           
        'train': f"{os.path.abspath(dataset_dir)}/train/images",                                                                                 
        'val': f"{os.path.abspath(dataset_dir)}/val/images",                                                                                     
        'nc': len(class_names),                                                                                                                  
        'names': class_names                                                                                                                     
    }                                                                                                                                            
                                                                                                                                                 
    with open(f"{dataset_dir}/dataset.yaml", 'w') as f:                                                                                          
        yaml.dump(dataset_config, f)                                                                                                             
                                                                                                                                                 
    print(f"\nDataset created successfully!")                                                                                                    
    print(f"Classes: {class_names}")                                                                                                             
    print(f"Train images: {len(train_frames)}")                                                                                                  
    print(f"Val images: {len(val_frames)}")                                                                                                      
    print(f"Dataset config: {dataset_dir}/dataset.yaml")                                                                                         
                                                                                                                                                 
if __name__ == "__main__":                                                                                                                       
    annotate_by_video_title()      