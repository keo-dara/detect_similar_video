from ultralytics import YOLO
                                                                                                              
# Load your trained model                                                                                                                        
model = YOLO('./runs/detect/video_training3/weights/best.pt')                                                                                     
                                                                                                                                                 
# Test on new images/videos                                                                                                                      
results = model('test_image/image1.jpg')                                                                                                        
results[0].show()