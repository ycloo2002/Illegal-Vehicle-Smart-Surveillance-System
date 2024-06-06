import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
import pytesseract
from PIL import Image, ImageEnhance
reader = easyocr.Reader(['en'], gpu=True)
import time
import sys
def drawbox(img,x1,x2,y1,y2,label_text,color = (255, 0, 0),thickness = 5):
    """draw the rectangle box to the image with label

    Args:
        img (_type_): image input
        x1 (int): x1 location
        x2 (int): x2 location
        y1 (int): y1 location
        y2 (int): y2 location
        label_text (str): the label
        color (tuple, optional): color of the line . Defaults to (255, 0, 0).
        thickness (int, optional): the thickness of line . Defaults to 5.
    """
    start_point = (x1, y1)
    end_point = (x2, y2)
    label_text = str(label_text)

    # Draw the rectangle on the image
    cv2.rectangle(img, start_point, end_point, color, thickness)
    
    # Add the label text to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0,0,0) #black
    (label_width, label_height), _ = cv2.getTextSize(label_text, font, font_scale,thickness)
    label_position = (start_point[0], start_point[1] - 5) 
    
    #draw the rectangle
    start_point = label_position
    end_point = (label_position[0] + label_width, label_position[1] - label_height)
    cv2.rectangle(img, start_point, end_point, color, thickness=cv2.FILLED)

    #putting the label text
    cv2.putText(img, label_text, label_position, font, font_scale, font_color, thickness=5)

def text_reader(reader,img):
    
    #define the pattern
    pattern =r'^[a-zA-Z][a-zA-Z0-9]*[0-9][a-zA-Z0-9]*$'
    
    #read the text
    text_result =reader.readtext(img)              
    #if non of the result showing out  
      
    if text_result:          
        total_score =0
        lp_text = "" 
               
        #combine the text
        for (bbox, text, score) in text_result:
            lp_text += text
            total_score += score
                                        
        avg_lp_score = total_score/len(text_result)
        
        lp_text = lp_text.upper().replace(' ', '')
    
        #continue when the pattern is correct.
        if re.match(pattern, lp_text) and avg_lp_score>=0.5:
            return True,lp_text
    
    return False,""    
                    
def search_vehicle(frame,vehicel_model,plate):
    
    vehicle_detect =""
    car_results = vehicel_model(frame,classes=[2,3,5,7],stream=True)[0]
    vehicle_temp = [] 
    success_detect = False   
    d_x1 = [] 
      
    for detection in car_results.boxes.data.tolist():
        cx1,cy1, cx2, cy2, pscore, classid = detection 
        drawbox(frame,int(cx1),int(cx2),int(cy1),int(cy2),f'{classid}_pscore',(255, 0, 0), 5) 
        if classid == 3:
            return ""
        
        print(f"{plate[0]} >= {cx1} = {plate[0] >= cx1}")
        print(f"{plate[1]} >= {cy1} = {plate[1] >= cy1}")
        print(f"{plate[2]} <= {cx2} = {plate[2] <= cx2}")
        print(f"{plate[3]} <= {cy2} = {plate[3] <= cy2}")
        print("\n")
        
        if plate[0] >= cx1 and plate[1] >= cy1 and plate[2] <= cx2 and plate[3] <= cy2:
            vehicle_detect = [cx1,cy1, cx2, cy2, classid]
            success_detect = True
            break
        
    """    else:
            vehicle_temp.append([cx1,cy1, cx2, cy2, classid])
            d_x1.append(cx1)
            
    if not success_detect:
        nearest_value = min(d_x1, key=lambda x: abs(x - plate[0]))
        position = d_x1.index(nearest_value)
        vehicle_detect = vehicle_temp[position]
        print("N : ",nearest_value, "\n postion : ",position)"""
            
    return vehicle_detect

def search_plate(frame,plate_detection,reader,save_plate):

    plate_detect = []
    
    car_plate_results = plate_detection(frame)[0]                   
    for detection in car_plate_results.boxes.data.tolist():
        px1,py1, px2, py2, pscore, classid = detection                       
                          
        if pscore >= 0.8 :#and (px1-5)>=0:
                        
            lp_crop = frame[int(py1):int(py2), int(px1): int(px2)]    

            reader_result,result = text_reader(reader,lp_crop)
            
            if reader_result:
                
                same_lp = False
                for plate_no in save_plate:
                    if result == plate_no:
                        same_lp = True
                        break
                
                if not same_lp : plate_detect.append([px1,py1,px2,py2,result])
    
    return plate_detect

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for file in self.files:
            if file is not None:
                file.write(text)
                file.flush()  # Ensure output is written immediately
            else:
                print("Warning: One of the file objects is None!")

    def flush(self):
        for file in self.files:
            if file is not None:
                file.flush()
                
with open('Log File.log', 'w') as log_file:
            # Duplicate stdout and stderr to the console and the log file
            
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = sys.stderr = Tee(sys.stdout, log_file)
    
    vehicel_model = YOLO(f'utils/model/yolov8n.pt')
    plate_detection = YOLO("F:\\fyp_system\\utils\\model\\car_plate_v5.pt")
    #open video
    cap = cv2.VideoCapture("F:\\FYP_save\\dataset\\video_raw\\IMG_0139.MOV")
    #cap.set(cv2.CAP_PROP_FPS, 30)
    
    vehicles = {2:"car",5:'bus', 7:'truck'}

    # Get the video frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Define the codec and create VideoWriter object 
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    out = cv2.VideoWriter(f'F:\\fyp_system\\save\\result_video.avi', fourcc, 25.0, (frame_width, frame_height))  

    save_plate = []              
    # Loop through the video frames

    start_time = time.time()
    while cap.isOpened():
                
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            new_frame = frame.copy()
                
            plate_detect = search_plate(frame,plate_detection,reader,save_plate)
            
            if len(plate_detect) > 0:
                
                for plate in plate_detect:
                    
                    save_plate.append(plate[4])
                    print(save_plate)
                    
                    vehicle_detect = search_vehicle(new_frame,vehicel_model,plate)
                    #if plate[0] > vehicle[0] and plate[1] > vehicle[1] and plate[2] < vehicle[2] and plate[3] < vehicle[3]:
                    if not vehicle_detect == "":
                        x1 = vehicle_detect[0] #max((plate[0]-500),0)
                        y1 = vehicle_detect[1] #max((plate[1]-500),0)
                        
                        x2 = vehicle_detect[2] #min((plate[2]+600),frame_width)
                        y2 = vehicle_detect[3] #min((plate[3]+800),frame_height)
                                
                        vehicle_crop = frame[int(y1):int(y2), int(x1): int(x2)]
                                
                            #save crop image path
                        img_path = f'F:\\fyp_system\\save\\{plate[4]}.jpg'
                                                                        #save the image with the lp name.                                        
                        cv2.imwrite(img_path, vehicle_crop)                                     
            
                            #drawbox(new_frame,int(vehicle[0]),int(vehicle[2]),int(vehicle[1]),int(vehicle[3]),f'{plate[4]}',(255, 0, 0), 5) 
                    else :
                        vehicle_crop = frame[int(plate[1]):int(plate[3]), int(plate[0]): int(plate[2])]
                        img_path = f'F:\\fyp_system\\save\\LP_{plate[4]}.jpg'                                       
                        cv2.imwrite(img_path, vehicle_crop)      
                        
                        img_path = f'F:\\fyp_system\\save\\original_{plate[4]}.jpg'                                       
                        cv2.imwrite(img_path, new_frame)   
                                            
                
            out.write(new_frame)    
            
             # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", new_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break                
        else:
                    # Break the loop if the end of the video is reached
            break
            
            # Release the video capture object and close the display window
    cap.release()
    out.release()
    
    running_time =time.time()- start_time

    print("Running time : ",(running_time/60))
    print(save_plate)

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    cv2.destroyAllWindows()