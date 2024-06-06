import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
import pytesseract
from PIL import Image, ImageEnhance

import time
import sys
import matplotlib
from collections import deque, defaultdict

# Force matplotlib to not use any backend.
matplotlib.use('Agg')

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

class Detection():
    
    def __init__(load): 
        
        load.vehicel_model = YOLO(f'utils/model/yolov8n.pt')
        load.plate_detection = YOLO("F:\\fyp_system\\utils\\model\\car_plate_v5.pt")
        #open video
        
        video_path = "F:\\FYP_save\\dataset\\video_raw\\IMG_0139.MOV"
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Loop through the video frames
        load.reader = easyocr.Reader(['en'], gpu=True)
        start_time = time.time()
        
        # Get video properties
        #width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #out = cv2.VideoWriter(f'F:\\fyp_system\\save\\result_video.avi', fourcc,20, (width, height))

        load.save_plate = []   
        load.lp_5_most = deque(maxlen=5)
        while cap.isOpened():
                    
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                new_frame = frame.copy()
                    
                new_frame = load.vid(new_frame) 
                                                   
                #out.write(new_frame)    
                           
            else:
                        # Break the loop if the end of the video is reached
                break
        
        cap.release()
        #out.release()
        
        running_time =time.time()- start_time

        print("Running time : ",(running_time/60))
        print(load.save_plate)

        cv2.destroyAllWindows()
    
    def vid(load,frame):
        
        plate_detect = load.search_plate(frame)
                
        if len(plate_detect) > 0:
                    
            for plate in plate_detect:
                        
                load.save_plate.append(plate[4])
                print(load.save_plate)
                        
                vehicle_detect = load.search_vehicle(frame,plate)
                
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
                
                else :    
                            
                    img_path = f'F:\\fyp_system\\save\\original_{plate[4]}.jpg'                                       
                    cv2.imwrite(img_path,frame)  
        return frame
                            
    def text_reader(load,img):
    
        #define the pattern
        pattern =r'^[a-zA-Z][a-zA-Z0-9]*[0-9][a-zA-Z0-9]*$'
        
        #read the text
        text_result =load.reader.readtext(img)              
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
            if re.match(pattern, lp_text) and avg_lp_score>=0.8:
                #lp_text = correct_ocr_errors(lp_text)
                return True,lp_text
        
        return False,""    
                    
    def search_vehicle(load,frame,plate):
        
        vehicle_detect =""
        car_results = load.vehicel_model(frame,classes=[2,3,5,7])[0]
        
        for detection in car_results.boxes.data.tolist():
            cx1,cy1, cx2, cy2, pscore, classid = detection 
            
            if plate[0] >= cx1 and plate[1] >= cy1 and plate[2] <= cx2 and plate[3] <= cy2:
                print(classid)
                vehicle_detect = [cx1,cy1, cx2, cy2, classid]
                break
                
        return vehicle_detect

    def search_plate(load,frame):

        plate_detect = []
        
        car_plate_results = load.plate_detection(frame)[0]                   
        for detection in car_plate_results.boxes.data.tolist():
            px1,py1, px2, py2, pscore, classid = detection                       
                            
            if True:#pscore >= 0.3 :
                            
                lp_crop = frame[int(py1):int(py2), int(px1): int(px2)]    

                reader_result,result = load.text_reader(lp_crop)
                
                if reader_result:
                    if load.get_the_most_frequent(result):
                        plate_detect.append([px1,py1,px2,py2,result])
                        
        return plate_detect

    def get_the_most_frequent(load,result):
        
        load.lp_5_most.append(result)
        
        if len(load.lp_5_most) < 5:
            return False
        # Count frequencies of elements
        freq = defaultdict(int)
        
        for elem in load.lp_5_most:
            freq[elem] += 1

        # Find the element with the highest frequency
        max_freq_elem = max(freq, key=freq.get)
        max_freq = freq[max_freq_elem]
        
        same_lp = False
        for plate_no in load.save_plate:
            if max_freq_elem == plate_no:
                same_lp = True
                break
        
        if not same_lp : 
            return True


with open('Log File.log', 'w') as log_file:
                # Duplicate stdout and stderr to the console and the log file
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = sys.stderr = Tee(sys.stdout, log_file)
    
    load = Detection()
    
    sys.stdout = original_stdout
    sys.stderr = original_stderr