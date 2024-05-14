import time
import cv2
import easyocr
import re
import csv
import os
from datetime import datetime
from ultralytics import YOLO

def show_output(frame):
    
    # Naming a window 
    cv2.namedWindow("Car_Plate_Detection", cv2.WINDOW_NORMAL) 
        
    # Using resizeWindow() 
    cv2.resizeWindow("Car_Plate_Detection", 1500, 1000)
    
    #window posistion
    cv2.moveWindow("Car_Plate_Detection", 100, 50) 
    
    #show image 
    cv2.imshow("Car_Plate_Detection", frame)
        

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        return True   

def drawbox(img,x1,x2,y1,y2,label_text,color = (255, 0, 0),thickness = 5):
    
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
    
    start_point = label_position
    end_point = (label_position[0] + label_width, label_position[1] - label_height)
    cv2.rectangle(img, start_point, end_point, color, thickness=cv2.FILLED)

    cv2.putText(img, label_text, label_position, font, font_scale, font_color, thickness=5)

def data_and_time():
    date = datetime.now()
    return str(date.strftime('%Y-%m-%d %H-%M-%S'))

def check_duplicated_plate_numbers_no(csv_file_path,data):
    # Read existing data from CSV file
    no = 0
    with open(csv_file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[1] == data[0]:
                return [False,0]
            no+=1      
            
        return [True,no]       
    
def insert_csv(data,csv_path):

    checking_result = check_duplicated_plate_numbers_no(csv_path,data)

    if checking_result[0]:
        data_with_index = [str(checking_result[1])] + data
        with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data_with_index)   
      
def create_csv(path, filename):

    csv_file_path = f"{path}/{filename}.csv"
    
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['No','Plate_number', 'Type','Brand','Colour'])
        print(f"CSV file '{filename}' successful create.")
            
def text_reconise(lp_crop,reader):
    
    #run the text prediction to get the text                         
    text_result = reader.readtext(lp_crop)
     
    #if non of the result showing out    
    if text_result:
                                
        #define 
        lp_text =""
        total_score =0
        
        #combine the text (which some of the lp having two rows)
        for text_detact in text_result:
            bbox, text, score = text_detact

            lp_text += text
            total_score += score
                
        avg_lp_score = total_score/len(text_result)
               
                                
        #continue when the lp reconigse having more that equal to 80%    
        if avg_lp_score >= 0.8:   
            
            #lp pattern set : start from letter and must contain two character and at least one number and not contain any symbol   
            pattern =r'^[a-zA-Z][a-zA-Z0-9]*[0-9][a-zA-Z0-9]*$'
            
            #continue when the pattern is correct.
            if re.match(pattern, lp_text):
                
                #remove the spacing
                lp_text = lp_text.upper().replace(' ', '')
                return [True,lp_text,avg_lp_score]

    return [False,"",""]

def run(input,path):
    
    name = data_and_time()
    
    cap = cv2.VideoCapture(input)

    #open new folder
    new_folder_path = f'{path}save/{name}'
    os.makedirs(new_folder_path)
    
    # Get the video frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object 
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    out = cv2.VideoWriter(f'{new_folder_path}/{name}.avi', fourcc, 25.0, (frame_width, frame_height))
    
    # Load the model
    vehicel_model = YOLO(path+'utils/yolov8n.pt')
    plate_detection = YOLO(path+'utils/car_plate_v2.pt')
    brand_detection = YOLO(path+'utils/brand.pt')
    
    # Initialize the OCR reader
    reader = easyocr.Reader(['en'], gpu=True)
    
    vehicles = {2:"car",5:'bus', 7:'truck'} # 2: 'car' ,3: 'motorcycle', 5: 'bus', 7: 'truck'

    #brand define
    brand= ['Audi', 'Chrysler', 'Citroen', 'GMC', 'Honda', 'Hyundai', 'Infiniti', 'Mazda', 'Mercedes', 'Mercury', 'Mitsubishi', 'Nissan', 'Renault', 'Toyota', 'Volkswagen', 'acura', 'bmw', 'cadillac', 'chevrolet', 'dodge', 'ford', 'jeep', 'kia', 'lexus', 'lincoln', 'mini', 'porsche', 'ram', 'range rover', 'skoda', 'subaru', 'suzuki', 'volvo']
    
    
    #create csv file
    create_csv(new_folder_path,name)
        
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            
            #copy a new frame for plotting 
            new_frame = frame.copy()
            
            # using YOLOV8 to detact the vehicle and the License plate

            # run the vehicle detatction
            vehicle_detaction_results = vehicel_model(frame)[0]
            
            for vehicle_detection in vehicle_detaction_results.boxes.data.tolist():
                vx1, vy1, vx2, vy2, vscore, vclass_id = vehicle_detection
                        
                #get the correct classes and the the predict score more that equal to 80%
                if int(vclass_id) in vehicles and vscore >= 0.8:
                    
                    
                    #crop out the vehicle frame    
                    vehicle_crop = frame[int(vy1):int(vy2), int(vx1): int(vx2), :]
                    
                    #predict the vehicle plate area    
                    car_plate_results = plate_detection(vehicle_crop)[0]
                    
                    drawbox(new_frame,int(vx1),int(vx2),int(vy1),int(vy2),f'{round(vscore,2)}',(255, 0, 0), 5)                    
                    for detection in car_plate_results.boxes.data.tolist():
                        px1,py1, px2, py2, pscore, classid = detection                       
                        
                        drawbox(new_frame,int(vx1+px1),int(vx1+px1+(px2-px1)),int(vy1+py1),int(vy1+py1+(py2-py1)),f"_{round(pscore,2)}",(0, 0, 255), 5)    
                        #run if the acuraccy predict is more that equal to % 
                        if pscore >= 0.8:
                            
                            #crop out the lp
                            lp_crop = vehicle_crop[int(py1):int(py2), int(px1): int(px2)]
                            
                            result = text_reconise(lp_crop,reader)
                            
                            if result[0]:
                                
                                #start brand detaction
                                v_brand  = "" 
                                
                                brand_detaction_results = brand_detection(vehicle_crop)[0]

                                for detection in brand_detaction_results.boxes.data.tolist():
                                    bx1,by1, bx2, by2, bscore, bclassid = detection
                                    drawbox(new_frame,int(bx1),int(bx2),int(by1),int(by2),f'{brand[int(bclassid)]}_{bscore}',(0, 255, 0), 5)

                                    if bscore >= 0.8:  
                                        v_brand = brand[int(bclassid)]
                                    
                                #define the csv file path
                                csv_file_path = f"{new_folder_path}/{name}.csv"
                                        
                                #insert data to the csv file
                                insert_csv([result[1],vehicles[vclass_id],v_brand],csv_file_path)
                                        
                                #save the image with the lp name.                                        
                                cv2.imwrite(f'{new_folder_path}/{result[1]}_{v_brand}.jpg', vehicle_crop)   
                                        
                                #plot lp
                                #drawbox(new_frame,int(vx1+px1),int(vx1+px1+(px2-px1)),int(vy1+py1),int(vy1+py1+(py2-py1)),f"{str(result[1])}_{round(result[2],2)}",(0, 0, 255), 5)
                                
                                #plot car
                                #drawbox(new_frame,int(vx1),int(vx2),int(vy1),int(vy2),f'{vehicles[vclass_id]}_{result[1]}',(255, 0, 0), 5)
                                
                                            
            
            out.write(new_frame)
            if show_output(new_frame): break             

        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    out.release() 
    cv2.destroyAllWindows()
    

def main():
    
    start_time = time.time()

    # Open the video file
    path = "F:/FYP_Programe/test2/"
    video_path = "F:/FYP_Programe/Data/video_raw/IMG_0141.MOV" #path+"IMG_0139.MOV" #total 11 car should be detacted

    #call the programe
    run(video_path,path)

    end_time = time.time()

    # Calculate the elapsed time
    running_time = end_time - start_time

    print("\tProgram running time:", running_time/60, "minutes")


if __name__ == '__main__':
    main()
