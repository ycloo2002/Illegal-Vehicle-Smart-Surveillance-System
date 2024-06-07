import time
import cv2
import easyocr
import re
import csv
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
from collections import deque, defaultdict
from PySide6.QtGui import QPixmap, QIcon,QImage
from PySide6.QtCore import QObject,Slot,Signal, QMutex, QMutexLocker,Qt
from PySide6.QtWidgets import (
    QApplication,
    QTableWidgetItem,
    QLabel
)

# Force matplotlib to not use any backend.
matplotlib.use('Agg')
    
def display_img(display,image):
    """display the image in the result page

    Args:
        display (_type_): get classes that contain the gui argument
        image (_type_): the image with the cv2 format
    """
    #turn the cv2 format to the pixmap format
    height, width, channel = image.shape
    bytes_per_line = 3 * width
    q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
    pixmap = QPixmap.fromImage(q_image)
    
    #set the image to the gui
    display.label_img.setPixmap(pixmap)

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

def data_and_time():
    """get the date and time as the '%Y-%m-%d %H-%M-%S' format

    Returns:
        str: Type of the date and time '%Y-%m-%d %H-%M-%S' 
    """
    date = datetime.now()
    return str(date.strftime('%Y-%m-%d %H-%M-%S')) 
    
def insert_csv(csv_path,data):
    """insert the text to the csv file

    Args:
        no (int): the no 
        data (dict): contain all the result data
        csv_path (str): the input csv file path
    """
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames= ['No','license_plate','type','brand','colour','warnning_message',"owner_name","owner_contact",'img_path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(data)   
    
def create_csv(path):
    """create the csv file for saving the detection result

    Args:
        path (str): the folder path

    Returns:
        str: the csv path that created
    """
    csv_file_path = f"{path}/result.csv"
    
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['No','license_plate','type','brand','colour','warnning_message',"owner_name","owner_contact",'img_path'])
        print(f"CSV file '{csv_file_path}' successful create.")
        
    return csv_file_path
                                             
def check_invalid_vehicle(data,path):
    """check the illegal vehicle and return any warnning message

    Args:
         data (dictionary): contain all the result data
        path (str): the database csv path

    Returns:
        bool: any illegal vehicle
        list: result_data
    """
    #default
    found_lp = False
    
    #open the database file
    with open(path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for col in reader:
            
            if col[2] == data['license_plate']: # true in vehicle plate 
                found_lp = True
                
                data['onwer_name'] = col[1] #get the onwer name
                
                #type
                if col[3] != data['type']: 
                    data['warnning_message'] +=f"<p>Invalid vehicle Type for this License Plate, the register type is <b>{col[3]}</b>.</p>"
                
                #brand
                if col[4] != data['brand']: 
                    data['warnning_message'] +=f"<p>Invalid vehicle brand for this License Plate, the register brand is <b>{col[4]}</b>.</p>"
                
                #color
                if col[5] != data['colour']: 
                    data['warnning_message'] += f"<p>Invalid vehicle colour for this License Plate, the register colour is <b>{col[5]}</b>.</p>"
                    
                #exp lp
                date_now = datetime.today()
                date_exp = datetime.strptime(col[6], '%Y-%m-%d')
                between_date = date_exp - date_now
                
                if  between_date.days < 0: 
                    data['warnning_message'] += f"<p>The road tax is expired, it expired <b>{-between_date.days}</b> days ago.</p>"
                    
                #warnning messege
                if col[7] != "": 
                    data['warnning_message'] += f"<p>This vehicle is <b>{col[7]}</b>.</p>"
                    
                break
        
        if found_lp and data['warnning_message'] == "":
            return False,data
        elif found_lp and not data['warnning_message']=="":
            return True,data
        else : 
            data['warnning_message'] = "This License Plate not register in the system."
            return True,data['warnning_message']
            
class Load_Object():
    """Load the nessasry item
    """
    def __init__(load):
        super().__init__()
        utils_basedir = os.path.dirname(__file__)
        
        print("\nload the nessary item")
        
        # Load the model
        load.vehicel_model = YOLO(f'{utils_basedir}/model/yolov8s.pt')
        print("\nSuccessfully load vehicle model")
        
        load.plate_detection = YOLO(f'{utils_basedir}/model/car_plate_v5.pt')
        print("\nSuccessfully load plate model")
        
        load.brand_detection = YOLO(f'{utils_basedir}/model/brand_v4.pt')
        print("\nSuccessfully load brand model")
        
        # Load the model for color recorigse
        color_model = models.googlenet(pretrained=False, aux_logits=True)  # Set aux_logits to True to match the saved model
        num_ftrs = color_model.fc.in_features
        color_model.fc = nn.Linear(num_ftrs, 15)  # Adjust num_classes to match your dataset
        color_model.aux1.fc2 = nn.Linear(color_model.aux1.fc2.in_features, 15)
        color_model.aux2.fc2 = nn.Linear(color_model.aux2.fc2.in_features, 15)

        model_path = f'{utils_basedir}/model/colour.pth'
        color_model.load_state_dict(torch.load(model_path))
        
        if torch.cuda.is_available(): load.device = torch.device('cuda')
        else : load.device = torch.device('cpu')
        
        color_model = color_model.to(load.device)
        color_model.eval()  # Set the model to evaluation mode
        
        load.color_model = color_model
        print("\nSuccessfully load colour model")
        
        # Initialize the OCR reader
        load.reader = easyocr.Reader(['en'], gpu=True)
        print("\nSuccessfully load reader model")
        
        #brand define
        #load.brand= ['Audi', 'Chrysler', 'Citroen', 'GMC', 'Honda', 'Hyundai', 'Infiniti', 'Mazda', 'Mercedes', 'Mercury', 'Mitsubishi', 'Nissan', 'Renault', 'Toyota', 'Volkswagen', 'acura', 'bmw', 'cadillac', 'chevrolet', 'dodge', 'ford', 'jeep', 'kia', 'lexus', 'lincoln', 'mini', 'porsche', 'ram', 'range rover', 'skoda', 'subaru', 'suzuki', 'volvo','Proton','Perodua','no class']
        #print("\nSuccessfully load brand class : ",load.brand)
        
        load.database_path = f'{utils_basedir}/database.csv'
        print("\nSuccessfully load Database path : ",load.database_path)
        
        
        if not os.path.exists("result"):
            os.makedirs("result")
            print("Folder save created.")
    
class Detection(QObject):
    """ The main vehicle detection classes.
        Include the running detection with live,video and image.
    """
    warnning = Signal(str)
    insert_data = Signal(dict,bool)
    finish = Signal(str)
    
    def __init__(load,define_object,source_path):
        super().__init__()

        for attr, value in define_object.load_object.__dict__.items():
            setattr(load, attr, value)
        load.gui = define_object
        load.detact_input = source_path
        
        load._is_running = True
        load.mutex = QMutex()

    def stop(load):
        """stop the detection

        Args:
            load (_type_): _description_
        """
        QMutexLocker(load.mutex)
        load._is_running = False
        
    def open_folder_csv(load):
        """Generate the folder and csv file

        Args:
            load (_type_): _description_
        """
        load.folder_name = data_and_time()
        #open new folder
        load.new_folder_path = f'result/{load.folder_name}'
        os.makedirs(load.new_folder_path)
        
        #open folder for save the crop image
        os.makedirs(f'{load.new_folder_path}/crop')
        
        #create csv file
        load.csv_file_path = create_csv(load.new_folder_path)
    
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
        vehicles = {2:"car",5:'bus', 7:'truck'} 
        for detection in car_results.boxes.data.tolist():
            cx1,cy1, cx2, cy2, pscore, classid = detection 
            if classid == 3:
                break
            
            if plate[0] >= cx1 and plate[1] >= cy1 and plate[2] <= cx2 and plate[3] <= cy2:
                vehicle_detect = [cx1,cy1, cx2, cy2, vehicles[classid]]
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

    def search_brand(load,frame):
        """Function to detact vehicle brand based on the given input

        Args:
            load (_type_): _description_
            frame (_type_): input

        Returns:
            str: brand name
        """

        brand_detaction_results = load.brand_detection(frame)[0]

        for detection in brand_detaction_results.boxes.data.tolist():
            bx1,by1, bx2, by2, bscore, bclassid = detection

            return load.brand_detection.names[int(bclassid)]

        return "Error"
            
    def color_reconigse(load,frame):
        """Function to detect the vehicle colour

        Args:
            load (_type_): _description_
            frame (_type_): input

        Returns:
            str: colour
        """
        
        colour_class = ['beige','black','blue','brown','gold','green','grey','orange','pink','purple','red','silver','tan','white','yellow']
        
        # Preprocess the image
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        image = transform(pil_image).unsqueeze(0)  # Add batch dimension

        # Move the image to the appropriate device
        image = image.to(load.device)

        # Set the model to evaluation mode and make prediction
        load.color_model.eval()
        with torch.no_grad():
            outputs = load.color_model(image)
            _, predicted = torch.max(outputs, 1)

        return colour_class[predicted.item()]

    def vehicle_illegal_detection(load,frame):
        """Function to call and run detection and prediction for the vehicle license plate,vehicle type,vehicle brand and vehicle colour.
        It also will call the function to determine the prediction data compare with the database given.
        It will showing the output as the gui and save the crop vehicle to the folder that create.
        Beside it also saving the result to the csv file for further view.
            
        Args:
            load (_type_): _description_
            frame(_type_): input

        """
        
        display_img(load.gui,frame) # display the image
            
        plate_detect = load.search_plate(frame) #detect the LP
        
        if len(plate_detect) > 0: #continue when any return value from the search_plate function
            
            #save result to both array
            #when gettin
            
            
            
            for plate in plate_detect:
                
                load.save_plate.append(plate[4]) #save the lp
                
                vehicle_detect = load.search_vehicle(frame,plate) #search for the vehicle base on the LP

                if not vehicle_detect == "":
                    #load the position
                    x1 = vehicle_detect[0]
                    y1 = vehicle_detect[1]
                    
                    x2 = vehicle_detect[2]
                    y2 = vehicle_detect[3]
                    
                    #crop the vehicle        
                    vehicle_crop = frame[int(y1):int(y2), int(x1): int(x2)]
                    
                    #detect the brand
                    detect_brand = load.search_brand(frame)  
                    
                    #detect vehicle colour           
                    detect_colour = load.color_reconigse(frame)
                    
                    #save crop image path
                    img_path = f'{load.new_folder_path}/crop/{plate[4]}.jpg'
                    
                    #combine result
                    result_data = {
                        'no':len(load.save_plate),
                        'license_plate':plate[4],
                        "type":vehicle_detect[4],
                        "brand":detect_brand,
                        "colour":detect_colour,
                        "warnning_message":"",
                        "owner_name":"",
                        "owner_contact":"",
                        "img_path":img_path
                        }
                            
                    #save the image with the lp name.                                        
                    cv2.imwrite(img_path, vehicle_crop)  
                    
                    load.insert_data.emit(result_data,False)            
                    #insert_table_info(load.gui,result_data) # adding result to the gui
                                    
                    #check the illger vehicle and return warnning message if illger
                    invalid,result_data = check_invalid_vehicle(result_data,load.database_path)
                                    
                    if invalid:
                        load.insert_data.emit(result_data,invalid)     
                        #insert_table_info(load.gui,result_data,invalid)
                        load.total_warnning +=1
                                    
                    #insert data to the csv file
                    insert_csv(load.csv_file_path,result_data)
                                
                    QApplication.processEvents() 
                    return 0
                    #drawbox(new_frame,int(vehicle[0]),int(vehicle[2]),int(vehicle[1]),int(vehicle[3]),f'{plate[4]}',(255, 0, 0), 5) 
        
        return  5                                                                                                                
    
    @Slot()
    def video_detaction(load):
        """This is the video detection function.It will get the input and covert the video to frame. Each frame will be call the run_detection to get the detection.
            Beside, it also will save the input as the new video after making some plotting.
        Args:
            load (_type_): _description_
        """
        
        #get the file path and csv path
        load.open_folder_csv()
        
        start_time = time.time()
        
        #open video
        cap = cv2.VideoCapture(load.detact_input)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get the video frame width and height
        #frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object 
        #fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        #out = cv2.VideoWriter(f'{load.new_folder_path}/result_video.avi', fourcc, 25.0, (frame_width, frame_height))
        
        load.save_plate = []   
        load.total_warnning = 0
        load.lp_5_most = deque(maxlen=5)
        skip_frame = 0 
        # Loop through the video frames
        while cap.isOpened():
            
            QMutexLocker(load.mutex)  # Ensure thread-safe access to _is_running
            if not load._is_running:
                print("Worker stopped.")
                break
            
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                if skip_frame != 0: 
                    skip_frame -= 1
                    print("skip_frame")
                else :
                    skip_frame=load.vehicle_illegal_detection(frame) #call the function and return the new_version_frame
                    load.gui.runing_text.setText(f"Loading. \n Total {len(load.save_plate)} vehicle detacted and \n{load.total_warnning} is detacted as illegel vehicle.")                                     
                #out.write(new_frame)# add the frame to the video.
            else:
                # Break the loop if the end of the video is reached
                break
        
        # Release the video capture object and close the display window
        cap.release()
        #out.release() 
        
        
        # Calculate the elapsed time
        running_time =time.time()- start_time

        print(f"Total {len(load.save_plate)} vehicle detacted and {load.total_warnning} is detacted as illegel vehicle")
        print("\nProgram running time:", running_time/60, "minutes")
        
        load.gui.runing_text.setText(f"End,Total {len(load.save_plate)} vehicle detacted and \n{load.total_warnning} is detacted as illegel vehicle \nTime Taken : {round(running_time/60 , 2)} minutes")
        load.gui.text_container.setStyleSheet("background-color:transparent;")
        load.gui.runing_text.setStyleSheet("color:black")
        load.gui.result_home_btn.setEnabled(True)
        load.gui.stop_running_btn.setEnabled(False)
        load.finish.emit(load.folder_name)

    @Slot()
    def image_detaction(load):
        """This is the image detection function.It will get the input and call the run_detection to get the detection.
            Beside, it also will save the input.
        Args:
            load (_type_): _description_
        """
        load.total_vehicle = 0  
        
        load.total_warnning = 0
        
        start_time = time.time()
        #get the file path and csv path
        load.open_folder_csv()
        
        frame = cv2.imread(load.detact_input)
        cv2.imwrite(f'{load.new_folder_path}/original.png', frame)
        load.save_plate = []  
        load.vehicle_illegal_detection()
        
        end_time = time.time()

        # Calculate the elapsed time
        running_time = end_time - start_time
        
        print(f"Total {load.total_vehicle} vehicle detacted and {load.total_warnning} is detacted as illegel vehicle")
        print("\nProgram running time:", running_time/60, "minutes")
        
        load.gui.runing_text.setText(f"detection End. \n Total {load.total_vehicle} vehicle detacted and \n{load.total_warnning} is detacted as illegel vehicle \nTime Taken : {round(running_time/60 , 2)} minutes")
        load.gui.text_container.setStyleSheet("background-color:lightgreen;")
        load.gui.result_home_btn.setEnabled(True)
        load.gui.stop_running_btn.setEnabled(False)
        load.finish.emit(load.folder_name)
    
    @Slot()
    def live_detaction(load):
        """This is the live detection function.It will get the input and covert the video to frame. Each frame will be call the run_detection to get the detection.
            Beside, it also will save the input as the new video after end the live
        Args:
            load (_type_): _description_
        """
        #open video
        cap = cv2.VideoCapture(0)
        load.save_plate = []  
        if not cap.isOpened():
            print("Error: Could not open video stream.")
            load.warnning.emit("Unable to open the camera")
            
        else:
            load.total_vehicle = 0  
        
            load.total_warnning = 0
            
            start_time = time.time()
            
            #get the file path and csv path
            load.open_folder_csv()
            
            
            # Get the video frame width and height
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Define the codec and create VideoWriter object 
            fourcc = cv2.VideoWriter_fourcc(*'XVID') 
            out = cv2.VideoWriter(f'{load.new_folder_path}/result_video.avi', fourcc, 25.0, (frame_width, frame_height))
        
            while True:
                QMutexLocker(load.mutex)  # Ensure thread-safe access to _is_running
                if not load._is_running:
                    print("Worker stopped.")
                    break
            
                # Capture frame-by-frame
                ret, frame = cap.read()

                if not ret:
                    print("Failed to grab frame")
                    break
                       
                new_frame = load.vehicle_illegal_detection() #call the function and return the new_version_frame
                                                                            
                out.write(new_frame)# add the frame to the video.
                
                # Press 'q' on the keyboard to exit the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            # Release the video capture object and close the display window
            cap.release()
            out.release() 
            
            end_time = time.time()

            # Calculate the elapsed time
            running_time = end_time - start_time

            print(f"Total {load.total_vehicle} vehicle detacted and {load.total_warnning} is detacted as illegel vehicle")
            print("\nProgram running time:", running_time/60, "minutes")
            
            load.gui.runing_text.setText(f"detection End. \n Total {load.total_vehicle} vehicle detacted and \n{load.total_warnning} is detacted as illegel vehicle \nTime Taken : {round(running_time/60 , 2)} minutes")
            load.gui.text_container.setStyleSheet("background-color:lightgreen;")
            load.gui.result_home_btn.setEnabled(True)
            load.gui.stop_running_btn.setEnabled(False)
            load.finish.emit(load.folder_name)
        
