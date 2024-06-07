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

def drawbox(img,x1,x2,y1,y2,label_text="",color = (255, 0, 0),thickness = 5):
    """draw the rectangle box to the image with label

    Args:
        img (_type_): image input
        x1 (int): x1 location
        x2 (int): x2 location
        y1 (int): y1 location
        y2 (int): y2 location
        label_text (str): the label. Defaults to "".
        color (tuple, optional): color of the line . Defaults to (255, 0, 0).
        thickness (int, optional): the thickness of line . Defaults to 5.
    """
    start_point = (x1, y1)
    end_point = (x2, y2)
    label_text = str(label_text)

    # Draw the rectangle on the image
    cv2.rectangle(img, start_point, end_point, color, thickness)
    
    if label_text != "":
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
        fieldnames= ['No','license_plate','type','brand','colour','warnning_message','owner_name','owner_contact','img_path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({k: data.get(k, '') for k in fieldnames})   
    
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
        writer.writerow(['No','license_plate','type','brand','colour','warnning_message','owner_name','owner_contact','img_path'])
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
        reader = csv.DictReader(csvfile)
        for col in reader:
            
            if col['Licence_Plate_Number'] == data['license_plate']: # true in vehicle plate 
                found_lp = True
                
                data['owner_name'] = col['Vehicle_owner'] #get the owner name
                data['owner_contact'] = col['Contact_number'] #get the owner name
                
                #type
                if col['Register_Vehicle_Type'] != data['type']: 
                    data['warnning_message'] +=f"<p>Invalid vehicle Type for this License Plate, the register type is <b>{col['Register_Vehicle_Type']}</b>.</p>"
                
                #brand
                if col['Register_Vehicle_Brand'] != data['brand']: 
                    data['warnning_message'] +=f"<p>Invalid vehicle brand for this License Plate, the register brand is <b>{col['Register_Vehicle_Brand']}</b>.</p>"
                
                #color
                if col['Register_Vehicle_Colour'] != data['colour']: 
                    data['warnning_message'] += f"<p>Invalid vehicle colour for this License Plate, the register colour is <b>{col['Register_Vehicle_Colour']}</b>.</p>"
                    
                #exp lp
                print(col['Road_tax_exp_date'])
                date_now = datetime.today()
                date_exp = parse_date(col['Road_tax_exp_date'])
                between_date = date_exp - date_now
                
                if  between_date.days < 0: 
                    data['warnning_message'] += f"<p>The road tax is expired, it expired <b>{-between_date.days}</b> days ago.</p>"
                    
                #warnning messege
                if col['Message'] != "": 
                    data['warnning_message'] += f"<p>This vehicle is <b>{col['Message']}</b>.</p>"
                    
                break
        
        if found_lp and data['warnning_message'] == "":
            return False,data
        elif found_lp and not data['warnning_message']=="":
            return True,data
        else : 
            data['warnning_message'] = "<p>This License Plate <b>not register</b> in the system.<p>"
            return True,data
 
def get_the_most_frequent(predict_list):
    """
    Function to get the most frequent base on the array given
        Args:
            predict_list (array): input

        Returns:
            str: most frequent
    """ 
    # Count frequencies of elements
    freq = defaultdict(int)
        
    for elem in predict_list:
        if elem in freq:
            freq[elem] += 1
        else:
            freq[elem] = 1

    # Find the element with the highest frequency
    max_freq_elem = max(freq, key=freq.get)
    return max_freq_elem

def parse_date(date_str):
    """function to change the date format

    Args:
        date_str (str): input

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    for fmt in ('%d/%m/%Y', '%Y-%m-%d'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Date {date_str} does not match any expected format")
             
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
        color_model.eval()  # Set the model to evaluation mode
        load.color_model = color_model
        load.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
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
    pop_illegal =Signal(str)
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
        """Function to read the text from the input given. It will fillter with the malaysia plate pattern and the score with above 80%

        Args:
            load (_type_): _description_
            img (_type_): _description_

        Returns:
            _type_: _description_
        """
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
        """Function to find the vehicle and vehicle type based on the position of the plate

        Args:
            load (_type_): _description_
            frame (_type_): input
            plate (dist): the info of the plate

        Returns:
            dist : the vehicle information
        """
        vehicle_detect =""
        car_results = load.vehicel_model(frame,classes=[2,3,5,7])[0]
        vehicles = {2:"car",5:'bus', 7:'truck'} 
        for detection in car_results.boxes.data.tolist():
            cx1,cy1, cx2, cy2, pscore, classid = detection 
            if classid == 3:
                break
            
            if plate['x1'] >= cx1 and plate['y1'] >= cy1 and plate['x2'] <= cx2 and plate['y2'] <= cy2:
                vehicle_detect = {"x1":cx1,"y1":cy1,"x2":cx2,"y2":cy2,"type":vehicles[classid]}
                break
                
        return vehicle_detect

    def search_plate(load,frame):
        """function to search the plate

        Args:
            load (_type_): _description_
            frame (_type_): input

        Returns:
            _type_: detectect plate
        """
        plate_detect = []
        
        car_plate_results = load.plate_detection(frame)[0]                   
        for detection in car_plate_results.boxes.data.tolist():
            px1,py1, px2, py2, pscore, classid = detection                       
                            
            lp_crop = frame[int(py1):int(py2), int(px1): int(px2)]    

            reader_result,result = load.text_reader(lp_crop)
            if reader_result:
                plate_detect.append({'x1':px1,'y1':py1,'x2':px2,'y2':py2,'lp':result})  
                        
        return plate_detect

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
        
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        image = load.transform(pil_image).unsqueeze(0)  # Add batch dimension

        # Set the model to evaluation mode and make prediction
        load.color_model.eval()
        with torch.no_grad():
            outputs = load.color_model(image)
            _, predicted = torch.max(outputs, 1)

        return colour_class[predicted]
    
    def vehicle_illegal_detection(load,frame,img=False):
        """Function to call and run detection and prediction for the vehicle license plate,vehicle type,vehicle brand and vehicle colour.
        It also will call the function to determine the prediction data compare with the database given.
        It will showing the output as the gui and save the crop vehicle to the folder that create.
        Beside it also saving the result to the csv file for further view.
            
        Args:
            load (_type_): _description_
            frame(_type_): input
            img(bool):check detect image or not
        return
            int: skip frame
            _type_ : new plotting frame
        """
        frame_skip = 0
        new_frame = frame.copy()
            
        plate_detect = load.search_plate(frame) #detect the LP
        
        if len(plate_detect) > 0: #continue when any return value from the search_plate function

            for plate in plate_detect:
                
                same_lp = False
                for plate_no in load.save_plate:
                    if plate['lp'] == plate_no:
                        same_lp = True
                        break
                
                if not same_lp : 
                    
                    vehicle_detect = load.search_vehicle(frame,plate) #search for the vehicle base on the LP

                    if not vehicle_detect == "":

                        #crop the vehicle        
                        vehicle_crop = frame[int(vehicle_detect['y1']):int(vehicle_detect['y2']), int(vehicle_detect['x1']): int(vehicle_detect['x2'])]
                        
                        #detect the brand
                        detect_brand = load.search_brand(vehicle_crop)  
                        
                        #detect vehicle colour           
                        detect_colour = load.color_reconigse(vehicle_crop)
                        
                        #combine result
                        result_data = {
                            'license_plate':plate['lp'],
                            "type":vehicle_detect['type'],
                            "brand":detect_brand,
                            "colour":detect_colour,
                            "vehicle_crop":vehicle_crop,
                            "x1":int(vehicle_detect['x1']),
                            "y1":int(vehicle_detect['y1']),
                            "x2":int(vehicle_detect['x2']),
                            "y2":int(vehicle_detect['y2'])
                            }
                        
                        drawbox(new_frame,result_data['x1'],result_data['x2'],result_data['y1'],result_data['y2'],color = (255, 0, 0))
                        load.lp_5_most.append(plate['lp'])
                        load.save_all_result.append(result_data)        
                               
                if (len(load.save_all_result) > 4 and not same_lp)or img:
                    lp = get_the_most_frequent(load.lp_5_most)
                    
                    for plate_no in load.save_plate:
                        if lp == plate_no:
                            same_lp = True
                            break
                        
                    if same_lp:
                        break
                    
                    load.save_plate.append(lp)
                    p_type = []
                    p_brand = []
                    p_colour = []
                            
                    for data in load.save_all_result:
                        if data['license_plate'] == lp:
                            p_type.append(data['type'])
                            p_brand.append(data['brand'])
                            p_colour.append(data['colour'])
                            vehicle_crop = data['vehicle_crop']
                                    
                    mf_type = get_the_most_frequent(p_type)
                    mf_brand = get_the_most_frequent(p_brand)
                    mf_colour = get_the_most_frequent(p_colour)
        
                    result_data = {
                                "No":len(load.save_plate),
                                "license_plate":lp,
                                "type":mf_type,
                                "brand":mf_brand,
                                "colour":mf_colour,
                                "warnning_message":"",
                                "owner_name":"-",
                                "owner_contact":"-",
                                "img_path":f'{load.new_folder_path}/crop/{lp}.jpg'
                    }
                            
                    #save the image with the lp name.                                        
                    cv2.imwrite(result_data['img_path'], vehicle_crop)  
                                    
                    load.insert_data.emit(result_data,False)            
                                                    
                    #check the illger vehicle and return warnning message if illger
                    invalid,result_data = check_invalid_vehicle(result_data,load.database_path)
                                                    
                    if invalid:
                        load.insert_data.emit(result_data,invalid)        
                        load.total_warnning +=1
                        message = f"Vehicle with plate {lp}.\n {result_data['warnning_message']}"
                        load.pop_illegal.emit(message) 
                                                    
                    #insert data to the csv file
                    insert_csv(load.csv_file_path,result_data)
                    QApplication.processEvents() 
        
        else: frame_skip = 5
        
        display_img(load.gui,new_frame) # display the image
        return frame_skip,new_frame
    
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
        
        #Get the video frame width and height
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object 
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        out = cv2.VideoWriter(f'{load.new_folder_path}/result_video.avi', fourcc, 20.0, (frame_width, frame_height))
        
        load.save_plate = []   
        load.total_warnning = 0
        load.lp_5_most = deque(maxlen=5)
        load.save_all_result=deque(maxlen=5)
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
                else :
                    skip_frame,new_frame=load.vehicle_illegal_detection(frame) #call the function and return the new_version_frame
                    load.gui.runing_text.setText(f"Loading. \n Total {len(load.save_plate)} vehicle detacted and \n{load.total_warnning} is detacted as illegel vehicle.")                                     
                    out.write(new_frame)
            else:
                # Break the loop if the end of the video is reached
                break
        
        # Release the video capture object and close the display window
        cap.release()
        out.release() 
        
        
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
        
        load.save_plate = []   
        load.total_warnning = 0
        load.lp_5_most = deque(maxlen=5)
        load.save_all_result=deque(maxlen=5)
        
        start_time = time.time()
        #get the file path and csv path
        load.open_folder_csv()
        
        frame = cv2.imread(load.detact_input)
        
        
        load.save_plate = []  
        _,new_frame = load.vehicle_illegal_detection(frame,img=True)
        cv2.imwrite(f'{load.new_folder_path}/result.png', new_frame)
        end_time = time.time()

        # Calculate the elapsed time
        running_time = end_time - start_time
        
        print(f"Total {len(load.save_plate)} vehicle detacted and {load.total_warnning} is detacted as illegel vehicle")
        print("\nProgram running time:", running_time/60, "minutes")
        
        load.gui.runing_text.setText(f"End,Total {len(load.save_plate)} vehicle detacted and \n{load.total_warnning} is detacted as illegel vehicle \nTime Taken : {round(running_time/60 , 2)} minutes")
        load.gui.text_container.setStyleSheet("background-color:transparent;")
        load.gui.runing_text.setStyleSheet("color:black")
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
        cap.set(cv2.CAP_PROP_FPS, 30)
        load.save_plate = []  
        if not cap.isOpened():
            print("Error: Could not open video stream.")
            load.warnning.emit("Unable to open the camera")
            
        else:
            load.save_plate = []   
            load.total_warnning = 0
            load.lp_5_most = deque(maxlen=5)
            load.save_all_result=deque(maxlen=5)
            skip_frame = 0 
                
            start_time = time.time()
            
            #get the file path and csv path
            load.open_folder_csv()
            
            # Get the video frame width and height
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Define the codec and create VideoWriter object 
            fourcc = cv2.VideoWriter_fourcc(*'XVID') 
            out = cv2.VideoWriter(f'{load.new_folder_path}/result_video.avi', fourcc, 20.0, (frame_width, frame_height))
        
            while True:
                QMutexLocker(load.mutex)  # Ensure thread-safe access to _is_running
                if not load._is_running:
                    print("Worker stopped.")
                    break
            
                # Capture frame-by-frame
                ret, frame = cap.read()

                if not ret:
                    print("Failed to grab frame")
                    load.warnning.emit("Failed to grab frame")
                    break
                       
                if skip_frame != 0: 
                    skip_frame -= 1
                else :
                    skip_frame,new_frame=load.vehicle_illegal_detection(frame) #call the function and return the new_version_frame
                    load.gui.runing_text.setText(f"Loading. \n Total {len(load.save_plate)} vehicle detacted and \n{load.total_warnning} is detacted as illegel vehicle.")                                                                                                                 
                    out.write(new_frame)# add the frame to the video.
                
            # Release the video capture object and close the display window
            cap.release()
            out.release() 
            
            end_time = time.time()

            # Calculate the elapsed time
            running_time = end_time - start_time

            print(f"Total {len(load.save_plate)} vehicle detacted and {load.total_warnning} is detacted as illegel vehicle")
            print("\nProgram running time:", running_time/60, "minutes")
            
            load.gui.runing_text.setText(f"End,Total {len(load.save_plate)} vehicle detacted and \n{load.total_warnning} is detacted as illegel vehicle \nTime Taken : {round(running_time/60 , 2)} minutes")
            load.gui.text_container.setStyleSheet("background-color:transparent;")
            load.gui.runing_text.setStyleSheet("color:black")
            load.gui.result_home_btn.setEnabled(True)
            load.gui.stop_running_btn.setEnabled(False)
            load.finish.emit(load.folder_name)
        
