import sys
from utils.detact import Detaction,Load_Object
from PySide6.QtCore import QSize, Qt,Slot,QThread
from PySide6.QtGui import QFont,QIcon,QPixmap,QColor,QPainter
from functools import partial
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QStackedWidget,
    QFileDialog,
    QTableWidget,
    QAbstractItemView,
    QHeaderView,
    QMessageBox,
    QTableWidgetItem
)
import os
import csv

#import setup_env

try:
    from ctypes import windll  # Only exists on Windows.
    myappid = 'mycompany.myproduct.subproduct.version'
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except ImportError:
    pass

basedir = os.path.dirname(__file__)

FF = 'Verdana'
    
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.table_setting = """
            
            QHeaderView::section 
            { 
                background-color: gray; 
            }
            QTableWidget::item:!alternate {
                background-color: #f2f2f2; /* Light gray for odd rows */
            }
            QTableWidget::item:selected {
                background-color: lightblue; /* Change selection color */
                color: black; /* Ensure selected text color is black */
            }
            QTableWidget::item:alternate {
                background-color: white; /* White for even rows */
            }
        """
        
        self.setWindowTitle("Illegal Vehicle Smart Surveillance")
        self.setStyleSheet("background-color: #add8e6;")
        
        # Create stacked widget to hold pages
        self.stacked_widget = QStackedWidget(self)
        
        # Create pages
        self.home = QWidget()
        self.input = QWidget()
        self.result = QWidget()
        self.history = QWidget()
        self.history_details = QWidget()
        
        # Add widgets to pages
        self.init_home()
        self.init_input()
        self.init_result()
        self.init_history()
        self.init_history_details()
        
        # Add pages to stacked widget
        self.stacked_widget.addWidget(self.home)
        self.stacked_widget.addWidget(self.input)
        self.stacked_widget.addWidget(self.result)
        self.stacked_widget.addWidget(self.history)
        self.stacked_widget.addWidget(self.history_details)
        
        self.setCentralWidget(self.stacked_widget)
        
        self.load_object = Load_Object()
            
    def init_home(self):
        layout = QVBoxLayout(self.home)
        #title 
        self.title = QLabel("Illegal Vehicle Smart Surveillance")
        self.title.setFont(QFont(FF, 30))
        self.title.setAlignment(Qt.AlignHCenter)
        self.title.setFixedSize(1000, 100)
        
        title_box = QVBoxLayout()
        title_box.addWidget(self.title)
        title_box.setAlignment(Qt.AlignHCenter)
        
        layout.addLayout(title_box)
        
        #start btn
        button = QPushButton("Start")
        button.clicked.connect(self.go_to_input)
        button.setFont(QFont(FF, 12))
        button.setFixedSize(150, 50)
        
        button_style = """
            QPushButton:hover {
                background-color: transparent;
                color: black;
                font-weight: bold;
                border-color: white;
            }
            QPushButton {
                background-color: #1c1cf0;
                border: 2px solid white;
                border-radius: 20px;
                color: white;
            }
        """
        
        button.setStyleSheet(button_style)
        
        button_box = QVBoxLayout()
        button_box.addWidget(button)
        button_box.setAlignment(Qt.AlignHCenter)
        
        layout.addLayout(button_box)
        
        #version
        version = QLabel("V 2.0 beta")
        version.setFont(QFont(FF, 11))
        #version.setAlignment(Qt.AlignRight)
        version.setFixedSize(100, 20)
        
        v_box = QVBoxLayout()
        v_box.addWidget(version)
        v_box.setAlignment(Qt.AlignRight)
        
        layout.addLayout(v_box)
        
    def init_input(self):
        layout = QVBoxLayout(self.input)

        # Create a back button
        back_button = QPushButton("", self.input)
        back_button.clicked.connect(self.back_to_home)
        
        icon = QIcon("utils\img\previous.png")  # Replace with any icon name from the list
        back_button.setIcon(icon)
        back_button.setIconSize(icon.actualSize(back_button.sizeHint()))
        back_button.setStyleSheet("background-color: transparent;")
        
        # Create a container widget to hold the back button
        container_layout = QVBoxLayout()
        layout.addWidget(back_button)
        layout.setAlignment(back_button, Qt.AlignTop | Qt.AlignLeft)
            
        #layout.addLayout(container_layout)
        
        #title
        title = QLabel("Choose Input Type")
        title.setFont(QFont(FF, 18))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title, alignment=Qt.AlignCenter)
        
        
        #define the button style
        button_style = """
                QPushButton {
                    background-color: skyblue;
                    color: white;
                    border: 2px solid white;
                    border-radius: 20px;
                }
                QPushButton:hover {
                    background-color: blue;
                    border-color: darkblue;
                    font-weight: bold;
                }
            """
            
        button_camera = QPushButton("Camere")
        button_camera.clicked.connect(self.live_camera)
        button_camera.setFont(QFont(FF, 15))
        button_camera.setFixedSize(200, 50)
        button_camera.setStyleSheet(button_style)
        
        button_v_i = QPushButton("Video/Image")
        button_v_i.clicked.connect(self.input_video_img)
        button_v_i.setFont(QFont(FF, 15))
        button_v_i.setFixedSize(200, 50)
        button_v_i.setStyleSheet(button_style)
        
        button_history = QPushButton("History")
        button_history.clicked.connect(self.go_to_history)
        button_history.setFont(QFont(FF, 15))
        button_history.setFixedSize(200, 50)
        button_history.setStyleSheet(button_style)
        
        button_box = QVBoxLayout()
        button_box.addWidget(button_camera)
        button_box.addWidget(button_v_i)
        button_box.addWidget(button_history)
        button_box.setAlignment(Qt.AlignCenter)
        
        # Add button layout to main layout
        layout.addStretch()
        layout.addLayout(button_box)
        layout.addStretch()   
   
    def init_result(self):
        
        self.layout_result = QVBoxLayout(self.result)
        
        # section 1
        h_layout = QHBoxLayout()
        
        t_layout = QVBoxLayout()
        
        self.label_result_table = QLabel("Detact Result")
        self.label_result_table.setAlignment(Qt.AlignCenter)
        self.label_result_table.setStyleSheet("font-size: 20px; font-weight: bold; padding: 5px;")
        t_layout.addWidget(self.label_result_table)
        
        self.table_info = QTableWidget()

        self.table_info.setColumnCount(5)
        self.table_info.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_info.setHorizontalHeaderLabels(['Vehicle_Image',"License_Plate","Vehicle_Type", "Vehicle_Brand",'Vehicle_Colour'])

        self.table_info.setAlternatingRowColors(True)
        
        self.table_info.setStyleSheet(self.table_setting)
        
        t_layout.addWidget(self.table_info)
        
        h_layout.addLayout(t_layout)
        
        #image
        self.label_img = QLabel(self)

        h_layout.addWidget(self.label_img)
        
        
        self.layout_result.addLayout(h_layout)
        
        # section 2
        self.label_table_warnning = QLabel("Warning")
        self.label_table_warnning.setAlignment(Qt.AlignCenter)
        self.label_table_warnning.setStyleSheet("font-size: 20px; font-weight: bold; padding: 5px;")
        self.layout_result.addWidget(self.label_table_warnning)
        
        
        self.table_warnning = QTableWidget()
        self.table_warnning.setColumnCount(7)
        self.table_warnning.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_warnning.setHorizontalHeaderLabels(['Vehicle_Image',"License_Plate","Vehicle_Type", "Vehicle_Brand",'Vehicle_Colour','Warning Message','Vehicle_Onwner'])
        self.table_warnning.setAlternatingRowColors(True)
        self.table_warnning.setStyleSheet(self.table_setting)
        
        # Set the column sizes
        header = self.table_warnning.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Fixed)
        header.resizeSection(5, 400) 
        
        self.layout_result.addWidget(self.table_warnning)
        
        # section 3 
        hh_text = QHBoxLayout()
        
        self.text_container = QWidget()
        self.text_container.setStyleSheet("background-color:red;")
        
        f_layout = QVBoxLayout(self.text_container)
        
        self.runing_text = QLabel("Loading")
        self.runing_text.setFont(QFont(FF, 20))
        self.runing_text.setAlignment(Qt.AlignHCenter)
        self.runing_text.setStyleSheet("text-align: center;margin: 10px 2px;color:white")
        
        f_layout.addWidget(self.runing_text)
        
        hh_text.addWidget(self.text_container)
        
        #for btn
        btn_layout = QVBoxLayout()
        
        #btn for stop the programe
        self.stop_running_btn = QPushButton("Stop")
        self.stop_running_btn.clicked.connect(self.stop_task)
        self.stop_running_btn.setFont(QFont(FF, 12))
        self.stop_running_btn.setFixedSize(150, 50)
            
        button_style = """
                QPushButton {
                    background-color: red;
                    color: white;
                    border: 2px solid white;
                    border-radius: 20px;
                }
                QPushButton:hover {
                    background-color: darkred;
                    border-color: white;
                    font-weight: bold;
                }
            """
        self.stop_running_btn.setStyleSheet(button_style)
        button_box = QVBoxLayout()
        button_box.addWidget(self.stop_running_btn )
        button_box.setAlignment(Qt.AlignHCenter)
        btn_layout.addWidget(self.stop_running_btn )
        
        #btn for back to home page
        self.result_home_btn = QPushButton("Home")
        self.result_home_btn.clicked.connect(self.back_to_home)
        self.result_home_btn.setFont(QFont(FF, 12))
        self.result_home_btn.setFixedSize(150, 50)
        self.result_home_btn.setEnabled(False)     
        button_style = """
                QPushButton {
                    background-color: lightgreen;
                    color: white;
                    border: 2px solid white;
                    border-radius: 20px;
                }
                QPushButton:hover {
                    background-color: green;
                    border-color: white;
                    font-weight: bold;
                }
            """
        self.result_home_btn .setStyleSheet(button_style)
        btn_layout.addWidget(self.result_home_btn )
        hh_text.addLayout(btn_layout)
        
        self.layout_result.addLayout(hh_text)     
        
    def init_history(self):
        self.layout_history = QVBoxLayout(self.history)

        # Create a back button
        back_button = QPushButton("", self.history)
        back_button.clicked.connect(self.go_to_input)
        
        icon = QIcon("utils\img\previous.png")  # Replace with any icon name from the list
        back_button.setIcon(icon)
        back_button.setIconSize(icon.actualSize(back_button.sizeHint()))
        back_button.setStyleSheet("background-color: transparent;")

        # Create a container widget to hold the back button
        container_layout = QVBoxLayout()
        container_layout.addWidget(back_button)
        container_layout.setAlignment(back_button, Qt.AlignTop | Qt.AlignLeft)
            
        self.layout_history.addLayout(container_layout)
        
        #start table
        self.history_table = QLabel("History")
        self.history_table.setAlignment(Qt.AlignCenter)
        self.history_table.setStyleSheet("font-size: 20px; font-weight: bold; padding: 5px;")
        self.layout_history.addWidget(self.history_table)
        
        self.table_history = QTableWidget()
        self.table_history.setColumnCount(4)
        #self.table_history.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_history.setHorizontalHeaderLabels(['Name',"Total Vehicle Detact","Total of Illegel vehicle",'Action'])
        self.table_history.setAlternatingRowColors(True)
        self.table_history.setStyleSheet(self.table_setting)
        
        self.layout_history.addWidget(self.table_history)
    
    def init_history_details(self):
        self.layout_history_details = QVBoxLayout(self.history_details)

        # Create a back button
        back_button = QPushButton("", self.history_details)
        back_button.clicked.connect(self.go_to_history)
        
        icon = QIcon("utils\img\previous.png")  # Replace with any icon name from the list
        back_button.setIcon(icon)
        back_button.setIconSize(icon.actualSize(back_button.sizeHint()))
        back_button.setStyleSheet("background-color: transparent;")

        # Create a container widget to hold the back button
        container_layout = QVBoxLayout()
        container_layout.addWidget(back_button)
        container_layout.setAlignment(back_button, Qt.AlignTop | Qt.AlignLeft)
            
        self.layout_history_details.addLayout(container_layout)
        
        #start table
        self.label_history_details_table = QLabel("History_details")
        self.label_history_details_table.setAlignment(Qt.AlignCenter)
        self.label_history_details_table.setStyleSheet("font-size: 20px; font-weight: bold; padding: 5px;")
        self.layout_history_details.addWidget(self.label_history_details_table)
        
        self.history_details_table = QTableWidget()
        self.history_details_table.setColumnCount(6)
        self.history_details_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.history_details_table.setHorizontalHeaderLabels(['Vehicle_Image',"License_Plate","Vehicle_Type", "Vehicle_Brand",'Vehicle_Colour','Warning Message'])
        self.history_details_table.setAlternatingRowColors(True)
        self.history_details_table.setStyleSheet(self.table_setting)
        
        self.layout_history_details.addWidget(self.history_details_table)
            
    def back_to_home(self):
        self.stacked_widget.setCurrentIndex(0)
        
    def go_to_input(self):
        self.stacked_widget.setCurrentIndex(1)
        
    def go_to_history(self):
        
        self.table_history.clearContents()  # Clear the cell contents
        self.table_history.setRowCount(0)
        
        save_folder_path = f"{basedir}/save"
        
        if os.path.isdir(save_folder_path):
            result_folder_path  = [f.path for f in os.scandir(save_folder_path) if f.is_dir()]
            result_folder_path.sort(reverse=True)
            
            if len(result_folder_path) == 0:
                print("No history found")
                self.table_history.setRowCount(1)     
                self.table_history.setItem(0, 0, QTableWidgetItem("No history found")  )
                self.table_history.setSpan(0, 0, 1, 4 - 1 + 1)
                
            else :  
                
                for folder in result_folder_path:
                    total_detact=-1
                    total_no_illeger=-1
                    folder_name = os.path.basename(folder)
                    with open(f'{folder}/result.csv', 'r', newline='') as csvfile:
                        reader = csv.reader(csvfile)
                        for row in reader:
                            total_detact += 1
                            if row[5] == "No error found.":
                                total_no_illeger += 1
                                
                    total_illeger = total_detact - total_no_illeger -1
                    
                    row_count = self.table_history.rowCount()   
                    self.table_history.insertRow(row_count)
                    self.table_history.setItem(row_count, 0, QTableWidgetItem(folder_name))
                    self.table_history.setItem(row_count, 1, QTableWidgetItem(str(total_detact)))
                    self.table_history.setItem(row_count, 2, QTableWidgetItem(str(total_illeger)))
                    
                    if total_illeger != 0:
                        #set button
                        action_btn = QPushButton("More")
                        action_btn.clicked.connect(partial(self.go_to_history_details, folder))
                        icon = QIcon("utils\img\mi.png")  # Replace with any icon name from the list
                        action_btn.setIcon(icon)
                        action_btn.setIconSize(icon.actualSize(action_btn.sizeHint()))
                        action_btn.setStyleSheet("""                                      
                                                QPushButton:hover {
                                                background-color: lightgray;
                                                border-color: black;
                                                font-weight: bold;
                                                }
                                                """)
        
                        self.table_history.setCellWidget(row_count, 3, action_btn)
        else:
            self.table_history.setRowCount(1)     
            self.table_history.setItem(0, 0, QTableWidgetItem("Invalid Path")  )
            self.table_history.setSpan(0, 0, 1, 4 - 1 + 1)
   
        self.table_history.resizeColumnsToContents()
        self.stacked_widget.setCurrentIndex(3)
     
    def go_to_history_details(self,folder_path):
        
        self.history_details_table.clearContents()  # Clear the cell contents
        self.history_details_table.setRowCount(0)
 
        total_detact=-1
        total_no_illeger=-1
        with open(f'{folder_path}/result.csv', 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for col in reader:
                if col[0] != "No":
                    row_count = self.history_details_table.rowCount()
                    self.history_details_table.insertRow(row_count)
                    
                    img_path = f'{folder_path}/crop/{col[1]}.jpg'
                    
                    #insert image
                    item = QTableWidgetItem()
                    pixmap = QPixmap(img_path).scaled(100, 100)  # Resize the image
                    icon = QIcon(pixmap)
                    item.setIcon(icon)

                    self.history_details_table.setIconSize(pixmap.size())

                    # Set a fixed size hint for the item to ensure it is displayed properly
                    item.setSizeHint(pixmap.size())
                        
                    self.history_details_table.setItem(row_count, 0, item)
                        
                    # Optionally set row height and column width to ensure the image fits
                    self.history_details_table.setRowHeight(row_count, 110)
                    self.history_details_table.setColumnWidth(0, 110)
                    self.history_details_table.setItem(row_count, 1, QTableWidgetItem(col[1]))
                    self.history_details_table.setItem(row_count, 2, QTableWidgetItem(col[2]))
                    self.history_details_table.setItem(row_count, 3, QTableWidgetItem(col[3]))
                    self.history_details_table.setItem(row_count, 4, QTableWidgetItem(col[4]))
                    self.history_details_table.setItem(row_count, 5, QTableWidgetItem(col[5]))
                    
                    total_detact += 1

                    if col[5] == "No error found.":
                        total_no_illeger += 1
                        
        total_illeger = total_detact - total_no_illeger -1
        
        row_count = self.history_details_table.rowCount()
        self.history_details_table.insertRow(row_count)  
        
        self.history_details_table.setItem(row_count, 0, QTableWidgetItem("Total No Illegel Vehicle"))
        self.history_details_table.setItem(row_count, 5, QTableWidgetItem(str(total_no_illeger)))
        self.history_details_table.setSpan(row_count, 0, 1, 5)
        
        row_count = self.history_details_table.rowCount()
        self.history_details_table.insertRow(row_count) 
        self.history_details_table.setItem(row_count, 0, QTableWidgetItem("Total Illegel Vehicle"))
        self.history_details_table.setItem(row_count, 5, QTableWidgetItem(str(total_illeger)))
        self.history_details_table.setSpan(row_count, 0, 1, 5)
        
        row_count = self.history_details_table.rowCount()
        self.history_details_table.insertRow(row_count) 
        self.history_details_table.setItem(row_count, 0, QTableWidgetItem("Total Vehicle"))
        self.history_details_table.setItem(row_count, 5, QTableWidgetItem(str(total_detact)))
        self.history_details_table.setSpan(row_count, 0, 1, 5)
        
        self.history_details_table.resizeColumnsToContents()
        self.stacked_widget.setCurrentIndex(4)
                                
    def resizeEvent(self, event):
        # Override resizeEvent to adjust layout and image size based on window size
        super().resizeEvent(event)

        # Calculate available width for the image label
        layout_margins = self.contentsMargins()
        available_width = self.width() - layout_margins.left() - layout_margins.right()
        vailable_height = self.height() - layout_margins.top() - layout_margins.bottom()


        #set for the result_page image
        self.label_img.setFixedSize(int(available_width*0.5),int(vailable_height*0.4))

        # Ensure image scales to fit the label
        self.label_img.setScaledContents(True)
    
    def live_camera(self):
        print("open camera")
        self.runing_text.setText(f"Loading")
        self.text_container.setStyleSheet("background-color:red;")
        QApplication.processEvents() 
        
        self.next_page()
        
        self.run_detaction = Detaction(self,"")
        self.worker_thread = QThread()

        self.run_detaction.moveToThread(self.worker_thread)
        
        self.worker_thread.started.connect(self.run_detaction.live_detaction)                                              
        self.worker_thread.start()
        
        self.run_detaction.warnning.connect(self.warnning_popout)
        self.run_detaction.finish.connect(self.detact_finish)  
                   
    def input_video_img(self):
        
        self.runing_text.setText(f"Loading")
        self.text_container.setStyleSheet("background-color:red;")
        QApplication.processEvents() 
        
        print("\nvideo/camera")
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Video/Image (*.png *.jpg *.jpeg *.bmp *.gif *.mp4 *.avi *.mov)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            if file_path:
                lower_file_path = file_path.lower()
                # Check if selected file is an image or a video
                if any(lower_file_path.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                    print("Image file path",file_path)
                    self.next_page()
                    
                    self.run_detaction = Detaction(self,file_path)
                    self.worker_thread = QThread()
                    
                    self.run_detaction.moveToThread(self.worker_thread)
                    self.worker_thread.started.connect(self.run_detaction.image_detaction) 
                    self.run_detaction.finish.connect(self.detact_finish)     
                    self.worker_thread.start()
                    
                elif any(lower_file_path.endswith(ext) for ext in (".mp4", ".avi", ".mov")):
                    print(file_path)
                    self.next_page()
                    
                    self.run_detaction = Detaction(self,file_path)
                    self.worker_thread = QThread()

                    self.run_detaction.moveToThread(self.worker_thread)
                    self.worker_thread.started.connect(self.run_detaction.video_detaction)  
                    self.run_detaction.finish.connect(self.detact_finish)  
                    self.worker_thread.start()
                else:
                    print("Unsupported file format.")
                    self.warnning_popout("Unsupported file format.")

    def stop_task(self):
        self.run_detaction.stop()  # Signal the worker to stop
        self.result_home_btn.setEnabled(True)
        self.stop_running_btn.setEnabled(False)
    
    @Slot(str)
    def warnning_popout(self,text):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Warning")
        dlg.setText(text)
        dlg.setStandardButtons(QMessageBox.Ok)
        dlg.setIcon(QMessageBox.Warning)
        button = dlg.exec()
        
        if button == QMessageBox.Ok:
            self.back_to_home()

        if self.worker_thread.isRunning():                   
            self.close_thread()            
                    
    @Slot(str)
    def detact_finish(self,folder_name):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(f"The detaction is complete. The result is save to {basedir}/{folder_name}")
        msg_box.setWindowTitle("Infomation")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec() 
        
        self.close_thread()
        
    def close_thread(self):
        self.run_detaction.stop()
        self.worker_thread.quit()
        self.worker_thread.wait()   

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()  # Ensure output is written immediately

    def flush(self):
        for file in self.files:
            file.flush()
                                      
if __name__ == "__main__":
    
    #setup_env.check_and_install_packages()
    
    # Redirect stdout and stderr to a file
    log_file = open('Log File.log', 'w')
    
    # Duplicate stdout and stderr to the console and the log file
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = sys.stderr = Tee(sys.stdout, log_file)
    
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(f'utils/img/icon.png'))
    window = MainWindow()
    window.setMinimumSize(QSize(1000, 600)) 
    window.show()
    
    # Execute the application
    exit_code = app.exec()

    log_file.close()
    sys.exit(exit_code)