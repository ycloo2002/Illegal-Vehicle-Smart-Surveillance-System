import sys
import threading
from random import choice
from detact import Detaction
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPalette, QColor,QFont
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QColorDialog,
    QGroupBox,
    QStackedWidget,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QHeaderView
)
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtCore import Qt, QTimer, QThread, Signal, Slot,QThreadPool


FF = 'Arial'

    
def create_btn(name,poit_to):
        #start btn
    button = QPushButton(name)
    button.clicked.connect(poit_to)
    button.setFont(QFont(FF, 12))
    button.setFixedSize(150, 50)
        
    button_style = """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: 2px solid #4CAF50;
                border-radius: 20px;
            }
            QPushButton:hover {
                background-color: #45a049;
                border-color: #45a049;
                font-weight: bold;
            }
        """
        
    button.setStyleSheet(button_style)
        
    button_box = QVBoxLayout()
    button_box.addWidget(button)
    button_box.setAlignment(Qt.AlignHCenter)
        
    return button_box
     
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Home")
        
        # Create stacked widget to hold pages
        self.stacked_widget = QStackedWidget(self)
        
        # Create pages
        self.home = QWidget()
        self.input = QWidget()
        self.result = QWidget()
        
        
        # Add widgets to pages
        self.init_home()
        self.init_input()
        self.init_result()
        
        # Add pages to stacked widget
        self.stacked_widget.addWidget(self.home)
        self.stacked_widget.addWidget(self.input)
        self.stacked_widget.addWidget(self.result)
        
        
        # Create layout for main window
        layout = QVBoxLayout(self)
        
        # Add stacked widget to main layout
        layout.addWidget(self.stacked_widget)

        self.run_detaction = Detaction()
        
    def init_home(self):
        layout = QVBoxLayout(self.home)
        
        #title 
        title = QLabel("Illegal-Vehicle-Smart-Surveillance-System")
        title.setFont(QFont(FF, 20))
        title.setAlignment(Qt.AlignHCenter)
        title.setFixedSize(1000, 100)
        
        title_box = QVBoxLayout()
        title_box.addWidget(title)
        #title_box.setAlignment(Qt.AlignHCenter)
        
        layout.addLayout(title_box)
        
        #start btn
        button = QPushButton("Start")
        button.clicked.connect(self.next_page)
        button.setFont(QFont(FF, 12))
        button.setFixedSize(150, 50)
        
        button_style = """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: 2px solid #4CAF50;
                border-radius: 20px;
            }
            QPushButton:hover {
                background-color: #45a049;
                border-color: #45a049;
                font-weight: bold;
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

        #title 
        title = QLabel("Choose Input Type")
        title.setFont(QFont(FF, 15))
        title.setAlignment(Qt.AlignHCenter)
        title.setFixedSize(1000, 40)
        
        title_box = QVBoxLayout()
        title_box.addWidget(title)
        
        layout.addLayout(title_box)
        
        
        layout.addLayout(create_btn("Camere",self.live_camera))
        layout.addLayout(create_btn("Video/Image",self.input_video_img))
        layout.addLayout(create_btn("back",self.back_to_home))
   
    def init_result(self):
        
        self.setWindowTitle("Result")
        
        layout = QVBoxLayout(self.result)
        
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
        
        self.table_info.setStyleSheet("""
            QTableWidget::item:!alternate {
                background-color: #f2f2f2; /* Light gray for odd rows */
            }
            QTableWidget::item:alternate {
                background-color: white; /* White for even rows */
            }
        """)
        
        t_layout.addWidget(self.table_info)
        
        h_layout.addLayout(t_layout)
        
        self.label_img = QLabel()
        # Load the image and set it to the QLabel
        pixmap = QPixmap("")
        self.label_img.setPixmap(pixmap)
        self.label_img.setFixedSize(pixmap.size())
        self.adjustSize()

        h_layout.addWidget(self.label_img)
        
        #video
        
        layout.addLayout(h_layout)
        
        # section 2
        self.label_table_warnning = QLabel("Warning")
        self.label_table_warnning.setAlignment(Qt.AlignCenter)
        self.label_table_warnning.setStyleSheet("font-size: 20px; font-weight: bold; padding: 5px;")
        layout.addWidget(self.label_table_warnning)
        
        
        self.table_warnning = QTableWidget()
        self.table_warnning.setColumnCount(7)
        self.table_warnning.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_warnning.setHorizontalHeaderLabels(['Vehicle_Image',"License_Plate","Vehicle_Type", "Vehicle_Brand",'Vehicle_Colour','Warning Message','Vehicle_Onwner'])
        self.table_warnning.setAlternatingRowColors(True)
        self.table_warnning.setStyleSheet("""
            QTableWidget::item:!alternate {
                background-color: #f2f2f2; /* Light gray for odd rows */
            }
            QTableWidget::item:alternate {
                background-color: white; /* White for even rows */
            }
        """)
        
        # Set the column sizes
        header = self.table_warnning.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Fixed)
        header.resizeSection(5, 400) 
        
        layout.addWidget(self.table_warnning)
        
        # section 3 
        hh_text = QHBoxLayout()
        
        self.text_container = QWidget()
        self.text_container.setStyleSheet("background-color:red;")
        
        f_layout = QVBoxLayout(self.text_container)
        
        self.runing_text = QLabel("Loading")
        self.runing_text.setFont(QFont(FF, 20))
        self.runing_text.setAlignment(Qt.AlignHCenter)
        self.runing_text.setStyleSheet("font-weight: bold;text-align: center;margin: 10px 2px;color:white")
        
        f_layout.addWidget(self.runing_text)
        
        hh_text.addWidget(self.text_container)
        
        hh_text.addLayout(create_btn("back",self.back_to_home))
        
        layout.addLayout(hh_text)     
        
    def next_page(self):
        current_index = self.stacked_widget.currentIndex()
        next_index = (current_index + 1) % self.stacked_widget.count()
        self.stacked_widget.setCurrentIndex(next_index)  
        
    def back_to_home(self):
        self.stacked_widget.setCurrentIndex(0)
    
    def live_camera(self):
        print("open camera")
        self.runing_text.setText(f"Loading")
        self.text_container.setStyleSheet("background-color:red;")
        QApplication.processEvents() 
        
        self.next_page()
        QTimer.singleShot(200, lambda:self.run_detaction.live_detaction(self))
             
    def input_video_img(self):
        
        self.runing_text.setText(f"Loading")
        self.text_container.setStyleSheet("background-color:red;")
        QApplication.processEvents() 
        
        print("video/camera")
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
                    print(file_path)
                    self.detact_input = file_path
                    self.next_page()
                    QTimer.singleShot(200, lambda:self.run_detaction.image_detaction(self))

                    
                elif any(lower_file_path.endswith(ext) for ext in (".mp4", ".avi", ".mov")):
                    print(file_path)
                    self.detact_input = file_path
                    self.next_page()
                    
                    QTimer.singleShot(200, lambda:self.run_detaction.video_detaction(self))          
                    
                else:
                    print("Unsupported file format.")

    
                
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setMinimumSize(QSize(1000, 600)) 
    window.show()
    
    sys.exit(app.exec())