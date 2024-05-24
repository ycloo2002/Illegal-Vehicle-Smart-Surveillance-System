import sys
from random import choice
from detact import result_window
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
    QTableWidgetItem
)


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
        self.w = None  # No external window yet
        self.setWindowTitle("Home")
        
        # Create stacked widget to hold pages
        self.stacked_widget = QStackedWidget(self)
        
        # Create pages
        self.home = QWidget()
        self.input = QWidget()
        
        # Add widgets to pages
        self.init_home()
        self.init_input()
        
        # Add pages to stacked widget
        self.stacked_widget.addWidget(self.home)
        self.stacked_widget.addWidget(self.input)
        
        # Create layout for main window
        layout = QVBoxLayout(self)
        
        # Add stacked widget to main layout
        layout.addWidget(self.stacked_widget)
        
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
        version = QLabel("V 0.1 beta")
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
        
    def next_page(self):
        current_index = self.stacked_widget.currentIndex()
        next_index = (current_index + 1) % self.stacked_widget.count()
        self.stacked_widget.setCurrentIndex(next_index)     
        
    def back_to_home(self):
        self.stacked_widget.setCurrentIndex(0)
    
    def live_camera(self):
        print("open camera")
        
    def input_video_img(self):
        
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
                    
                elif any(lower_file_path.endswith(ext) for ext in (".mp4", ".avi", ".mov")):
                    print(file_path)
                    if self.w is None:
                        self.w = result_window(file_path)
                        self.w.show()

                    else:
                        self.w.close()  # Close window.
                        self.w = None  # Discard reference.
                    #detact.run(file_path)
                    
                else:
                    print("Unsupported file format.")
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setMinimumSize(QSize(1000, 600)) 
    window.show()
    sys.exit(app.exec())