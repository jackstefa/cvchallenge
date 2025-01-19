import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage, QIcon
import cv2
from collections import deque
from model import *
from preprocessing import *

class CameraGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CVChallenge GUI")
        self.video_label = QLabel("Camera feed will appear here")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.start_button = QPushButton("Start Camera")
        
        # Pyinstaller
        if getattr(sys, 'frozen', False):
            self.setWindowIcon(QIcon(os.path.join(sys._MEIPASS, 'icon.png')))
        else:
            self.setWindowIcon(QIcon('icon.png'))
    
        self.start_button.clicked.connect(self.start_camera)
        
        self.rules_label = QLabel("Place the digit inside the green box")
        self.rules_label.hide()
        self.name_label_left = QLabel("Giacomo Stefanizzi")

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.name_label_left)
        left_layout.addWidget(self.video_label)
        left_layout.addWidget(self.start_button)
        left_layout.addWidget(self.rules_label, 0, Qt.AlignTop | Qt.AlignHCenter)
        
        self.name_label_right = QLabel("Giacomo Stefanizzi")
        self.name_label_right.hide()
        self.crop_label = QLabel("")
        self.crop_label.hide()
        self.digit_label = QLabel("")

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.name_label_right, 0, Qt.AlignTop | Qt.AlignHCenter)
        right_layout.addWidget(self.rules_label, 0, Qt.AlignTop | Qt.AlignHCenter)
        self.crop_label.setFixedSize(150, 150)
        self.crop_label.setScaledContents(True)
        right_layout.addWidget(self.crop_label, 0, Qt.AlignTop | Qt.AlignHCenter)
        right_layout.addWidget(self.digit_label, 0, Qt.AlignTop | Qt.AlignHCenter)
        
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        
        self.model = define_model()
        
        self.frame_counter = 0
        self.smoothed_probs = np.zeros(10)  # Per 10 classi (MNIST)
        self.str_out_digit = ""
        
        self.BUFFER_SIZE = 30
        self.buffer = deque(maxlen=self.BUFFER_SIZE)

    def start_camera(self):
        # Open the default camera
        self.cap = cv2.VideoCapture(0)
        self.timer.start(0)
        self.start_button.hide()
        self.rules_label.show()
        self.name_label_right.show()
        self.name_label_left.hide()
        self.crop_label.show()

    def update_frame(self):
        
        if self.cap is not None and self.cap.isOpened():
            ret, new_frame = self.cap.read()
            if ret:
                
                # FRAME CROP
                height, width, _ = new_frame.shape
                rect_width, rect_height = 96, 96
                top_left = ((width - rect_width) // 2, (height - rect_height) // 2) 
                bottom_right = (top_left[0] + rect_width, top_left[1] + rect_height) 
                new_image = new_frame[(top_left[1]):(bottom_right[1]), (top_left[0]):(bottom_right[0]), ::-1]
                
                # PREPROCESSING
                proc_image = process_image(new_image)
                centered_image = center_image(proc_image)
                        
                # PREDICTION
                probabilities = predict_image(centered_image, self.model)
                
                self.buffer.append(probabilities.numpy())
                self.frame_counter += 1
                
                # Compute mean of the buffer
                prob_array = np.array(self.buffer)
                mean_probs = np.mean(prob_array, axis=0)
                
                # Smoothing to reduce flickering
                alpha = 0.9
                self.smoothed_probs = alpha * self.smoothed_probs + (1-alpha) * mean_probs

                # Find the best digit
                best_digit = np.argmax(self.smoothed_probs)
                best_prob = self.smoothed_probs[best_digit]
                                
                # Show the best digit
                if(self.frame_counter == 30):
                    # Show only if the probability is greater than 60%
                    if(best_prob > 60):
                        self.str_out_digit = f"Digit: {best_digit} ({best_prob:.0f}%)"
                    else:
                        self.str_out_digit = ""
                    self.frame_counter = 0
                   
                # UI
                # Rectangle
                cv2.rectangle(new_frame, top_left, bottom_right, (0, 255, 0), 2)
                
                # Digit predicted and probability
                self.digit_label.setText(self.str_out_digit)
                self.digit_label.setStyleSheet("font-size: 18px;")
                
                # Webcam feed
                rgb_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qimg))
                
                # Preprocessed image
                rgb_frame = cv2.cvtColor(img_as_ubyte(centered_image), cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.crop_label.setPixmap(QPixmap.fromImage(qimg))

    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
        self.timer.stop()
        self.video_label.clear()

def main():
    app = QApplication(sys.argv)
    window = CameraGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
