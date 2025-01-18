import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage
import cv2
from cvcamera import *

class CameraGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CVChallenge GUI")
        self.video_label = QLabel("Camera feed will appear here")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.video_label)
        left_layout.addWidget(self.start_button)
        #left_layout.addWidget(self.stop_button)
        

        self.name_label = QLabel("Giacomo Stefanizzi")
        self.info_crop_label = QLabel("Cropped/Processed image")
        self.crop_label = QLabel("")
        self.digit_label = QLabel("")

        
        # Adjust spacing between widgets
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.name_label, 0, Qt.AlignTop | Qt.AlignHCenter)
        #right_layout.addWidget(self.info_crop_label, 0, Qt.AlignTop)
        right_layout.addWidget(self.crop_label, 0, Qt.AlignTop | Qt.AlignHCenter)
        right_layout.addWidget(self.digit_label, 0, Qt.AlignTop | Qt.AlignHCenter)
        
        # ---------------------
        #  LAYOUT ORIZZONTALE
        # ---------------------
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # Contenitore centrale
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
        # Open the default camera (same idea as cvcamera.py)
        self.cap = cv2.VideoCapture(0)
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        #self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.timer.start(0)
        self.start_button.hide()

    def update_frame(self):
        
        if self.cap is not None and self.cap.isOpened():
            ret, new_frame = self.cap.read()
            if ret:
                
                # Draw a centered rectangle on the new_frame
                crop_size = 192
                height, width, _ = new_frame.shape
                rect_width, rect_height = crop_size, crop_size
                top_left = ((width - rect_width) // 2, (height - rect_height) // 2) 
                bottom_right = (top_left[0] + rect_width, top_left[1] + rect_height) 
                color = (0, 255, 0)  # Green color in BGR
                thickness = 2
                        
                # Process image
                new_image = new_frame[(top_left[1]):(bottom_right[1]), (top_left[0]):(bottom_right[0]), ::-1]
                proc_image = process_image(new_image)
                centered_image = center_image(proc_image)
                        
                probabilities = predict_image(centered_image, self.model)
                
                self.buffer.append(probabilities.numpy())
                self.frame_counter += 1
                
                # Calcola la media delle probabilità per ogni classe
                prob_array = np.array(self.buffer)  # Array 2D: [num_frames, num_classes]
                mean_probs = np.mean(prob_array, axis=0)  # Media sulle righe (frame)
                
                self.smoothed_probs = 0.5 * self.smoothed_probs + 0.5 * mean_probs

                # Trova la classe con la probabilità media più alta
                best_digit = np.argmax(self.smoothed_probs)
                best_prob = self.smoothed_probs[best_digit]
                
                print(self.frame_counter)
                
                if(self.frame_counter == 30):
                    print(f"Best digit: {best_digit} with mean probability: {best_prob:.2f}%")
                    # Generate updated output string
                    if(best_prob > 60):
                        self.str_out_digit = f"Digit: {best_digit} ({best_prob:.0f}%)"
                    else:
                        self.str_out_digit = ""
                    self.frame_counter = 0
                    
                text_position = (top_left[0] - 50 , top_left[1] - 10)
                cv2.putText(new_frame, "Place the digit here", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                cv2.rectangle(new_frame, top_left, bottom_right, color, thickness)
                
                self.digit_label.setText(self.str_out_digit)
                self.digit_label.setStyleSheet("font-size: 18px;")
                
                # Convert frame to QImage
                rgb_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qimg))
                
                # Convert frame to QImage
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
