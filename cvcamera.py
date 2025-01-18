from skimage import color
from skimage.util import img_as_ubyte
from skimage.util import img_as_float
from skimage.filters import threshold_otsu, gaussian, median, prewitt
from skimage.morphology import closing, disk, remove_small_holes
from skimage.measure import regionprops, label
from skimage.transform import rescale
import numpy as np
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
# import time
import cv2
import sys
import os
from collections import deque

def define_model():
    
    if getattr(sys, 'frozen', False):
        # Se l'applicazione è stata congelata con PyInstaller,
        # i file aggiuntivi verranno estratti in _MEIPASS
        model_path = os.path.join(sys._MEIPASS, 'digit_recognition_model.pth')
    else:
        model_path = "digit_recognition_model.pth"

    class Network(nn.Module):

        def __init__(self):
            super(Network, self).__init__()
            # Convolutional Neural Network Layer 
            self.convolutaional_neural_network_layers = nn.Sequential(
                    # Here we are defining our 2D convolutional layers
                    # We can calculate the output size of each convolutional layer using the following formular
                    # outputOfEachConvLayer = [(in_channel + 2*padding - kernel_size) / stride] + 1
                    # We have in_channels=1 because our input is a grayscale image
                    nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, padding=1, stride=1), # (N, 1, 28, 28) 
                    nn.ReLU(),
                    # After the first convolutional layer the output of this layer is:
                    # [(28 + 2*1 - 3)/1] + 1 = 28. 
                    nn.MaxPool2d(kernel_size=2), 
                    # Since we applied maxpooling with kernel_size=2 we have to divide by 2, so we get
                    # 28 / 2 = 14
            
                    # output of our second conv layer
                    nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1, stride=1),
                    nn.ReLU(),
                    # After the second convolutional layer the output of this layer is:
                    # [(14 + 2*1 - 3)/1] + 1 = 14. 
                    nn.MaxPool2d(kernel_size=2) 
                    # Since we applied maxpooling with kernel_size=2 we have to divide by 2, so we get
                    # 14 / 2 = 7
            )

            # Linear layer
            self.linear_layers = nn.Sequential(
                    # We have the output_channel=24 of our second conv layer, and 7*7 is derived by the formular 
                    # which is the output of each convolutional layer
                    nn.Linear(in_features=24*7*7, out_features=64),          
                    nn.ReLU(),
                    nn.Dropout(p=0.2), # Dropout with probability of 0.2 to avoid overfitting
                    nn.Linear(in_features=64, out_features=10) # The output is 10 which should match the size of our class
            )

        # Defining the forward pass 
        def forward(self, x):
            x = self.convolutaional_neural_network_layers(x)
            # After we get the output of our convolutional layer we must flatten it or rearrange the output into a vector
            x = x.view(x.size(0), -1)
            # Then pass it through the linear layer
            x = self.linear_layers(x)
            # The softmax function returns the prob likelihood of getting the input image. 
            # We will see a much graphical demonstration below
            x = F.log_softmax(x, dim=1)
            return x

    # Create model instance
    model = Network()

    # Load state dictionary
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    
    return model


def predict_image(image, model):
    # Convert numpy array to PIL image
    image_pil = Image.fromarray((image * 255).astype(np.uint8))

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  
        transforms.ToTensor(),  
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Apply transformations
    input_image = transform(image_pil).unsqueeze(0)
    
    with torch.no_grad():
        log_probs = model(input_image)  # Log-probabilities
        probs = torch.exp(log_probs)   # Converti in probabilità lineari
        probabilities = probs[0] * 100  # Trasforma in percentuale
    #    predicted_digit = torch.argmax(probs, dim=1).item()  # Predizione
                
    # with torch.no_grad():
    #     logits = model.forward(input_image)

    #     # We take the softmax for probabilities since our outputs are logits
    #     probabilities = F.softmax(logits, dim=1).detach().cpu().numpy().squeeze()
    #     predicted_digit = np.argmax(probabilities)
        
    return probabilities

def show_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name, img)
    
    
def histogram_stretch(img_in):
    """
    Stretches the histogram of an image 
    :param img_in: Input image
    :return: Image, where the histogram is stretched so the min values is 0 and the maximum value 255
    """
    img_float = img_as_float(img_in)
    min_val = img_float.min()
    max_val = img_float.max()
    min_desired = 0.0
    max_desired = 1.0
	
    img_out = ((img_float-min_val)*(max_desired-min_desired)/(max_val-min_val))+min_desired
    return img_as_ubyte(img_out)
    
def process_image(image):
    
    img_grey = color.rgb2gray(image)
    
    img_stretched = histogram_stretch(img_grey)
    img_filtered = median(img_stretched, np.ones((5,5)))
    img_outline = prewitt(img_filtered)
    img_closed = closing(img_outline, disk(5))
    threshold = threshold_otsu(img_closed)
    img_tresh = img_closed > threshold
    img_filled = remove_small_holes(img_tresh, connectivity=2)
        
    return img_filled

def center_image(image):
    
    shape = image.shape
    label_image = label(image)
    props = regionprops(label_image)
    
    props_sorted = sorted(props, key=lambda x: x.area, reverse=True)
    
    main_blob = props_sorted[0]
    
    minr, minc, maxr, maxc = main_blob.bbox
    cropped_blob = image[minr:maxr, minc:maxc]
    
    centered_image = np.zeros(shape, dtype=np.bool_)
    
    center_r, center_c = shape[0] // 2, shape[1] // 2
    blob_r, blob_c = cropped_blob.shape
    start_r = center_r - blob_r // 2
    start_c = center_c - blob_c // 2
    
    centered_image[start_r:start_r + blob_r, start_c:start_c + blob_c] = cropped_blob
    
    scale_factor = (2*shape[0]/3) / (maxr - minr)
    centered_image_rescaled = rescale(centered_image, scale_factor, anti_aliasing=False)
    
    new_shape = centered_image_rescaled.shape
    
    if new_shape[0] < shape[0]:
        pad = shape[0] - new_shape[0]
        pad_top = pad // 2
        pad_bottom = pad - pad_top
        centered_image = np.pad(centered_image_rescaled, ((pad_top, pad_bottom), (0, 0)), mode='constant')
        
    if new_shape[0] > shape[0]:
        crop = new_shape[0] - shape[0]
        crop_top = crop // 2
        crop_bottom = crop - crop_top
        centered_image = centered_image_rescaled[crop_top:(new_shape[0] - crop_bottom), :]
        
    if new_shape[1] < shape[1]:
        pad = shape[1] - new_shape[1]
        pad_left = pad // 2
        pad_right = pad - pad_left
        centered_image = np.pad(centered_image, ((0, 0), (pad_left, pad_right)), mode='constant')
        
    if new_shape[1] > shape[1]:
        crop = new_shape[1] - shape[1]
        crop_left = crop // 2
        crop_right = crop - crop_left
        centered_image = centered_image[:, crop_left:(new_shape[1] - crop_right)]
    
    return centered_image
    

def capture_from_camera():
    
    BUFFER_SIZE = 30
    
    model = define_model()
    
    print("Starting image capture")

    print("Opening connection to camera")
    url = 0
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
        
    print("Starting camera loop")
    # To keep track of frames per second using a high-performance counter
    #old_time = time.perf_counter()
    #fps = 0
    
    buffer = deque(maxlen=BUFFER_SIZE)
    smoothed_probs = np.zeros(10)  # Per 10 classi (MNIST)
    frame_counter = 0
    str_out_digit = ""
    
    stop = False
    while not stop:
        ret, new_frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

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
                
        probabilities = predict_image(centered_image, model)
        
        buffer.append(probabilities.numpy())
        frame_counter += 1
        
        # Calcola la media delle probabilità per ogni classe
        prob_array = np.array(buffer)  # Array 2D: [num_frames, num_classes]
        mean_probs = np.mean(prob_array, axis=0)  # Media sulle righe (frame)
        
        smoothed_probs = 0.5 * smoothed_probs + 0.5 * mean_probs

        # Trova la classe con la probabilità media più alta
        best_digit = np.argmax(smoothed_probs)
        best_prob = smoothed_probs[best_digit]
        
        if(frame_counter == 30):
            print(f"Best digit: {best_digit} with mean probability: {best_prob:.2f}%")
            # Generate updated output string
            if(best_prob > 50):
                str_out_digit = f"Digit: {best_digit} ({best_prob:.0f}%)"
            else:
                str_out_digit = ""
            frame_counter = 0
            
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(new_frame, str_out_digit, (top_left[0] - 18, top_left[1] - 15), font, 1, (255, 0, 0), 2)
        cv2.putText(new_frame, "Giacomo Stefanizzi", ((width - int(width/4)), (int(height/20))), font, 1, (0, 0, 255), 1)
        cv2.putText(new_frame, "Press 'q' to quit", ((width - int(width/4)), (int(height/12))), font, 1, (0, 0, 255), 1)
        cv2.rectangle(new_frame, top_left, bottom_right, color, thickness)
        
        # Debug
        #print(f"Predicted digit: {prediction}, Probability: {probabilities[prediction]}")
        
        # Display the resulting frame
        show_window('Input', new_frame, 0, 0)
        show_window('Processed', img_as_ubyte(centered_image), 0, 500)


        if cv2.getWindowImageRect('Input')[2] == 0 or cv2.getWindowImageRect('Processed')[2] == 0:
            stop = True

        if cv2.waitKey(1) == ord('q'):
            #cv2.imwrite("debug_frame.png", centered_image)
            stop = True
            

    print("Stopping image loop")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_from_camera()