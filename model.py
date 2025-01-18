import numpy as np
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import sys
import os

def define_model():
    
    # For Pyinstaller
    if getattr(sys, 'frozen', False):
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
        log_probs = model(input_image) 
        probs = torch.exp(log_probs)   
        probabilities = probs[0] * 100  
                
    return probabilities