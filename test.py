import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from model import *
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from skimage.color import rgb2lab, lab2rgb, rgb2gray

# Load test images
test_path = './images/images_test_gray/'
colorize = []
for filename in os.listdir(test_path):
    img = cv2.imread(os.path.join(test_path, filename))
    img = cv2.resize(img, (256, 256))
    img = rgb2lab( img)[:,:,0]
    colorize.append(img)


# Convert colorize to a PyTorch tensor
colorize = torch.tensor(colorize, dtype=torch.float32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AE()
# Load the weights from the OrderedDict
model.load_state_dict(torch.load('colorization_model.pt'))
##USE CPU 
# model.load_state_dict(torch.load('colorization_model.pt', map_location=torch.device('cpu')))

model.to(device)
colorize = colorize.to(device)
output = model(colorize.unsqueeze(1))  # Add a channel dimension to the input tensor
output = output * 110.  # Scale the output back to the range [-128, 127]

output = output.cpu().detach().numpy()

# Display results
fig, ax = plt.subplots(len(output), 2, figsize=(16, 100))
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    colorize[i].shape
    colorize.unsqueeze(1)
    cur[:, :, 0] = colorize[i].cpu().detach().numpy()
    # Resize the output tensor to match the shape of the image
    output_resized = cv2.resize(output[i].transpose(1, 2, 0), (256, 256))
    # Assign the resized output to the last two channels of cur
    cur[:, :, 1:] = output_resized
    resImage = lab2rgb(cur)
    ax[i, 0].imshow(cv2.cvtColor(cv2.imread(os.path.join(test_path, os.listdir(test_path)[i])), cv2.COLOR_BGR2RGB), interpolation='nearest')
    ax[i, 1].imshow(resImage, interpolation='nearest')
plt.show()
