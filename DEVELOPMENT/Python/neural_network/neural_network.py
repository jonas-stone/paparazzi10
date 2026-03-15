"""
    =================================================================================================================
    |<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< FIRST ATTEMPT AT CREATING A NEURAL NETWORK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>|
    =================================================================================================================

Notes:
    - This is currently trained on like 50 photos so it is kind of inaccurate and is just "memorizing" photos
    - More data is needed
    - The settings can be played around with to tune the neural network
    - Do not even dare to remove the beautiful comments (they took a while to make) ;)
    - READ THE NOTES!!!!!!!!!
    - Python is better than MATLAB
    - COMMENTS STILL NEED FINISHING
    - Helper function at the end to visualize stuff was heavily made with Chat
    - There is a vbery easy way to get the Neural Network into C (apparently)

Note on the Usage of AI:
    The LLM Chat-Bot Claude has been used to create a separate PyTorch tutorial to assist with the learning of the
python library. The aforementioned tutorial is completely separate and is for a much simpler problem. Additionally, this
specific script that is currently being read, CLaude has been used to assist with debugging.


Date Last Updated:
13/03/2026
                                                                                                                     """
# IMPORTS ##############################################################################################################
# Torch
import torch
import torch.nn as nn
import torch.optim as optim

# Numpy
import numpy as np

# PIL
from PIL import Image

# JSON
import json

# INPUTS ###############################################################################################################
# JSON_Path = r"C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_Team_10\TEAM-10-PROTOTYPING\Python\training_software\labeled_data_6_march_first_100_test_alex.json"
JSON_Path = r"C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_Team_10\TEAM-10-PROTOTYPING\Python\training_software\labeled_data_6_march_494_pictures.json"
Image_Width = 104 # pixels
Image_Height = 48 # pixelsrkill cause
Partitions = 7 # might be slighly an ovewe can get aaway with 5 but seemed more fun
Neurons_Number = 32 # to be played around with
Epochs = 500 # to be played around with
Learning_Rate = 0.001 # step size thing like in fmincon v


input_size = Image_Width * Image_Height

# PIXEL COLUMN VECTOR -> PORTIONS VECTOR ###############################################################################

def get_zones_vector(vector, partitions):
    """
    take the entire vector and number of partitions and turn the vector into one with n number elements to say nth
    partition has a dangerous object

    Note: currently assuming that if over half the partition has a dangerous object that it counts as having a dangerous
    objects, otherwise not

    """
    vector = np.array(vector)
    partition_size = len(vector) / partitions
    label_partition = []

    for partition in range(partitions):
        start_element = int(partition * partition_size)
        end_element = int((partition + 1) * partition_size)
        partition_elements = vector[start_element:end_element]
        label_partition.append(1 if np.mean(partition_elements == 1) > 0.5 else 0)
    return np.array(label_partition, dtype=np.float32)

# JSON-ing AROUND AND FLATTENING PICTURES ##############################################################################

with open(JSON_Path, 'r') as f:
    json_data = json.load(f)

images = []
labels = []

number_to_normalise_by = 255 # apparently because ach pixel is stored as a bit and each bit can hold 11111111
# combinations in binary: i.e. 255 in normal number. Info found online, and it was also mentioned that the binary
# version of 255 is actually white and 0 is black. We need to normalize for the same reason as MDO cause the neural
# network uses a kind of gradient based method to tune itself

for filename, entry in json_data.items():
    image_path = entry['path_b']
    vector = entry['vector']
    image = Image.open(image_path).convert('L') # converts the image to black and white cause its RGB
    image = image.resize((Image_Width, Image_Height), Image.NEAREST) # resizes the image the wrong size just in case
    image_flattened = (np.array(image, dtype=np.float32).flatten()) / number_to_normalise_by # matrices need to be vectors
    images.append(image_flattened)
    labels.append(get_zones_vector(vector, Partitions))

all_labels = np.array(labels)

# SETTING UP THE MODEL AND TRAINING ####################################################################################

# Convert to tensors
X = torch.tensor(np.array(images))
y = torch.tensor(np.array(labels))

# According to claude what this does is: Take the flattened image, multiply by some learned weights, kill negatives,
#  multiply by more learned weights, and sigmoid between 0 and 1 (sigmoid(x) = 1 / (1 + e^(-x)))
model = nn.Sequential(nn.Linear(input_size, Neurons_Number), nn.ReLU(),
                      nn.Linear(Neurons_Number, Partitions), nn.Sigmoid())

criterion = nn.BCELoss() # Binary Cross Entropy Loss measures how wrong the network's prediction is compared to label
"""
loss = -( y * log(p) + (1-y) * log(1-p) )

y = true label (0 or 1)
p = predicted probability (0.0 to 1.0)

0.0  is a  perfect predictions
high number is a terrible predictions
"""

optimizer = optim.Adam(model.parameters(), Learning_Rate) # adam is the optimizer kinda like fmincon in MATLAB

# Looping over the "epochs" (ie training cycles) to improve guess. Kind of like in fmincon where you can set the
# maximum number of evaluations
for epoch in range(Epochs):
    model.train() # you train the model
    output = model(X) # you extract the output
    loss = criterion(output, y) # calculate the loss

    optimizer.zero_grad() # resets the gradients cause apparently by default they just get accumulated
    loss.backward() # allows algorithm to figure out which weight cause the error
    optimizer.step() # this is the weight nudger

    if (epoch + 1) % 100 == 0: # just to print the epoch number and loss very 100 epochs, learnt about this cool
        # % operator thing which give you the remainder
        print(f"Epoch {epoch + 1}/{Epochs}  Loss :  {loss.item():.4f}")

# TESTING ON THE ACTUAL IMAGES #########################################################################################

model.eval() # using the model in evalution mode
with torch.no_grad(): # no more gradient tracking cause it slows stuff down
    for i in range(1):
        output = model(X[i].unsqueeze(0))[0] # adds a fake batch dimension, run through the network and removes the fake dimension
        predicted_label = (output > 0.5).numpy().astype(int) # makes anython lower than 0.5 0 and visa versa for 1
        correct_label = y[i].numpy().astype(int)

        print(f"\nImage {i + 1}:")
        print(f"  True:             {correct_label}")
        print(f"  Predicted:  {predicted_label}")

        safest_direction = int(torch.argmax(output).item())
        direction_taken = safest_direction - 3  # 0->-3, 1->-2, 2->-1, 3->0, 4->+1, 5->+2, 6->+3
        print(f"  Direction: {direction_taken:+d}")  # +/- sign always shown

# HELPER FUNCTION TO SHOW IT WORKS #####################################################################################

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualise(num_images=5):
    # PLEASE NOTE THIS FUNCTION WAS MADE WITH A LOT OF HELP FROM AI (CLAUDE) BUT IS JUST USED TO VISUALISE STUFF
    entries = list(json_data.items())[:num_images]

    fig, axes = plt.subplots(num_images, 2, figsize=(12, num_images * 3))
    #fig.suptitle("Top: color image with danger zones | Bottom: B&W mask", fontsize=13)


    for row, (filename, entry) in enumerate(entries):
        # --- load images ---
        color_img = Image.open(entry['path_a']).convert('RGB')
        bw_img = Image.open(entry['path_b']).convert('L')

        color_w, color_h = color_img.size

        # --- get prediction for this image ---
        idx = list(json_data.keys()).index(filename)
        output = model(X[idx].unsqueeze(0))[0].detach()
        true_z = y[idx].numpy().astype(int)

        # --- top: color image with red shading over dangerous zones ---
        ax_top = axes[row, 0]
        ax_top.imshow(color_img)
        ax_top.set_title(true_z, fontsize=8)
        ax_top.axis('off')

        zone_width = color_w / Partitions
        for z in range(Partitions):
            if true_z[z] == 1:
                rect = patches.Rectangle(
                    (z * zone_width, 0),  # (x, y) bottom-left
                    zone_width, color_h,  # width, height
                    linewidth=0,
                    edgecolor='none',
                    facecolor='red',
                    alpha=0.35
                )
                ax_top.add_patch(rect)

        # draw zone dividers
        for z in range(1, Partitions):
            ax_top.axvline(x=z * zone_width, color='white', linewidth=0.8, alpha=0.6)

        # --- bottom: B&W image with predicted zones ---
        ax_bot = axes[row, 1]
        ax_bot.imshow(bw_img, cmap='gray')
        pred_z = (output > 0.5).numpy().astype(int)
        #ax_bot.set_title(f"Predicted: {pred_z}  ->  Fly: {ZONE_NAMES[int(torch.argmin(output).item())]}", fontsize=8)
        ax_bot.axis('off')

        bw_w, bw_h = bw_img.size
        zone_width_bw = bw_w / Partitions
        for z in range(Partitions):
            if pred_z[z] == 1:
                rect = patches.Rectangle(
                    (z * zone_width_bw, 0),
                    zone_width_bw, bw_h,
                    linewidth=0,
                    edgecolor='none',
                    facecolor='red',
                    alpha=0.35
                )
                ax_bot.add_patch(rect)

        for z in range(1, Partitions):
            ax_bot.axvline(x=z * zone_width_bw, color='white', linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    plt.savefig('visualisation.png', dpi=120, bbox_inches='tight')
    print("\nSaved visualisation.png")
    plt.show()

visualise(num_images=5)