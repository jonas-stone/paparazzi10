"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
====================================================================================================================|
A|B|N --------------------------------------------------------------------------------------------------------------|
A|B|N <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< NEURAL NETWORK >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>|
A|B|N --------------------------------------------------------------------------------------------------------------|
====================================================================================================================|
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# below is the old code which is not very modular
''' OLD CODE LEFT JUST FOR REFFERENCE
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

# OS
import os

# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2
import numpy as np


# INPUTS ###############################################################################################################
# json_path = r"..\training_software\labeled_data_6_march_494_pictures.json"
base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "..", "training_software", "labeled_data_6_march_494_pictures.json")
image_width = 104 # pixels
image_height = 48 # pixelsrkill cause

partitions = 7 # might be slighly an ovewe can get aaway with 5 but seemed more fun
neurons_number = 32 # to be played around with
epochs = 1000 # to be played around with
learning_rate = 0.001 # step size thing like in fmincon v

# image_width = 52
# image_height = 24
input_size = image_width * image_height

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
        label_partition.append(1 if np.mean(partition_elements == 1) > 0.25 else 0)
    return np.array(label_partition, dtype=np.float32)

# JSON-ing AROUND AND FLATTENING PICTURES ##############################################################################

with open(json_path, 'r') as f:
    json_data = json.load(f)

images = []
labels = []

number_to_normalise_by = 255 # apparently because ach pixel is stored as a bit and each bit can hold 11111111
# combinations in binary: i.e. 255 in normal number. Info found online, and it was also mentioned that the binary
# version of 255 is actually white and 0 is black. We need to normalize for the same reason as MDO cause the neural
# network uses a kind of gradient based method to tune itself
counter = 0
for filename, entry in json_data.items():
    # image_path = entry['path_a']
    image_path = os.path.join(base_dir, "..", "..", "training_images", "training_b&w_6_march", entry['filename'])
    vector = entry['vector']
    image = Image.open(image_path).convert('L') # converts the image to black and white cause its RGB
    # image = Image.fromarray(undistort(np.array(image)))    # fresh
    image = image.resize((image_width, image_height), Image.NEAREST) # resizes the image the wrong size just in case
    image_flattened = (np.array(image, dtype=np.float32).flatten()) / number_to_normalise_by # matrices need to be vectors
    images.append(image_flattened)
    labels.append(get_zones_vector(vector, partitions))

    counter = counter + 1
    print(counter)

all_labels = np.array(labels)

# SETTING UP THE MODEL AND TRAINING ####################################################################################

# Convert to tensors
X = torch.tensor(np.array(images))
y = torch.tensor(np.array(labels))

# According to claude what this does is: Take the flattened image, multiply by some learned weights, kill negatives,
#  multiply by more learned weights, and sigmoid between 0 and 1 (sigmoid(x) = 1 / (1 + e^(-x)))
# model = nn.Sequential(nn.Linear(input_size, neurons_number), nn.ReLU(),
#                       nn.Linear(neurons_number, partitions), nn.Sigmoid())

model = nn.Sequential(
    nn.Linear(input_size, 256), nn.ReLU(),
    nn.Linear(256, 64),         nn.ReLU(),
    nn.Linear(64, partitions),  nn.Sigmoid()
)


criterion = nn.BCELoss() # Binary Cross Entropy Loss measures how wrong the network's prediction is compared to label
"""
loss = -( y * log(p) + (1-y) * log(1-p) )

y = true label (0 or 1)
p = predicted probability (0.0 to 1.0)

0.0  is a  perfect predictions
high number is a terrible predictions
"""

optimizer = optim.Adam(model.parameters(), learning_rate) # adam is the optimizer kinda like fmincon in MATLAB

# Looping over the "epochs" (ie training cycles) to improve guess. Kind of like in fmincon where you can set the
# maximum number of evaluations

for epoch in range(epochs):
    model.train() # you train the model
    output = model(X) # you extract the output
    loss = criterion(output, y) # calculate the loss

    optimizer.zero_grad() # resets the gradients cause apparently by default they just get accumulated
    loss.backward() # allows algorithm to figure out which weight cause the error
    optimizer.step() # this is the weight nudger

    if (epoch + 1) % 100 == 0: # just to print the epoch number and loss very 100 epochs, learnt about this cool
        # % operator thing which give you the remainder
        print(f"Epoch {epoch + 1}/{epochs}  Loss :  {loss.item():.4f}")

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

# def visualise(num_images=5, random=False):
#     # PLEASE NOTE THIS FUNCTION WAS MADE WITH A LOT OF HELP FROM AI (CLAUDE) BUT IS JUST USED TO VISUALISE STUFF
#
#     if random:
#         entries = __import__('random').sample(list(json_data.items()), num_images)
#     else:
#         entries = list(json_data.items())[:num_images]
#
#     fig, axes = plt.subplots(num_images, 2, figsize=(12, num_images * 3))
#     #fig.suptitle("Top: color image with danger zones | Bottom: B&W mask", fontsize=13)

def visualise(num_images=5, random=False, folder="6_march"):
    bw_folder = os.path.join(base_dir, "..", "..", "training_images", f"training_b&w_{folder}")
    color_folder = os.path.join(base_dir, "..", "..", "training_images", f"training_color_{folder}")

    if folder == "6_march":
        all_entries = list(json_data.items())
        entries = __import__('random').sample(all_entries, num_images) if random else all_entries[:num_images]
        filenames = [e[0] for e in entries]
        has_labels = True
    else:
        all_filenames = os.listdir(bw_folder)
        filenames = __import__('random').sample(all_filenames, num_images) if random else all_filenames[:num_images]
        has_labels = False

    fig, axes = plt.subplots(num_images, 2, figsize=(12, num_images * 3))
    # # fig, axes = plt.subplots(num_images, 2, figsize=(12, num_images * 4))
    # # if num_images == 1:
    # #     axes = np.array([axes])
    # img_height = max(3, 15 - num_images // 5)  # shrinks row height as num_images grows
    # fig, axes = plt.subplots(num_images, 2, figsize=(12, num_images * img_height))

    for row, filename in enumerate(filenames):
        color_img = Image.open(os.path.join(color_folder, filename)).convert('RGB')
        bw_img = Image.open(os.path.join(bw_folder, filename)).convert('L')
        color_w, color_h = color_img.size
        bw_w, bw_h = bw_img.size

        # run through model
        image_resized = bw_img.resize((image_width, image_height), Image.NEAREST)
        image_tensor = torch.tensor(np.array(image_resized, dtype=np.float32).flatten() / 255).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)[0]
        pred_z = (output > 0.5).numpy().astype(int)

        # top: color image
        ax_top = axes[row, 0]
        ax_top.imshow(color_img)
        ax_top.axis('off')
        zone_width = color_w / partitions

        if has_labels:
            idx = list(json_data.keys()).index(filename)
            true_z = y[idx].numpy().astype(int)
            ax_top.set_title(true_z, fontsize=8)
            for z in range(partitions):
                if true_z[z] == 1:
                    rect = patches.Rectangle((z * zone_width, 0), zone_width, color_h,
                                             linewidth=0, edgecolor='none', facecolor='red', alpha=0.35)
                    ax_top.add_patch(rect)
        # else:
        #     ax_top.set_title("no labels", fontsize=8)
        else:
            ax_top.set_title("no labels", fontsize=8)
            for z in range(partitions):
                if pred_z[z] == 1:
                    rect = patches.Rectangle((z * zone_width, 0), zone_width, color_h,
                                             linewidth=0, edgecolor='none', facecolor='red', alpha=0.35)
                    ax_top.add_patch(rect)

        for z in range(1, partitions):
            ax_top.axvline(x=z * zone_width, color='white', linewidth=0.8, alpha=0.6)

        # bottom: b&w image with predicted zones
        ax_bot = axes[row, 1]
        ax_bot.imshow(bw_img, cmap='gray')
        ax_bot.axis('off')
        zone_width_bw = bw_w / partitions
        for z in range(partitions):
            if pred_z[z] == 1:
                rect = patches.Rectangle((z * zone_width_bw, 0), zone_width_bw, bw_h,
                                         linewidth=0, edgecolor='none', facecolor='red', alpha=0.35)
                ax_bot.add_patch(rect)
        for z in range(1, partitions):
            ax_bot.axvline(x=z * zone_width_bw, color='white', linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    plt.savefig('visualisation.png', dpi=120, bbox_inches='tight')
    print("\nSaved visualisation.png")
    plt.show()

# visualise(num_images=5, random=True)
visualise(num_images=10, random=True, folder="13_march")
visualise(num_images=10, random=True, folder="13_march")
visualise(num_images=10, random=True, folder="13_march")
visualise(num_images=10, random=True, folder="13_march")
visualise(num_images=10, random=True, folder="13_march")
visualise(num_images=10, random=True, folder="13_march")
visualise(num_images=10, random=True, folder="13_march")
visualise(num_images=10, random=True, folder="13_march")
'''
# below is the modularized code that was made with some AI help for debugging: both codes when they were originally made
# did the same thing

########################################################################################################################
# IMPORTS ##############################################################################################################
########################################################################################################################

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from PIL import Image

import json

import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches

########################################################################################################################
# INPUTS ###############################################################################################################
########################################################################################################################

def get_inputs():
    """
    Function to store all inputs.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "..", "training_software", "labeled_data_6_march_494_pictures.json")
    image_width  = 104  # pixels
    image_height = 48   # pixels

    partitions     = 7     # might be slightly an overkill cause we can get away with 5 but seemed more fun
    neurons_number = 32    # to be played around with
    epochs         = 1000  # to be played around with
    learning_rate  = 0.001 # step size thing like in fmincon

    input_size = image_width * image_height

    return base_dir, json_path, image_width, image_height, partitions, neurons_number, epochs, learning_rate, input_size

########################################################################################################################
# FUNCTIONS ############################################################################################################
########################################################################################################################

def get_zones_vector(vector, partitions):
    """
    Take the entire vector and number of partitions and turn the vector into one with n number elements to say nth
    partition has a dangerous object.

    pixel vector:  [0, 1, 1, 0,    0, 0, 0, 0,     0, 0, 1, 0]
    zones:         |-- zone 0--|  |-- zone 1--|  |-- zone 2--|

    """
    threshold_value = 0.5
    vector = np.array(vector)
    partition_size = len(vector) / partitions
    label_partition = []

    for partition in range(partitions):
        start_element = int(partition * partition_size)
        end_element = int((partition + 1) * partition_size)
        partition_elements = vector[start_element:end_element]
        label_partition.append(1 if np.mean(partition_elements == 1) > threshold_value else 0)
    return np.array(label_partition, dtype=np.float32)


def load_dataset_from_json(base_dir, json_path, image_width, image_height, partitions):
    """
    Loading data for the X and Y vector, only works with JSON
    """
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    images = []
    labels = []

    number_to_normalise_by = 255 # apparently because each pixel is stored as a bit and each bit can hold 11111111
    # combinations in binary: i.e. 255 in normal number. Info found online, and it was also mentioned that the binary
    # version of 255 is actually white and 0 is black. We need to normalize for the same reason as MDO cause the neural
    # network uses a kind of gradient based method to tune itself
    counter = 0
    for filename, entry in json_data.items():
        image_path = os.path.join(base_dir, "..", "..", "training_images", "training_b&w_6_march", entry['filename'])
        vector = entry['vector']
        image = Image.open(image_path).convert('L') # converts the image to black and white cause its RGB
        image = image.resize((image_width, image_height), Image.NEAREST) # resizes the image the wrong size just in case
        image_flattened = (np.array(image, dtype=np.float32).flatten()) / number_to_normalise_by # matrices need to be vectors
        images.append(image_flattened)
        labels.append(get_zones_vector(vector, partitions))

        counter = counter + 1
        print(counter)

    # Convert to tensors
    X = torch.tensor(np.array(images))
    y = torch.tensor(np.array(labels))

    return X, y, json_data


# SETTING UP THE MODEL #################################################################################################
def build_model(input_size, partitions, overkill=False):
    """
    According to claude what this does is: Take the flattened image, multiply by some learned weights, kill negatives,
    multiply by more learned weights, and sigmoid between 0 and 1 (sigmoid(x) = 1 / (1 + e^(-x))) (true for the not
    overkill version)
    """
    if overkill:
        model = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(),
            nn.Linear(256, 64),         nn.ReLU(),
            nn.Linear(64, partitions),  nn.Sigmoid()
        )
    else:
        model = nn.Sequential(
            nn.Linear(input_size, 32), nn.ReLU(),
            nn.Linear(32, partitions), nn.Sigmoid()
        )
    return model


def train_model(model, X, y, epochs, learning_rate):
    """
    Training the model repeatedly
    """
    criterion = nn.BCELoss() # Binary Cross Entropy Loss measures how wrong the network's prediction is compared to label
    """
    loss = -( y * log(p) + (1-y) * log(1-p) )

    y = true label (0 or 1)
    p = predicted probability (0.0 to 1.0)

    0.0  is a  perfect predictions
    high number is a terrible predictions
    """

    optimizer = optim.Adam(model.parameters(), learning_rate) # adam is the optimizer kinda like fmincon in MATLAB

    # Looping over the "epochs" (ie training cycles) to improve guess. Kind of like in fmincon where you can set the
    # maximum number of evaluations
    for epoch in range(epochs):
        model.train() # you train the model
        output = model(X) # you extract the output
        loss = criterion(output, y) # calculate the loss

        optimizer.zero_grad() # resets the gradients cause apparently by default they just get accumulated
        loss.backward() # allows algorithm to figure out which weight cause the error
        optimizer.step() # this is the weight nudger

        if (epoch + 1) % 100 == 0: # just to print the epoch number and loss every 100 epochs, learnt about this cool
            # % operator thing which gives you the remainder
            print(f"Epoch {epoch + 1}/{epochs}  Loss :  {loss.item():.4f}")

    return model


def evaluate_model(model, X, y):
    """
    Function which just shows that the model predicts correctly its own data
    """
    model.eval() # using the model in evaluation mode
    with torch.no_grad(): # no more gradient tracking cause it slows stuff down
        for i in range(1):
            output = model(X[i].unsqueeze(0))[0] # adds a fake batch dimension, run through the network and removes the fake dimension
            predicted_label = (output > 0.5).numpy().astype(int) # makes anything lower than 0.5 0 and visa versa for 1
            correct_label = y[i].numpy().astype(int)

            print(f"\nImage {i + 1}:")
            print(f"  True:             {correct_label}")
            print(f"  Predicted:  {predicted_label}")

            safest_direction = int(torch.argmax(output).item())
            direction_taken = safest_direction - 3  # 0->-3, 1->-2, 2->-1, 3->0, 4->+1, 5->+2, 6->+3
            print(f"  Direction: {direction_taken:+d}")  # +/- sign always shown


def visualise(model, base_dir, json_data, y, image_width, image_height, partitions, num_images=5, random=False,
              folder="6_march"):
    """
    Visualisation function on a data set to test
    """
    # Made heavily with the help of AI
    bw_folder    = os.path.join(base_dir, "..", "..", "training_images", f"training_b&w_{folder}")
    color_folder = os.path.join(base_dir, "..", "..", "training_images", f"training_color_{folder}")

    if folder == "6_march":
        all_entries = list(json_data.items())
        entries = __import__('random').sample(all_entries, num_images) if random else all_entries[:num_images]
        filenames = [e[0] for e in entries]
        has_labels = True
    else:
        all_filenames = os.listdir(bw_folder)
        filenames = __import__('random').sample(all_filenames, num_images) if random else all_filenames[:num_images]
        has_labels = False

    fig, axes = plt.subplots(num_images, 2, figsize=(12, num_images * 3))

    for row, filename in enumerate(filenames):
        color_img = Image.open(os.path.join(color_folder, filename)).convert('RGB')
        bw_img    = Image.open(os.path.join(bw_folder, filename)).convert('L')
        color_w, color_h = color_img.size
        bw_w, bw_h = bw_img.size

        # run through model
        image_resized = bw_img.resize((image_width, image_height), Image.NEAREST)
        image_tensor = torch.tensor(np.array(image_resized, dtype=np.float32).flatten() / 255).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)[0]
        pred_z = (output > 0.5).numpy().astype(int)

        # top: color image
        ax_top = axes[row, 0]
        ax_top.imshow(color_img)
        ax_top.axis('off')
        zone_width = color_w / partitions

        if has_labels:
            idx = list(json_data.keys()).index(filename)
            true_z = y[idx].numpy().astype(int)
            ax_top.set_title(true_z, fontsize=8)
            for z in range(partitions):
                if true_z[z] == 1:
                    rect = patches.Rectangle((z * zone_width, 0), zone_width, color_h,
                                             linewidth=0, edgecolor='none', facecolor='red', alpha=0.35)
                    ax_top.add_patch(rect)
        else:
            ax_top.set_title("no labels", fontsize=8)
            for z in range(partitions):
                if pred_z[z] == 1:
                    rect = patches.Rectangle((z * zone_width, 0), zone_width, color_h,
                                             linewidth=0, edgecolor='none', facecolor='red', alpha=0.35)
                    ax_top.add_patch(rect)

        for z in range(1, partitions):
            ax_top.axvline(x=z * zone_width, color='white', linewidth=0.8, alpha=0.6)

        # bottom: b&w image with predicted zones
        ax_bot = axes[row, 1]
        ax_bot.imshow(bw_img, cmap='gray')
        ax_bot.axis('off')
        zone_width_bw = bw_w / partitions
        for z in range(partitions):
            if pred_z[z] == 1:
                rect = patches.Rectangle((z * zone_width_bw, 0), zone_width_bw, bw_h,
                                         linewidth=0, edgecolor='none', facecolor='red', alpha=0.35)
                ax_bot.add_patch(rect)
        for z in range(1, partitions):
            ax_bot.axvline(x=z * zone_width_bw, color='white', linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    # plt.savefig('visualisation.png', dpi=120, bbox_inches='tight')
    # print("\nSaved visualisation.png")
    plt.show()

########################################################################################################################
# MAIN #################################################################################################################
########################################################################################################################

def main():
    """
    Main function which just runs the entire script
    """
    (base_dir, json_path, image_width, image_height, partitions, neurons_number, epochs, learning_rate,
     input_size) = get_inputs()

    X, y, json_data = load_dataset_from_json(base_dir, json_path, image_width, image_height, partitions)

    model = build_model(input_size, partitions)

    model = train_model(model, X, y, epochs, learning_rate)

    evaluate_model(model, X, y)

    for _ in range(2):
        visualise(model, base_dir, json_data, y, image_width, image_height, partitions, num_images=10, random=True,
                  folder="13_march")


if __name__ == "__main__":
    main()