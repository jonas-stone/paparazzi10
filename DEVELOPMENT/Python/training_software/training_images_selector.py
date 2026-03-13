# Imports
import os
import random
import json

# How many photos in a folder

parent_folder = r"C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_Team_10\TEAM-10-PROTOTYPING\downloads from drone"

for folder in os.listdir(parent_folder):
    folder_path = os.path.join(parent_folder, folder)
    photos = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp"))]
    print(f"{folder} - {len(photos)} photos")

# Function to sample the photos


def sample_photos(folder_name, number_buckets, number_samples_bucket, output_file=None, testing=False):
    """
    INPUTS
    folder_name
    number_buckets : the number of intervals kind of thing
    number_samples_bucket : the number of samples in each individual interval
    output_file
    testing

    OUTPUTS
    sampled_flat : all the names of the samples photos in that folder
    """
    # Folder Selection and All Photos in a Folder Extraction
    photos = [os.path.join(folder_name, f) for f in os.listdir(folder_name) if f.endswith((".jpg",
                                                                                           ".jpeg",
                                                                                           ".png",
                                                                                           ".gif",
                                                                                           ".webp"))]

    # Bucketing of the Photos Vector
    number_photos = len(photos)
    photos_in_bucket = number_photos // number_buckets
    sampled = {i: random.sample(photos[i*photos_in_bucket:(i+1)*photos_in_bucket],
                                number_samples_bucket) for i in range(number_buckets)}
    sampled_flat = [photo for bucket in sampled.values() for photo in bucket]

    # Write to JSON if output_file is specified
    if output_file:
        data = {}
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                data = json.load(f)
        data[folder_name] = sampled_flat
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)

    # Testing and Printing Stuff
    if testing == True:
        total_number_samples = number_buckets * number_samples_bucket
        print('photos               - ', photos)
        print('photos[0]            - ', photos[0])
        print('number_photos        - ', number_photos)
        print('photos_in_bucket     - ', photos_in_bucket)
        print('sampled              - ', sampled)
        print('total_number_samples - ', total_number_samples)
        print('total_number_samples - ', sum(len(v) for v in sampled.values()))
        print('sampled_flat         - ', sampled_flat)
        print('len(sampled_flat)    - ', len(sampled_flat))

    return sampled_flat

# name_of_folder = r"C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_Team_10\TEAM-10-PROTOTYPING\downloads from drone\20240322-084506"
# name_of_folder = r"C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_Team_10\TEAM-10-PROTOTYPING\downloads from drone\20240322-091508"

def sample_adapter(input_json_file, number_buckets, number_samples_bucket):
    with open(input_json_file, 'r') as f:
        data = json.load(f)

    output_file = input_json_file.replace('.json', '_sampled.json')

    for folder_name in data.keys():
        sample_photos(folder_name, number_buckets=number_buckets, number_samples_bucket=number_samples_bucket, output_file=output_file)


sample_adapter(r"C:\Users\neytc\Documents\TU_Delft\lecture_notes\mav\MAV_Team_10\TEAM-10-PROTOTYPING\Python\training_software\useful_photos_color_6_march_new_trial.json",
               26, 19)

#

# # name_of_folder = "../../downloads from drone/20240322-084506"
# result = sample_photos(folder_name=name_of_folder, number_buckets=10, number_samples_bucket=9,output_file='test_samples_1', testing=False)
# print(result)
