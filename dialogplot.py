# read celeba dialog json

import json
import random
from PIL import Image
import matplotlib.pyplot as plt

json_path = "./dldataset/CelebA-Dialog/celeba_caption/captions.json"
def read_dialog_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

# random sample n keys to constract image path array and capture array
def sample_dialog(data, n):
    keys = random.sample(list(data.keys()), n)
    img_paths = []
    captions = []
    for key in keys:
        img_paths.append(key)
        captions.append(data[key]["overall_caption"])
    return img_paths, captions

# read 10 samples then plot and save
def plot_dialog(img_paths, captions):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    count = 0
    for img_path, caption in zip(img_paths, captions):
        count += 1
        plt.subplot(2, 2, count)
        # plot image
        image_path = "./dldataset/CelebA/Img/img_celeba/" + img_path
        img = Image.open(image_path)
        plt.imshow(img)
        # plot caption
        plt.title(caption, fontsize=8)
        # save
    save_path = "./01capture/" + "sample.png"
    plt.savefig(save_path)

#
data = read_dialog_json(json_path)
img_paths, captions = sample_dialog(data, 4)
plot_dialog(img_paths, captions)
print(img_paths)
print(captions)
