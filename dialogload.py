# read celeba dialog json

import json
import random
from prompttemplate import dialog_prompt

celeb_path = "F:\DLdataset\CelebA\Img\img_celeba\img_celeba\\"
json_path = "F:\DLdataset\CelebA-Dialog\celeba_caption\captions.json"
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
        img_paths.append(celeb_path + key)
        text = dialog_prompt(data[key]["overall_caption"], 5)
        # print(text)
        captions.append(text)
    return img_paths, captions

def load_samples(n):
    data = read_dialog_json(json_path)
    img_paths, captions = sample_dialog(data, n)
    return img_paths, captions


# test
# img_paths, captions = load_samples(10)
# print(img_paths)
# print(captions)
