import glob
import numpy as np
import torch
import clip
from PIL import Image
from matplotlib import pyplot as plt

# load model
clip_type = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 
             'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'][4]
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(clip_type, device=device)
mask_path = "F:/DLdataset/mask/yes/"
unmask_path = "F:/DLdataset/mask/no/"

# glob mask and unmask
mask_paths = glob.glob(mask_path + "*.jpg")
unmask_paths = glob.glob(unmask_path + "*.jpg")
images_paths = mask_paths + unmask_paths

# labes
labels = [1] * len(mask_paths) + [0] * len(unmask_paths)

# prompt

prompt = [
    "a person no wearing a mask",  
    "a person wearing a mask",
]

images_features = []
for img_path in images_paths:
    img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feature = model.encode_image(img)
    images_features.append(img_feature)
images_features = torch.cat(images_features, dim=0)

text_features = model.encode_text(clip.tokenize(prompt).to(device))

# similarity matrix
similarity_matrix = images_features @ text_features.T
# softmax
similarity_matrix = torch.softmax(similarity_matrix, dim=1)
#argmax
similarity_matrix = torch.argmax(similarity_matrix, dim=1).to("cpu")
print(similarity_matrix)
# accuracy
accuracy = (similarity_matrix == torch.tensor(labels)).sum().item() / len(labels)
print(accuracy)

# give lables and predict to return recall and precision
def recall_precision(labels, predict):
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(labels)):
        if labels[i] == 1 and predict[i] == 1:
            TP += 1
        elif labels[i] == 0 and predict[i] == 1:
            FP += 1
        elif labels[i] == 1 and predict[i] == 0:
            FN += 1
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    return recall, precision
### 0.97

print(recall_precision(labels, similarity_matrix))


# 口罩识别的实验

# manual prompt
# "a photo featuring a person not wearing a mask",
# "a photo featuring a person wearing a mask",  
# acc 0.97 recall 0.94 precision 0.97

# llm prompt
# "a person wearing a mask",
# "a person without a mask",  
# acc 0.935 recall 0.94 precision 0.9306