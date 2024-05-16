import numpy as np
import torch
import clip
# from dialogload import load_samples
from celebAload import load_samples
from PIL import Image
from matplotlib import pyplot as plt
# load model
# calc matrix and softmax
# calc top1 top 3 and so on
# draw matrix figure


# load model
clip_type = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 
             'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'][4]
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(clip_type, device=device)

# calc matrix and softmax
def calc_matrix(img_paths, captions):
    images = [preprocess(Image.open(img_path)).unsqueeze(0).to(device) for img_path in img_paths]
    # images to tensor
    images = torch.cat(images, dim=0)
    text = clip.tokenize(captions).to(device)

    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text)

    # similarity matrix
    similarity_matrix = image_features @ text_features.T
    # softmax
    similarity_matrix = torch.softmax(similarity_matrix, dim=1)
    return similarity_matrix

# from similarity matrix to top1 accuracy
def topk(matrix):
    top1 = matrix.topk(1, dim=1)
    count_top1 = 0
    for i in range(len(top1.indices)):
        if top1.indices[i] == i:
            count_top1 += 1

    top3 = matrix.topk(3, dim=1)
    count_top3 = 0
    for i in range(len(top3.indices)):
        if i in top3.indices[i]:
            count_top3 += 1

    return count_top1, count_top3

def test_batchs(n):
    batch_size = 10
    count_top1_sum = 0
    count_top3_sum = 0
    for i in range(n):
        image_paths, captions = load_samples(batch_size)
        matrix = calc_matrix(image_paths, captions)
        count_top1,count_top3 = topk(matrix)
        count_top1_sum += count_top1
        count_top3_sum += count_top3
    total = n * batch_size
    return count_top1_sum/total, count_top3_sum/total
    
# draw matrix figure
def draw_matrix(matrix):
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix, vmin=0., vmax=0.5)
    plt.colorbar()
    plt.show()
    # save plot
    plt.savefig('01capture/matrix.png')


# test dialog
# top1,top3 = test_batchs(30)
# print(top1, top3)


# test celebA
# top1,top3 = test_batchs(30)
# print(top1, top3)







# image_paths, captions = load_samples(20)
# matrix = calc_matrix(image_paths, captions)
# draw_matrix(matrix)
