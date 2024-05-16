# chinese
# ffmpeg

# read frame from video

import cv2
import numpy as np
import os
import time
import datetime
import insightface
from skimage import transform
# from PIL import Image, ImageDraw
from PIL import Image, ImageDraw, ImageFont


det = insightface.model_zoo.SCRFD("det.onnx")


expression_name = "测试中文"
det = insightface.model_zoo.SCRFD("det.onnx")


def face_align_landmarks_sk(img, landmarks, image_size=(112, 112), method="similar"):
    tform = transform.AffineTransform() if method == "affine" else transform.SimilarityTransform()
    src = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.729904, 92.2041]], dtype=np.float32)
    ret = []
    for landmark in landmarks:
        # landmark = np.array(landmark).reshape(2, 5)[::-1].T
        tform.estimate(landmark, src)
        ret.append(transform.warp(img, tform.inverse, output_shape=image_size))
    # ret = np.transpose(ret, axes=[0,3,1,2])
    return (np.array(ret) * 255).astype(np.uint)

def do_detect_in_image(image, det, image_format="BGR"):
    imm_BGR = image if image_format == "BGR" else image[:, :, ::-1]
    imm_RGB = image[:, :, ::-1] if image_format == "BGR" else image
    bboxes, pps = det.detect(imm_BGR, input_size = (640,640))
    nimgs = face_align_landmarks_sk(imm_RGB, pps)
    bbs, ccs = bboxes[:, :4].astype("int"), bboxes[:, -1]
    return bbs, ccs,pps, nimgs

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def print_image(frame,saveImagePath):
    #detect face and save path
    bbs,ccs,pps,imgs = do_detect_in_image(frame,det)
    #draw box on image with nice color
    for i in range(len(bbs)):
        # use pil to draw box and text
        # draw = ImageDraw.Draw(frame)
        # draw.rectangle(bbs[i], outline=(255, 0, 0))
        # draw.text((bbs[i][0], bbs[i][1]), expression_name, fill=(255, 0, 0))

        cv2.rectangle(frame, (bbs[i][0], bbs[i][1]+30), (bbs[i][2], bbs[i][3]), (0, 255, 0), 2)
        frame = cv2AddChineseText(frame, expression_name, (bbs[i][0], bbs[i][1]))
    cv2.imwrite(saveImagePath, frame)
    # img.save(saveImagePath)



video_path = "source2.mkv"
video = cv2.VideoCapture()
video.open(video_path)

# mkdir
tmp_dir = "tmp"
if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)

# empty tmp dir
for file in os.listdir(tmp_dir):
    os.remove(os.path.join(tmp_dir,file))

count = 0
image_count = 0
while True:
    ret,frame = video.read()
    if ret:
        count += 1
        print(frame.shape)
        if count % 10 == 0:
            image_count += 1
            # image_name with fixed length
            image_name = str(image_count).zfill(6) + ".jpg"
            image_path = os.path.join(tmp_dir,image_name)
            print_image(frame,image_path)
    else:
        break

# merge image in tmp dir to video using ffmpeg
# os.system("ffmpeg -r 25 -i tmp/%d.jpg -vcodec libx264 -crf 25 -pix_fmt yuv420p output.mp4")

# ffmpeg -r 10 -f image2 -i C:/Users/btwlo/Desktop/final/tmp/%6d.jpg output.mp4