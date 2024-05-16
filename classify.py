import tensorflow as tf
import cv2
import numpy as np

class Classify():

    def __init__(self,model_path):
        # load model to second gpu
        with tf.device('/device:GPU:0'):
            self.model = tf.keras.models.load_model(model_path)

    def classify_matimg_result(self,mat_img):
        img_resize = cv2.resize(mat_img,(224,224))
        img_scaled = img_resize / 255.
        img_expand = np.expand_dims(img_scaled,axis=0)
        probability_dis = tf.nn.softmax(self.model(img_expand))
        index = np.argmax(probability_dis)
        return index

    def classify_batch(self,batch_img):
        img_resize = np.array([cv2.resize(item,(224,224)) for item in batch_img])
        img_scaled = img_resize / 255.
        features = self.model(img_scaled)
        probability_dis = tf.nn.softmax(features,axis=1)
        print(probability_dis)
        return probability_dis

classify = Classify("best_model.hdf5")

# list all gpu
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())