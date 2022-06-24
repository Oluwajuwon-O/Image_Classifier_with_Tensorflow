import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

import warnings
warnings.filterwarnings('ignore')
import logging
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL  import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

p = argparse.ArgumentParser()
p.add_argument('input',action='store',type=str,help='image path')
p.add_argument('model',action='store',type=str,help='model path')
p.add_argument('--category_names',default='./label_map.json',action='store',type=str,help='label mapping')
p.add_argument('--top_k',default=5,action='store',type=int,help='most likely class labels')
arg_parser = p.parse_args()
top_k = arg_parser.top_k


def process_image(image):
    image = tf.cast(image,tf.float32)
    image = tf.image.resize(image,(224,224))
    image /= 255
    return image

def predict(image_path,model,top_k):
    img = Image.open(image_path)
    image = np.asarray(img)
    processed_image = process_image(image)
    prob_predictions = model.predict(np.expand_dims(processed_image,axis=0))
    probs,labels = tf.nn.top_k(prob_predictions,k=top_k)
    probs = list(probs.numpy()[0])
    classes = list(labels.numpy()[0])
    return probs,classes

with open(arg_parser.category_names, 'r') as f:
    class_names = json.load(f)
    

model = tf.keras.models.load_model(arg_parser.model,compile = False,custom_objects={'KerasLayer':hub.KerasLayer})
path = './test_images/cautleya_spicata.jpg'

probs,labels = predict(arg_parser.input,model,top_k)
print('Probabilities:\n',probs)
print('Labels:\n',labels)
