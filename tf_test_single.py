# coding=utf-8
# summary:
# author: Jianqiang Ren
# date:

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from util import load_image
from PIL import Image
import cv2

f = gfile.FastGFile("./download/tf_model.pb", 'rb')
graph_def = tf.GraphDef()
# Parses a serialized binary message into the current message.
graph_def.ParseFromString(f.read())
persisted_graph = tf.import_graph_def(graph_def, name='')
f.close()

sess = tf.InteractiveSession(graph=persisted_graph)

lr_img = tf.get_default_graph().get_tensor_by_name("input_1:0")

output = tf.get_default_graph().get_tensor_by_name("lambda_5/add:0")

lr = load_image('demo/0829x4-crop.png')
lr_feed = np.expand_dims(lr, axis=0)
output_value = sess.run(output, feed_dict={lr_img: lr_feed})[0]
sr = np.clip(output_value, 0, 255)
sr = sr.astype('uint8')
sr = Image.fromarray(sr)
sr.save('download/sr.png')
print('done')