# coding=utf-8
# summary:
# author: Jianqiang Ren
# date:

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from PIL import Image
import cv2


def load_image(path):
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


if __name__ == "__main__":
    f = gfile.FastGFile("./download/wdsr-a-32-x4.pb", 'rb')
    graph_def = tf.GraphDef()
    # Parses a serialized binary message into the current message.
    graph_def.ParseFromString(f.read())
    persisted_graph = tf.import_graph_def(graph_def, name='')
    f.close()
    sess = tf.InteractiveSession(graph=persisted_graph)
    lr_img = tf.get_default_graph().get_tensor_by_name("input_1:0")
    output = tf.get_default_graph().get_tensor_by_name("lambda_5/add:0")

    lr = cv2.imread('demo/0829x4-crop.png')

    if len(lr.shape) == 2:
        lr = np.dstack((lr, lr, lr))
    elif lr.shape[2] == 4:
        alpha = lr[:, :, 3]
        lr = lr[:, :, :3]
    lr = lr[:, :, ::-1]
    
    lr_feed = np.expand_dims(lr, axis=0)
    output_value = sess.run(output, feed_dict={lr_img: lr_feed})[0]
    sr = np.clip(output_value, 0, 255)
    sr = sr.astype('uint8')
    sr = sr[:,:,::-1]
    cv2.imwrite('download/sr.png', sr)
    print('done')
    