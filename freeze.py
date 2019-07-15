# coding=utf-8
# summary:
# author: Jianqiang Ren
# date:

import os
import glob
import logging
import argparse
import numpy as np
# from model import load_model
from util import load_image, init_session
from PIL import Image
import datetime
import tensorflow as tf
from keras import backend as K
from keras.models import  load_model
from optimizer import weightnorm as wn
from keras.losses import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

 
def mae(hr, sr):
    return mean_absolute_error(hr, sr)

def psnr(hr, sr):
    return mean_absolute_error(hr, sr)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
 
    K.set_learning_phase(0)
 
    _custom_objects = {
        'tf': tf,
        'AdamWithWeightnorm': wn.AdamWithWeightnorm,
        'mae': mae,
        'psnr': psnr
    }

    _custom_objects_backwards_compat = {
        'mae_scale_2': mae,
        'mae_scale_3': mae,
        'mae_scale_4': mae,
        'psnr_scale_2': psnr,
        'psnr_scale_3': psnr,
        'psnr_scale_4': psnr
    }
    
    custom = {}
    custom.update(_custom_objects)
    custom.update(_custom_objects_backwards_compat)
    model = load_model('download/wdsr-a-32-x4-psnr-29.1736.h5', custom_objects=custom)
 
    print(model.inputs)
    print(model.outputs)
    frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, "download", "wdsr-a-32-x4.pb", as_text=False)
    print("freeze done")