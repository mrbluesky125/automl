# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 08:37:49 2020

@author: zimmermann
"""

import os
import time
from typing import Text, Tuple, List

from absl import app
from absl import flags
from absl import logging

import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf

import cv2

import hparams_config
import inference
import utils
from tensorflow.python.client import timeline  # pylint: disable=g-direct-tensorflow-import
import microserviceclient

flags.DEFINE_string('service_name', 'k4aservice_000093201412', 'Service name.')
flags.DEFINE_string('model_name', 'efficientdet-d0', 'Model.')
flags.DEFINE_string('logdir', '/tmp/deff/', 'log directory.')
flags.DEFINE_string('trace_filename', None, 'Trace file name.')

flags.DEFINE_integer('threads', 0, 'Number of threads.')
flags.DEFINE_string('tensorrt', None, 'TensorRT mode: {None, FP32, FP16, INT8}')
flags.DEFINE_bool('delete_logdir', True, 'Whether to delete logdir.')
flags.DEFINE_bool('use_xla', False, 'Run with xla optimization.')
flags.DEFINE_integer('batch_size', 1, 'Batch size for inference.')
flags.DEFINE_string('ckpt_path', None, 'checkpoint dir used for eval.')
flags.DEFINE_string(
    'hparams', '', 'Comma separated k=v pairs of hyperparameters or a module'
                   ' containing attributes to use as hyperparameters.')

# For visualization.
flags.DEFINE_integer('line_thickness', 1, 'Line thickness for box.')
flags.DEFINE_integer('max_boxes_to_draw', 100, 'Max number of boxes to draw.')
flags.DEFINE_float('min_score_thresh', 0.4, 'Score threshold to show box.')
flags.DEFINE_string('nms_method', 'hard', 'nms method, hard or gaussian.')

# For saved model.
flags.DEFINE_string('saved_model_dir', '/tmp/saved_model', 'Folder path for saved model.')

FLAGS = flags.FLAGS

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)


def main(_):
    #do magic with model_config
    model_config = hparams_config.get_detection_config(FLAGS.model_name)
    model_config.override(FLAGS.hparams)  # Add custom overrides
    model_config.is_training_bn = False
    model_config.image_size = utils.parse_image_size(model_config.image_size)
    model_config.nms_configs.score_thresh = FLAGS.min_score_thresh
    model_config.nms_configs.method = FLAGS.nms_method
    model_config.nms_configs.max_output_size = FLAGS.max_boxes_to_draw

    client = microserviceclient.MicroserviceClient("efficientdetservice_" + FLAGS.service_name)
    driver = inference.ServingDriver(
        FLAGS.model_name,
        FLAGS.ckpt_path,
        batch_size=1,
        use_xla=FLAGS.use_xla,
        model_params=model_config.as_dict())
    driver.load(FLAGS.saved_model_dir)

    def on_binaryNotification_handler(methodName, payload):
        nonlocal driver
        nonlocal client
        if methodName == "doInferencePlease":
            frame = cv2.imdecode(np.asarray(bytearray(payload), dtype=np.uint8), cv2.IMREAD_COLOR)
            raw_frames = [np.array(frame)]
            detections_bs = driver.serve_images(raw_frames)
            client.notify("inferenceResult", driver.to_json(detections_bs[0]))
            
            #new_frame = driver.visualize(raw_frames[0], detections_bs[0], min_score_thresh=FLAGS.min_score_thresh, max_boxes_to_draw=FLAGS.max_boxes_to_draw, line_thickness=FLAGS.line_thickness )
            #res, boxedFrame = cv2.imencode('.jpg', new_frame)
            #client.binaryNotify("inferenceResult", boxedFrame.tobytes())

    client.on_binaryNotification = on_binaryNotification_handler
    client.start()

    while True:
        time.sleep(0.10)

    client.stop()

if __name__ == '__main__':
    
    logging.set_verbosity(logging.WARNING)
    tf.enable_v2_tensorshape()
    tf.disable_eager_execution()
    app.run(main)
