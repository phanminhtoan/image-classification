import keras
from tensorflow.keras.applications import MobileNetV2
import argparse
import tensorflowjs

model = MobileNetV2(weights = 'imagenet')
tensorflowjs.converters.save_keras_model(model, 'imagenet')