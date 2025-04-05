import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications import vgg19, resnet50, mobilenet_v2
from construct import construct
import random
import json

jobs = []

model_list = []
model = mobilenet_v2.MobileNetV2()
model_list.append(model)
model = vgg19.VGG19()
model_list.append(model)
model = resnet50.ResNet50()
model_list.append(model)

for i in range(50):
    model = model_list[random.randint(0,2)]
    fields = model.summary()
    job = construct(fields, 0, 0, 0, 0)
    jobs.append(job)
    print(job)

with open("jobs.json", "w") as wf:
    json.dump(jobs, wf)
