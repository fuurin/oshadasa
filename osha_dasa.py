# チュートリアルを見ながらコメントをつけよう

import os.path, sys
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Flatten, Dropout
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard
import keras.callbacks
import numpy as np
from keras.optimizers import SGD

IMAGE_SIZE = 224
BATCH_SIZE = 32

NUM_TRAINING = 1260
NUM_VALIDATION = 140

# モデルの構築
input_tensor = Input(shape=(224, 224, 3))
resnet50_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
x = resnet50_model.output
x = Flatten(name='flatten')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax', name='predictions')(x)
model = Model(inputs=resnet50_model.input, outputs=predictions)
model.compile(
    optimizer=SGD(lr=1e-4, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

classes = ['osya', 'dasa']


train_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1.0 / 255
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_generator = train_datagen.flow_from_directory(
    'data/train_osha_dasa',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=classes,
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    'data/train_osha_dasa',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=classes,
    shuffle=True
)

# モデルの学習
hist = model.fit_generator(
    train_generator,
    steps_per_epoch=NUM_TRAINING//BATCH_SIZE,
    epochs=500,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=NUM_VALIDATION//BATCH_SIZE*10
)

model.save('model/osha_dasa_last.h5')