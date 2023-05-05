import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models, optimizers, losses, callbacks,\
                             regularizers
drive.mount('/content/drive')

!unzip -q "/content/drive/MyDrive/Group_Project_Data.zip"

train = tf.keras.utils.image_dataset_from_directory('/content/Group_Project_Data/Train', labels='inferred')
valid = tf.keras.utils.image_dataset_from_directory('/content/Group_Project_Data/Valid', labels='inferred')

class_names = train.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

for image_batch, labels_batch in train:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

Input = layers.Input((256, 256, 3), dtype='float32', name='Input')
conv_1 = layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                       padding='same', name='conv_1')(Input)
drop_1 = layers.Dropout(0.2)(conv_1)
conv_2 = layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                       padding='same', name='conv_2',
                       kernel_regularizer=regularizers.l2(0.0001))(drop_1)
pool_1 = layers.MaxPool2D(pool_size=(2,2), name='pool_1')(conv_2)
conv_3 = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                       padding='same', name='conv_3')(pool_1)
drop_2 = layers.Dropout(0.2)(conv_3)
conv_4 = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                       padding='same', name='conv_4',
                       kernel_regularizer=regularizers.l2(0.0001))(drop_2)
pool_2 = layers.MaxPool2D(pool_size=(2,2), name='pool_2')(conv_4)
conv_5 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                       padding='same', name='conv_5')(pool_2)
drop_3 = layers.Dropout(0.2)(conv_5)
conv_6 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                       padding='same', name='conv_6',
                       kernel_regularizer=regularizers.l2(0.0001))(drop_3)
pool_3 = layers.MaxPool2D(pool_size=(2,2), name='pool_3')(conv_6)

conv_7 = layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                       padding='same', name='conv_7')(pool_3)
drop_4 = layers.Dropout(0.2)(conv_7)
conv_8 = layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                       padding='same', name='conv_8',
                       kernel_regularizer=regularizers.l2(0.0001))(drop_4)
pool_4 = layers.MaxPool2D(pool_size=(2,2), name='pool_4')(conv_8)
flat = layers.Flatten()(pool_4)
fc_1 = layers.Dense(256, activation='relu', name='fc_1')(flat)
Output = layers.Dense(1, activation='sigmoid', name='Output')(fc_1)

model = models.Model(inputs=Input, outputs=Output, name='model')
model.summary()

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,
              loss='BinaryCrossentropy',
              metrics=['accuracy'])

# Utility function that resets the weights of your model. Call this before
# recompiling your model with updated settings, to ensure you train the model
# from scratch.

def reinitialize(model):
    # Loop over the layers of the model
    for l in model.layers:
        # Check if the layer has initializers
        if hasattr(l,"kernel_initializer"):
            # Reset the kernel weights
            l.kernel.assign(l.kernel_initializer(tf.shape(l.kernel)))
        if hasattr(l,"bias_initializer"):
            # Reset the bias
            l.bias.assign(l.bias_initializer(tf.shape(l.bias)))

# Function modified from here: https://stackoverflow.com/questions/63435679/reset-all-weights-of-keras-model

reinitialize(model)
history = model.fit(train, validation_data=valid, batch_size=32, epochs=15)
