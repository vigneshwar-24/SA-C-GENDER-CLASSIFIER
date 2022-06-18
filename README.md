# SA-C-GENDER-CLASSIFIER
## Algorithm

1.Import the necessary packages such as tensorflow, tensorflow hub, pandaa,matplotlib, and splitfloders.

2.Create your dataset of male and female images.

3.Using tensorflow.keras.preprocessing.image generate same image data for all the train and test images.

4.Using tensorflow.keras.preprocessing preprocess the images into numbers for gender prediction.

5.From tensorflow hub use B0 Feature vectors of EfficientNet models trained on Imagenet.

6.Use fit() to train the model.

7.Predict the gender of the image.

8.Create a loss and accuracy graph for understanding the model learning rate.

## Program:
```
/*
Program to implement 
Developed by   :Vigneshwar S
RegisterNumber :212220230058  
*/
```
```python
import splitfolders  # or import split_folders
splitfolders.ratio("Male and Female face dataset", output="output", seed=1337, ratio=(.9, .1), group_prefix=None) # default values

import matplotlib.pyplot as plt
import matplotlib.image as mping
img = mping.imread('rash.jpg')
plt.imshow(img)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory("output/train/",target_size=(224,224),seed=42,batch_size=32,class_mode="categorical")
test = train_datagen.flow_from_directory("output/val/",target_size=(224,224),seed=42,batch_size=32,class_mode="categorical")

from tensorflow.keras.preprocessing import image

test_image = image.load_img('rash.jpg', target_size=(224,224))
test_image = image.img_to_array(test_image)
test_image = tf.expand_dims(test_image,axis=0)
test_image = test_image/255.
test_image.shape

import tensorflow_hub as hub
m = tf.keras.Sequential([
hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"),
tf.keras.layers.Dense(2, activation='softmax')])

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

tf.debugging.set_log_device_placement(True)
m.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(),metrics=["accuracy"])

history = m.fit(train,epochs=2,steps_per_epoch=len(train),validation_data=test,validation_steps=len(test))

classes=train.class_indices
classes=list(classes.keys())

m.predict(test_image)

classes[tf.argmax(m.predict(test_image),axis=1).numpy()[0]]

import pandas as pd
pd.DataFrame(history.history).plot()

```


## OUTPUT:
![Capture73](https://user-images.githubusercontent.com/77089276/174422250-ad243c83-a250-4e5f-8c0b-cdd2d0ecac5c.PNG)
![Capture74](https://user-images.githubusercontent.com/77089276/174422253-1664ae00-58d7-4a80-9ba8-88356595b118.PNG)
![Capture75](https://user-images.githubusercontent.com/77089276/174422256-862c5780-03ac-46f4-9373-a838a12f093b.PNG)
![Capture76](https://user-images.githubusercontent.com/77089276/174422261-7a28e70e-89ae-4ad5-8392-911e1820390b.PNG)



DEMO VIDEO YOUTUBE LINK:https://youtu.be/YGMhLcPNcrc


## RESULT: 
The B0 Efficientnet NN model has been created for gender classification and has sucessfully predicted gender of the input images.
