# Outline
- [1-Packages](#1-packages)
- [2-Data Preprocessing](#2-data-preprocessing)
  - [2.1 Input Data](#4.1)
  - [2.2 Training and Validation Data](#4.1)
- [3-Neural Network](#3-softmax-function)
- [4-Model Validation](#4)
- [5-Predictions](#4)

## 1-Packages
First we import all the packages used for the code.

``` python
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, BatchNormalization, Dropout
from tensorflow.keras.activations import relu
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

## 2-Data Preprocessing
### Input Data
We now input the training and test data and print their shape. After the data loading we distinguish the features (x_train) from the labels (y_train).

```python
train_data = np.loadtxt("train.csv",delimiter=",",skiprows=1)  
print("train data shape = ",train_data.shape)
test_data = np.loadtxt("test.csv",delimiter=",",skiprows=1)    
print("test data shape = ",test_data.shape)
x_train = train_data[:,1:]
y_train = train_data[:,0]
```

### Training and Validation Data
We split train set into training and cost validation data using train_test_split() function of sklearn (scikit) package.

```python
x_train = train_data[:,1:]
x_train_scaled = x_train/255.0
y_train = train_data[:,0]

########## Create train and cross validate data ################
x_tr, x_cv, y_tr, y_cv = train_test_split(x_train_scaled,y_train,test_size=0.20, random_state=1)

########## reshape the images for conv2D layer #################
#### shape of array is (batch_size, height, width, channels) ###
#### batch_size = number of samples
#### channels = number of values for it's pixel (1 for grayscale)

print (x_tr.shape)
samples = x_tr.shape[0]
x_tr = x_tr.reshape(samples,28,28,1)
cv_num = x_cv.shape[0]
x_cv = x_cv.reshape(cv_num,28,28,1)
print (x_tr.shape)
```

## 3-Neural Network
The chosen architecture consists of two Convolutional layers followed by two Dense layers. Each Convolutional layer is followed by a BatchNormalization and a MaxPooling layer to stabilize training, accelerate convergence, and help mitigate overfitting. A Flatten layer is applied after the second Convolutional block to convert feature maps into a 1D vector. Additionally, Dropout layers are included before each Dense layer for regularization purposes. The model’s parameters were fine-tuned to avoid both underfitting (high bias) and overfitting (high variance).
 
| Layer (type)       | Output Shape        | Param # |
|--------------------|---------------------|---------|
| Conv2D             | (None, 28, 28, 32)  | 320     |
| BatchNormalization | (None, 28, 28, 32)  | 128     |
| Activation         | (None, 28, 28, 32)  | 0       |
| MaxPooling2D       | (None, 14, 14, 32)  | 0       |
| Conv2D             | (None, 14, 14, 64)  | 18496   |
| BatchNormalization | (None, 14, 14, 64)  | 256     |
| Activation         | (None, 14, 14, 64)  | 0       |
| MaxPooling2D       | (None, 7, 7, 64)    | 0       |
| Flatten            | (None, 3136)        | 0       |
| Dropout            | (None, 3136)        | 0       |
| Dense              | (None, 128)         | 401536  |
| BatchNormalization | (None, 128)         | 512     |
| Activation         | (None, 128)         | 0       |
| Dropout            | (None, 128)         | 0       |
| Dense              | (None, 10)          | 1290    |

```python
model = Sequential([
    tf.keras.Input(shape=(28,28,1)),
    Conv2D(filters=32, kernel_size = 3, padding = "same", name="L1"),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=2),

    Conv2D(filters=64, kernel_size = 3, padding = "same", name="L2"),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=2),
    Flatten(),

    Dropout(0.25),
    Dense(128, name="L3"),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.4),
    Dense(10, activation='softmax', name="L4")
])
model.summary()

model.compile(optimizer=Adam(learning_rate=0.001),loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
history = model.fit(
   	x_tr, y_tr,
   	validation_data = (x_cv, y_cv),
   	epochs=50
)
```

## 4-Model Validation
To assess the model’s learning behavior, we visualize the training and validation loss over epochs. This helps identify issues such as overfitting or underfitting.

```python
label_1 = 'Training Loss '
label_2 = 'Validation Loss '
plt.plot(history.history['loss'], label=label_1)
plt.plot(history.history['val_loss'], label=label_2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()
```



## 5-Predictions
After confirming that the model generalizes well—showing no signs of overfitting or underfitting—we move on to generate predictions on the test set.

```python
test_samples = test_data.shape[0]
x_test_scaled = test_data/255.0
x_test_scaled = x_test_scaled.reshape(test_samples,28,28,1)
# predictions is 2d array with shape [1,4]
predictions = model.predict(x_test_scaled)
predictions_descaled = predictions*255.0
im_id = np.arange(1,28001)
indexes = np.argmax(predictions,axis=1).astype(int)
predict = np.column_stack((im_id,indexes))
print(predict)
```
