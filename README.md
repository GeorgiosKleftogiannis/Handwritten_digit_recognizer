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

```
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