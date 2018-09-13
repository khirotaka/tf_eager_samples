# coding: utf-8
"""
TensorFlow Eager Execution
CIFAR-10 using Convolution Neural Network


Required: Python 3.6
          TensorFlow 1.10.1
          Matplotlib 2.2.3


Copyright (c) 2018 Hirotaka Kawashima
"""
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.utils import to_categorical
from tensorflow.train import AdamOptimizer
from tensorflow.layers import flatten
from tensorflow.nn import relu
import matplotlib.pyplot as plt
import utils


class Net(Model):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2D(32, (2, 2), padding="same", activation=None)
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPooling2D((2, 2))
        self.conv2 = Conv2D(64, (2, 2), padding="same", activation=None)
        self.bn2 = BatchNormalization()
        self.pool2 = MaxPooling2D((2, 2))
        self.fc1 = Dense(512, activation=None)
        self.fc2 = Dense(10, activation=None)

    def call(self, inputs, training=False, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = relu(x)
        x = self.pool2(x)
        x = flatten(x)
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    tf.enable_eager_execution()

    num_classes = 10
    batch_size = 32
    epochs = 2

    (x_train, y_train), (x_test, y_test) = load_data()

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    print(x_test.shape)
    x_train /= 255
    x_test /= 255

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    train_ds = Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(batch_size)
    test_ds = Dataset.from_tensor_slices((x_test, y_test)).shuffle(10000).batch(batch_size)

    model = Net()
    optimizer = AdamOptimizer()

    # If you have a gpu on your computer, specify device="gpu:0".
    utils.train(model, optimizer, train_ds, epochs, device="cpu:0")
    utils.test(model, test_ds, device="cpu:0")

    class_name = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    x, y = iter(test_ds).next()
    pred = utils.predict(model, x)

    plt.imshow(x[0].numpy())
    plt.title(class_name[pred[0]])
    plt.show()
