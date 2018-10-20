import sys
sys.path.append("..")
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from trainer.functions import Trainer
from tensorflow.data import Dataset
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.train import AdamOptimizer
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.utils import to_categorical
from tensorflow.losses import mean_squared_error
from tensorflow.nn import relu, softmax
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class Net(Model):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = Dense(512, activation=None)
        self.fc2 = Dense(512, activation=None)
        self.fc3 = Dense(10, activation=None)

    def call(self, inputs, training=False, mask=None):
        x = relu(self.fc1(inputs))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return softmax(x)


if __name__ == '__main__':
    tf.enable_eager_execution()
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_val = x_val.astype("float32")

    x_buf = x_train.shape[0]
    x_test_buf = x_test.shape[0]
    x_val_buf = x_val.shape[0]

    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)
    x_val = x_val.reshape(x_val.shape[0], 784)



    print(x_test.shape)
    x_train /= 255.0
    x_test /= 255.0
    x_val /= 255.0

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    train_ds = Dataset.from_tensor_slices((x_train, y_train)).shuffle(x_buf).batch(32)
    test_ds = Dataset.from_tensor_slices((x_test, y_test)).shuffle(x_test_buf).batch(32)
    val_ds = Dataset.from_tensor_slices((x_val, y_val)).shuffle(x_val_buf).batch(32)

    optimizer = AdamOptimizer()
    model = Net()

    trainer = Trainer(model, mean_squared_error)
    trainer.fit(optimizer, train_ds, val_ds, epochs=1)
    trainer.test(test_ds)
