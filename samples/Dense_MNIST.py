# coding: utf-8
"""
TensorFlow Eager Execution
MNIST using dense neural network


Required: Python 3.6
          TensorFlow 1.10.1
          Matplotlib 2.2.3


Copyright (c) 2018 Hirotaka Kawashima
"""
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.data import Dataset
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.train import AdamOptimizer
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.utils import to_categorical
from tensorflow.nn import relu, softmax_cross_entropy_with_logits_v2
import matplotlib.pyplot as plt


class Net(Model):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = Dense(512, activation=None)
        self.fc2 = Dense(512, activation=None)
        self.fc3 = Dense(num_classes, activation=None)

    def call(self, inputs, training=False, mask=None):
        x = relu(self.fc1(inputs))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x


def loss(model, x, y, training=False):
    prediction = model(x, training)
    return softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y)


def grad(model, x, y, training=False):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, y, training)
    return tape.gradient(loss_value, model.variables)


def train(model, dataset, epochs):
    for e in range(epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        train_accuracy = tfe.metrics.Accuracy()
        x, y = iter(train_ds).next()
        for (i, (x, y)) in enumerate(dataset):
            grads = grad(model, x, y, training=True)
            optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())
            epoch_loss_avg(loss(model, x, y, training=True))
            train_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32),
                           tf.argmax(y, axis=1, output_type=tf.int32))
            if i % 200 == 0:
                print("Loss: {:.4f} - Acc: {:.4f}".format(epoch_loss_avg(loss(model, x, y)), train_accuracy.result()))

        print("-"*50)
        print("Epochs {} / {} | Loss: {:.4f} - Accuracy: {:.3%}".format(e+1, epochs, epoch_loss_avg(loss(model, x, y)), train_accuracy.result()))


def test(model, dataset):
    test_accuracy = tfe.metrics.Accuracy()
    for (x, y) in dataset:
        test_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32),
                      tf.argmax(y, axis=1, output_type=tf.int32))

    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))


def predict(model, x, categories):
    pred = model(x)
    result = []
    for logits in pred:
        class_idx = tf.argmax(logits).numpy()
        name = categories[class_idx]
        result.append(name)
    return result


if __name__ == '__main__':
    tf.enable_eager_execution()

    num_classes = 10
    batch_size = 32
    epochs = 10

    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train.reshape(60000, 784) / 255
    x_test = x_test.reshape(10000, 784) / 255

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    train_ds = Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(batch_size)
    test_ds = Dataset.from_tensor_slices((x_test, y_test)).shuffle(10000).batch(batch_size)

    optimizer = AdamOptimizer()
    model = Net()

    train(model, train_ds, epochs=2)
    test(model, test_ds)

    class_name = [str(i) for i in range(num_classes)]

    x, y = iter(test_ds).next()
    pred = predict(model, x, class_name)

    plt.imshow(x[0].numpy().reshape(28, 28))
    plt.title(pred[0])
    plt.show()
