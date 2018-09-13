# coding: utf-8
"""
Utility Functions for training, test and prediction of model.


Required: Python 3.6
          TensorFlow 1.10.1


Copyright (c) 2018 Hirotaka Kawashima
"""
import tensorflow as tf
import tensorflow.contrib.eager as tfe


def loss(model, x, y, training=False):
    prediction = model(x, training)
    return tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y)


def grad(model, x, y, training=False):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, y, training)
    return tape.gradient(loss_value, model.variables)


def train(model, optimizer, train_ds, val_ds, epochs, device="cpu:0"):
    with tf.device(device):
        for e in range(epochs):
            epoch_loss_avg = tfe.metrics.Mean()
            train_accuracy = tfe.metrics.Accuracy()
            val_accuracy = tfe.metrics.Accuracy()

            x, y = iter(train_ds).next()

            for (i, (x, y)) in enumerate(train_ds):
                grads = grad(model, x, y, training=True)
                optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())
                train_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32),
                               tf.argmax(y, axis=1, output_type=tf.int32))

                if i % 200 == 0:
                    for (x_val, y_val) in val_ds:
                        val_accuracy(tf.argmax(model(x_val), axis=1, output_type=tf.int32),
                                     tf.argmax(y_val, axis=1, output_type=tf.int32))
                    print("Loss: {:.4f} - Acc: {:.4f} | Val Acc: {:.4f}".format(
                        epoch_loss_avg(loss(model, x, y)), train_accuracy.result(), val_accuracy.result()
                    ))

            print("-"*50)
            print("Epochs {} / {} | Loss: {:.4f} - Accuracy: {:.3%}".format(
                e + 1, epochs, epoch_loss_avg(loss(model, x, y)), train_accuracy.result()
            ))


def test(model, dataset, device="cpu:0"):
    with tf.device(device):
        test_accuracy = tfe.metrics.Accuracy()
        for (x, y) in dataset:
            test_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32),
                          tf.argmax(y, axis=1, output_type=tf.int32))

        print("Test set accuracy: {:.3%}".format(test_accuracy.result()))


def predict(model, x):
    pred = model(x)
    result = []
    for p in pred:
        class_idx = tf.argmax(p).numpy()
        result.append(class_idx)
    return result
