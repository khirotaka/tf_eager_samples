import tensorflow as tf
import tensorflow.contrib.eager as tfe


class Trainer:
    def __init__(self, model, loss_func):
        self.model = model
        self.loss_func = loss_func
        self.device = "/gpu:0" if tfe.num_gpus() > 0 else "/cpu:0"

    def loss(self, x, y, training=False):
        predictions = self.model(x, training)
        return self.loss_func(labels=y, predictions=predictions)            # tf.nn の softmax_cross_entropy_with_logitsは使えないので、ネットワークの構造を新たに設計する！

    def grad(self, x, y, training=False):
        with tf.GradientTape() as tape:
            loss_value = self.loss(x, y, training)
        return tape.gradient(loss_value, self.model.variables)

    def fit(self, optimizer, train_ds, val_ds, epochs=1):
        x, y = iter(train_ds).next()
        x_val, y_val = iter(val_ds).next()
        with tf.device(self.device):
            for epoch in range(epochs):
                epoch_loss_avg = tfe.metrics.Mean()
                train_accuracy = tfe.metrics.Accuracy()
                val_accuracy = tfe.metrics.Accuracy()

                for (i, (x, y)) in enumerate(train_ds):
                    grads = self.grad(x, y, training=True)
                    optimizer.apply_gradients(zip(grads, self.model.variables),
                                              global_step=tf.train.get_or_create_global_step())
                    train_accuracy(tf.argmax(self.model(x), axis=1, output_type=tf.int32),
                                   tf.argmax(y,             axis=1, output_type=tf.int32))

                    if i % 200 == 0:
                        for (x_val, y_val) in val_ds:
                            val_accuracy(tf.argmax(self.model(x_val), axis=1, output_type=tf.int32),
                                         tf.argmax(y_val,             axis=1, output_type=tf.int32))
                        print("Loss: {:.>4f} - Acc: {:.3%} | Val Acc: {:.3%}".format(
                            epoch_loss_avg(self.loss(x, y)), train_accuracy.result(), val_accuracy.result())
                        )

                val_accuracy(tf.argmax(self.model(x_val), axis=1, output_type=tf.int32),
                             tf.argmax(y_val,             axis=1, output_type=tf.int32))
                print("-"*70)
                print("Epochs {} / {} | Loss: {:.>4f} - Acc: {:.3%} | Val Acc: {:.3%}".format(
                    epoch+1, epochs, epoch_loss_avg(self.loss(x, y)), train_accuracy.result(), val_accuracy.result())
                )
                print("-"*70)

    def test(self, test_ds):
        with tf.device(self.device):
            test_accuracy = tfe.metrics.Accuracy()
            for (x, y) in test_ds:
                test_accuracy(tf.argmax(self.model(x), axis=1, output_type=tf.int32),
                              tf.argmax(y,             axis=1, output_type=tf.int32))

            print("Test Accuracy: {:.3%}".format(test_accuracy.result()))
