import tensorflow as tf
import logging
from evaluation.metrics import ConfusionMatrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def evaluate(model, ds_test):
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # For a better evaluation, we need AUC of ROC curve. Our goal is to get left upper corner,
    # namely, the best case (the best model)
    roc_curve_auc = tf.keras.metrics.AUC(curve='ROC')
    cm = ConfusionMatrix(12)

    # Reset test metrics
    test_loss.reset_states()
    test_accuracy.reset_states()

    for idx, (test_image, test_label) in enumerate(ds_test):
        test_label = test_label - 1
        pred = model(test_image, training=False)
        t_loss = loss_object(test_label, pred)
        cm.update_state(test_label, pred)
        test_loss(t_loss)
        test_accuracy(test_label, pred)

        # convert from one-hot coding to normal index
        #pred = tf.math.argmax(pred, axis=1)
        #roc_curve_auc(test_label, pred)
    cm_result = np.ma.round(cm.result() / np.sum(cm.result()), 3) * 100

    visualize_cm(cm_result)

    template = 'Test Loss: {}, Test Accuracy: {}, Confusion Matrix:{}'
    logging.info(template.format(
        test_loss.result(),
        test_accuracy.result() * 100,
        cm_result
        #roc_curve_auc.result(),
        #cm.result_detailed()
        ))

    return

def visualize_cm(cm):
    # Visualize Confusion Matrix
    sns.heatmap(cm, annot=True)
    plt.show()