import tensorflow as tf


class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self, num_classes, **kwargs):
        super(ConfusionMatrix, self).__init__(name='confusion_matrix', **kwargs)  # handles base args (e.g., dtype)
        self.num_classes = num_classes
        self.total_cm = self.add_weight("total", shape=(num_classes, num_classes), initializer="zeros")

    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total_cm.assign_add(self.confusion_matrix(y_true, y_pred))
        return self.total_cm

    def result_detailed(self):
        return self.process_confusion_matrix()

    def result(self):
        return self.total_cm

    def confusion_matrix(self, y_true, y_pred):
        """Make a confusion matrix"""
        y_pred = tf.reshape(y_pred, [-1, 12])

        y_pred = tf.argmax(y_pred, 1)
        y_true = tf.reshape(y_true, [-1])

        cm = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32, num_classes=self.num_classes)
        return cm

    def process_confusion_matrix(self):
        """returns precision, recall and f1 along with overall accuracy"""
        cm = self.total_cm
        diag_part = tf.linalg.diag_part(cm)
        # Precision = TP/(TP+FP)
        precision = diag_part / (tf.reduce_sum(cm, 0) + tf.constant(1e-15))
        # Recall = Sensitivity = TP/(TP+FN)
        recall = diag_part / (tf.reduce_sum(cm, 1) + tf.constant(1e-15))
        # F1 Score = 2*Precision*Recall/(Precision+Recall)
        f1 = 2 * precision * recall / (precision + recall + tf.constant(1e-15))
        return precision, recall, f1