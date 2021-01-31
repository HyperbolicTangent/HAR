import gin
import tensorflow as tf
import logging
import datetime


@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, run_paths, total_steps, log_interval, ckpt_interval):
        # Summary Writer
        self.train_summary_writer = tf.summary.create_file_writer
        self.valid_summary_writer = tf.summary.create_file_writer

        # Checkpoint Manager
        self.ckpt = tf.train.Checkpoint
        self.manager = tf.train.CheckpointManager

        # Loss objective
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.001, 3000, 0.9)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval

    @tf.function
    def train_step(self, signal, labels):
        labels = labels - 1    # change label from [1,12] to [0,11]
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(signal, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, signal, labels):
        labels = labels - 1
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(signal, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def train(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + current_time + '/train'
        valid_log_dir = 'logs/' + current_time + '/valid'
        train_summary_writer = self.train_summary_writer(train_log_dir)
        valid_summary_writer = self.valid_summary_writer(valid_log_dir)
        ckpt = self.ckpt(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
        manager = self.manager(ckpt, self.run_paths["path_ckpts_train"], max_to_keep=10)
        tf.profiler.experimental.start('logs/'+ current_time)

        for idx, (signal, labels) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(signal, labels)

            # Profiler of first 20 step
            if step == 20:
                tf.profiler.experimental.stop()

            if step % self.log_interval == 0:

                # Reset test metrics
                self.test_loss.reset_states()
                self.test_accuracy.reset_states()
                i = 0
                for test_signal, test_labels in self.ds_val:
                    i += 1
                    self.test_step(test_signal, test_labels)

                template = 'Step {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.test_loss.result(),
                                             self.test_accuracy.result() * 100))

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                # Write summary to tensorboard
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss.result(), step=step)
                    tf.summary.scalar('accuracy', self.train_accuracy.result(), step=step)
                with valid_summary_writer.as_default():
                    tf.summary.scalar('loss', self.test_loss.result(), step=step)
                    tf.summary.scalar('accuracy', self.test_accuracy.result(), step=step)

                yield self.test_accuracy.result().numpy()

            if step % self.ckpt_interval == 0:
                logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                # Save checkpoint
                manager.save()

            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                # Save final checkpoint
                manager.save()
                return self.test_accuracy.result().numpy()
