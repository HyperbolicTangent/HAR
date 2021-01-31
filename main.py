import gin
import logging
import tensorflow as tf
from models.TransformerS2L import TransformerS2L
from models.LSTM import LSTM
from models.Dense import DenseNet
from models.TransformerS2S import TransformerS2S
from absl import app, flags
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import load_tfrecords
from utils import utils_params, utils_misc


FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')


def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test = load_tfrecords.load_from_tfrecords()

    # print number of available GPUs
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    if FLAGS.train:
        model = TransformerS2S()
        model.build((None, 250, 6))
        model.summary()
        trainer = Trainer(model, ds_train, ds_val, run_paths)
        for _ in trainer.train():
            continue

    else:
        # get one completely trained model to do evaluating
        opt = tf.keras.optimizers.Adam()
        model = TransformerS2S()
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(1), optimizer=opt, net=model)

        # change ckpt dir to load the ckpt you want
        manager = tf.train.CheckpointManager(ckpt,
                                             "/content/drive/MyDrive/experiments/run_2021-01-24T13-52-22-787253/ckpts",
                                             max_to_keep=3)
        ckpt.restore(manager.latest_checkpoint)
        print("Restored from {}".format(manager.latest_checkpoint))
        evaluate(model, ds_test)


if __name__ == "__main__":
    app.run(main)