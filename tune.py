import logging
import gin
from ray import tune
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB

from input_pipeline.load_tfrecords import load_from_tfrecords
from models.LSTM import LSTM
from models.TransformerS2L import TransformerS2L
from models.Dense import DenseNet
from models.TransformerS2S import TransformerS2S
from train import Trainer
from utils import utils_params, utils_misc


def train_func(config):
    # Hyperparameters
    bindings = []
    for key, value in config.items():
        bindings.append(f'{key}={value}')

    # generate folder structures
    run_paths = utils_params.gen_run_folder(','.join(bindings[2]))

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], bindings)
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test = load_from_tfrecords()

    # model
    model = TransformerS2S()

    trainer = Trainer(model, ds_train, ds_val, run_paths)
    for val_accuracy in trainer.train():
        tune.report(val_accuracy=val_accuracy)


algo = TuneBOHB(max_concurrent=4, metric="val_accuracy", mode="max")
bohb = HyperBandForBOHB(
    time_attr="training_iteration",
    metric="val_accuracy",
    mode="max",
    max_t=100)

config_name = 'transformerS2S'

if config_name == 'lstm':
    config = {
      "LSTM.rnn_units1": tune.randint(16, 64),
      "LSTM.rnn_units2": tune.randint(8, 32),
      "LSTM.dense_units": tune.randint(12, 64),
      "LSTM.dropout_rate": tune.uniform(0, 0.8),
      "load_from_tfrecords.batch_size": tune.choice([8, 16, 32, 64, 128])
    }

elif config_name == 'dense':
    config = {
      'DenseNet.nb_dense_block': tune.randint(2, 5),
      'DenseNet.nb_layers': tune.randint(2, 4),
      'DenseNet.growth_rate': tune.choice([2, 4, 8, 16, 32]),
      'DenseNet.nb_filter': tune.randint(8, 64),
      'DenseNet.reduction': tune.uniform(0, 0.6),
      'DenseNet.dropout_rate': tune.uniform(0, 0.8)
    }

elif config_name == 'transformerS2L':
    config = {
      'TransformerS2L.ff_dim': tune.choice([32, 64, 128, 256, 512]),
      'TransformerS2L.num_layer': tune.randint(1, 6),
      'TransformerS2L.num_heads': tune.randint(4, 64),
      'TransformerS2L.dropout_rate': tune.uniform(0, 0.8),
      'TransformerS2L.dense_units': tune.randint(16, 128)
    }

elif config_name == 'transformerS2S':
    config = {
    'TransformerS2S.num_layers': tune.randint(1, 5),
    'TransformerS2S.d_model': tune.choice([32, 64, 128]),
    'TransformerS2S.num_heads': tune.choice([8, 16, 32]),
    'TransformerS2S.dff': tune.choice([64, 128, 256]),
    'TransformerS2S.kernel_size': tune.choice([1, 3, 5, 7]),
    'TransformerS2S.rate': tune.uniform(0, 0.8)
    }



analysis = tune.run(
    train_func, scheduler=bohb, search_alg=algo, num_samples=100, resources_per_trial={'gpu': 1, 'cpu': 2},
    checkpoint_at_end=True, checkpoint_score_attr='val_accuracy',
    config=config)

print("Best config: ", analysis.get_best_config(metric="val_accuracy", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()

logdir = analysis.get_best_logdir("val_accuracy", mode="max")
print(logdir)