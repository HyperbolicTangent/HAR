# HAR
Using 3 different architecure to do the human activity classification, namely 1D DenseNet, LSTM, and Transformer

1. Dataset:
```
https://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions#
```
2. Dataset dir:
```
HAPT_Data_Set/
        -RawData/
        -Test/
        -Train/
        -activity_labels
        -features
        -features_info
        -README
```
3. Prepare:\
Switch to datasets.py. Set first_time_run with True value for generation of 6-channel-data and labels, z-score normalization and data augmentation using sliding window. Then set first_time_run with False value and run again to generate tfrecords files.
```
python input_pipeline/datasets.py
```
4. Training:\
Change paths of tfrecords files in config.gin to your own paths, run main.py, datasets loaded automatically from tfrecords files. The default model is Transformer.
```
python input_pipeline/load_tfrecords.py
python main.py
```
5. Evaluation:\
Set train flag to False, change model directory to the saved model in log folder in main.py
```
python main.py
```
6. Hyperparameter tuning:\
Change config.gin path to your own path, the default tuning model is TransformerS2S
```
python tune.py
```
