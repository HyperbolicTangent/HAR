import gin
import logging
import tensorflow as tf
import numpy as np
import zipfile
import os

from sklearn import preprocessing
AUTOTUNE = tf.data.experimental.AUTOTUNE
first_time_run = False


def load(name, file_path):
    if name == "HAPT Data Set":
        logging.info(f"Preparing dataset： {name}...")
        # zip file
        #zfile = zipfile.ZipFile(os.path.join(file_path, "HAPT Data Set.zip"))
        #zfile.extractall(path=os.path.join(file_path, "HAPT_Data_Set"))

        rawdata_path = os.path.join(file_path, "RawData")
        names_list = os.listdir(rawdata_path)

        # get all users data and combine them into 6-channel data
        ds = {}
        train_ds = []
        test_ds = []
        val_ds = []
        train_win_dict = {}
        test_win_dict = {}
        val_win_dict = {}
        # set precision of decimal with 18, i.e. 18 digits after decimal point are kept
        np.set_printoptions(precision=18)
        if first_time_run:
            for x in range(61):
                train_acc_path = os.path.join(rawdata_path, names_list[x])
                y = x + 61
                train_gyro_data_path = os.path.join(rawdata_path, names_list[y])
                exp_path = os.path.join(rawdata_path, "ds_exp%d.txt" % (x+1))
                # open txt file of acc data
                with open(train_acc_path, 'r') as fa:
                    # open txt file of gyro data
                    with open(train_gyro_data_path, 'r') as fb:
                        # create a new txt file and save all data into it
                        with open(exp_path, 'w') as fc:
                            for line in fa:

                                # remove line break of lines in acc data txt file
                                fc.write(line.strip('\n'))
                                fc.write(" " + fb.readline())

        else:
            for x in range(43):
                exp_path = os.path.join(rawdata_path, "ds_exp%d.txt" % (x + 1))
                data = np.loadtxt(exp_path)
                for line in data:
                    train_ds.append(line)
            # z-score normalization
            train_ds = preprocessing.scale(train_ds)
            # print(train_ds.mean(axis=0))  # get mean of every channel
            # print(train_ds.std(axis=0))  # get standard deviation of every channel
            # print(len(train_ds))
            train_ds = tf.data.Dataset.from_tensor_slices(train_ds)
            train_ds_windows = train_ds.window(size=250, shift=125, stride=1, drop_remainder=True)

            iter_index = 0
            for window in train_ds_windows:
                train_win_dict[iter_index] = np.array(list(window.as_numpy_iterator()))
                iter_index = iter_index + 1
            # print(train_win_dict[1][0])

            for x in range(43, 55):
                exp_path = os.path.join(rawdata_path, "ds_exp%d.txt" % (x + 1))
                data = np.loadtxt(exp_path)
                for line in data:
                    test_ds.append(line)
            # z-score normalization
            test_ds = preprocessing.scale(test_ds)
            test_ds = tf.data.Dataset.from_tensor_slices(test_ds)
            test_ds_windows = test_ds.window(size=250, shift=125, stride=1, drop_remainder=True)

            iter_index = 0
            for window in test_ds_windows:
                test_win_dict[iter_index] = np.array(list(window.as_numpy_iterator()))
                iter_index = iter_index + 1

            for x in range(55, 61):
                exp_path = os.path.join(rawdata_path, "ds_exp%d.txt" % (x + 1))
                data = np.loadtxt(exp_path)
                for line in data:
                    val_ds.append(line)
            # z-score normalization
            val_ds = preprocessing.scale(val_ds)
            val_ds = tf.data.Dataset.from_tensor_slices(val_ds)
            val_ds_windows = val_ds.window(size=250, shift=125, stride=1, drop_remainder=True)

            iter_index = 0
            for window in val_ds_windows:
                val_win_dict[iter_index] = np.array(list(window.as_numpy_iterator()))
                iter_index = iter_index + 1

        # generate label dataset according to labels.txt
        label_path = os.path.join(rawdata_path, "labels.txt")
        label_ds = {}
        labels = np.loadtxt(label_path).astype(int)

        for x in range(61):
            first_row = True
            list2 = []
            for num in range(len(labels)):
                a_0 = labels[num-1][4]
                a = labels[num][3] - 1
                b = labels[num][4]
                c = labels[num][2]
                if labels[num][0] == x+1:
                    if first_row:
                        for y in range(a):
                            list2.append(0)
                        for y in range(a, b):
                            list2.append(c)
                        first_row = False
                    else:
                        if a_0 == a:
                            for y in range(a, b):
                                list2.append(c)
                        else:
                            for y in range(a_0, a):
                                list2.append(0)
                            for y in range(a, b):
                                list2.append(c)
                if labels[num][0] == x+2 and labels[num-1][2] == 2:
                    a = len(list2)
                    exp_path = os.path.join(rawdata_path, "ds_exp%d.txt" % (x + 1))
                    ds[x] = np.loadtxt(exp_path)
                    b = ds[x].shape[0]
                    # print(b)
                    for y in range(a, b):
                        list2.append(0)
                    break
                if labels[num][0] == 61 and labels[num][4] == 18097:
                    a = len(list2)
                    exp_path = os.path.join(rawdata_path, "ds_exp%d.txt" % (x + 1))
                    ds[x] = np.loadtxt(exp_path)
                    b = ds[x].shape[0]
                    # print(b)
                    for y in range(a, b):
                        list2.append(0)
                    break
            # print(len(list2))
            label_ds[x] = np.array(list2)

        # generate label dataset for each train, test and validation dataset
        train_label = []
        test_label = []
        val_label = []
        for x in range(43):
            for line in label_ds[x]:
                train_label.append(line)
        train_label = tf.data.Dataset.from_tensor_slices(train_label)
        train_label_windows = train_label.window(size=250, shift=125, stride=1, drop_remainder=True)
        iter_index = 0
        train_label_win_dict = {}
        for window in train_label_windows:
            train_label_win_dict[iter_index] = list(window.as_numpy_iterator())
            iter_index = iter_index + 1
        # print(train_label_win_dict[1][0])

        for x in range(43, 55):
            for line in label_ds[x]:
                test_label.append(line)
        test_label = tf.data.Dataset.from_tensor_slices(test_label)
        test_label_windows = test_label.window(size=250, shift=125, stride=1, drop_remainder=True)
        iter_index = 0
        test_label_win_dict = {}
        for window in test_label_windows:
            test_label_win_dict[iter_index] = list(window.as_numpy_iterator())
            iter_index = iter_index + 1

        for x in range(55, 61):
            for line in label_ds[x]:
                val_label.append(line)
        val_label = tf.data.Dataset.from_tensor_slices(val_label)
        val_label_windows = val_label.window(size=250, shift=125, stride=1, drop_remainder=True)
        iter_index = 0
        val_label_win_dict = {}
        for window in val_label_windows:
            val_label_win_dict[iter_index] = list(window.as_numpy_iterator())
            iter_index = iter_index + 1

        # generate tfrecords file for every dataset
        def save_tfrecords(ds_name, data, label):  # desfile is the path where the tfrecord file is saved
            desfile = file_path + '/%s.tfrecords' % ds_name
            with tf.io.TFRecordWriter(desfile) as writer:
                for i in range(len(data)):

                    # check if label for each time instant is identical with others in the same window of label dataset
                    b = 0
                    for element in label[i]:
                        a = label[i][0]
                        if element == a and element != 0:
                            continue
                        if element != a or element == 0:
                            b = 1  # b = 1 means that we should discard this window and its labels
                            break

                    # discard window and its labels
                    if b:
                        continue

                    # convert shape of label[i] from (250,) to (1,)
                    label_num = label[i][0]

                    features = tf.train.Features(
                        feature={
                            'data': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[data[i].astype(np.float64).tostring()])),
                            'label': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[label_num]))
                        }
                    )
                    example = tf.train.Example(features=features)
                    serialized = example.SerializeToString()
                    writer.write(serialized)
            return desfile

        traintf_path = save_tfrecords('train', train_win_dict, train_label_win_dict)
        testtf_path = save_tfrecords('test', test_win_dict, test_label_win_dict)
        valtf_path = save_tfrecords('validation', val_win_dict, val_label_win_dict)

        raw_train_ds = tf.data.TFRecordDataset(traintf_path)
        raw_test_ds = tf.data.TFRecordDataset(testtf_path)
        raw_val_ds = tf.data.TFRecordDataset(valtf_path)

        def _parse_function(example_proto):
            feature_description = {'data': tf.io.FixedLenFeature((), tf.string),
                                   'label': tf.io.FixedLenFeature((), tf.int64)}
            parsed_features = tf.io.parse_single_example(example_proto, feature_description)
            data = tf.io.decode_raw(parsed_features['data'], tf.float64)
            data = tf.reshape(data, (-1, 6))
            return data, parsed_features['label']

        parsed_train_ds = raw_train_ds.map(_parse_function, num_parallel_calls=AUTOTUNE)
        parsed_val_ds = raw_val_ds.map(_parse_function, num_parallel_calls=AUTOTUNE)
        parsed_test_ds = raw_test_ds.map(_parse_function, num_parallel_calls=AUTOTUNE)

        # Visualization of some values in parsed_train_ds
        trydict = {}
        trydict2 = {}
        for index, (data, label) in enumerate(parsed_train_ds):
            trydict[index] = data
            trydict2[index] = label
        print(len(trydict[0]))
        print(trydict[1][0])
        print(trydict2[0])
        print(trydict2[1])
        print(trydict2[2])
        print(trydict2[3])
        print(trydict2[4])
        print(trydict2[5])
        print(trydict2[6])

        return parsed_train_ds, parsed_val_ds, parsed_test_ds

    else:
        print("no idrid dataset found")

if __name__ == "__main__":
    load("HAPT Data Set", r'D:\Uni Stuttgart\Deep learning lab\Human Activity Recognition\HAPT Data Set')