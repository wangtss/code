import tensorflow as tf
import numpy as np
import scipy.io as sio
import os
import glob


class DataProcessor:
    def getBlockSet(self, file_list, batch_size, shuffle=True, repeate=True):
        def parse_func(filename):
            filename = filename.decode()
            raw_data = sio.loadmat(filename)
            block, label = raw_data['block'], raw_data['label'][0][0] / 8
            block = np.divide(block - np.mean(block), np.std(block))
            return block.astype(np.float32), label.astype(np.float32)

        dataset = tf.data.Dataset.from_tensor_slices(file_list)
        dataset = dataset.map(
            lambda filename: tf.py_func(
                parse_func, [filename], [tf.float32, tf.float32]
            )
        )
        dataset = dataset.map(lambda x1, x2: {'block': x1, 'label': x2}, num_parallel_calls=batch_size)
        dataset = dataset.batch(batch_size)
        if shuffle:
            dataset = dataset.shuffle(4)
        if repeate:
            dataset = dataset.repeat()

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

if __name__ == '__main__':
    data_processor = DataProcessor()
    file_list = glob.glob(os.path.join('videoSlice', '*.mat'))
    batch = data_processor.getBlockSet(file_list, 128)
    sess = tf.Session()
    data = sess.run(batch)
    print(data['map'].shape)