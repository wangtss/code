import tensorflow as tf
import numpy as np
from models import Models


class Inference:
    def __init__(self):
        self.bs = [64, 64, 64]
        self.stride = [64, 64, 64]
        self.checkpoint = 'model/92400-model'
        
    def readYuvVideo(self, video_name, sizes):
        frame_size = sizes[0] * sizes[1]
        color_frame_size = frame_size // 2

        with open(video_name, 'rb') as f:
            f.seek(0, 2)
            frame_num = int(f.tell() / (frame_size * 1.5))

            f.seek(0, 0)
            video = np.zeros((frame_num, sizes[0], sizes[1]), np.uint8)
            for t in range(frame_num):
                data = np.frombuffer(f.read(sizes[0] * sizes[1]), np.uint8)
                video[t, :, :] = np.reshape(data, sizes)
                f.read(color_frame_size)

            self.video = video.astype(np.float32)
            self.frame_num = frame_num

    def splitVideo(self, video_name, sizes):
        self.readYuvVideo(video_name, sizes)

        def getBlock(video, t, w, h, frame_num):
            if t + self.bs[0] > frame_num:
                t = frame_num - self.bs[0]
            if w + self.bs[1] > sizes[0]:
                w = sizes[0] - self.bs[1]
            if h + self.bs[1] > sizes[1]:
                h = sizes[1] - self.bs[2]

            block = video[t:t + self.bs[0],
                          w:w + self.bs[1],
                          h:h + self.bs[2]]
            return block

        self.vbs, count = [], 0
        for t in range(0, self.frame_num, self.stride[0]):
            for w in range(0, sizes[0], self.stride[1]):
                for h in range(0, sizes[1], self.stride[2]):
                    self.vbs.append(getBlock(self.video, t, w, h, self.frame_num))
            count += 1
        self.block_num = len(self.vbs)
        self.vbs = np.reshape(
            np.array(self.vbs, np.float32),
            [count, -1, self.bs[0], self.bs[1], self.bs[2]]
        )

    def inference(self):
        model = Models()
        input_placeholder = tf.placeholder(tf.float32, (None, 64, 64, 64))
        phase_placeholder = tf.placeholder(tf.bool, name='phase')
        output = model.createSTNet(input_placeholder, phase_placeholder)
        saver = tf.train.Saver()
        self.display_score = []

        with tf.Session() as sess:
            tf.summary.FileWriter(logdir='inference', graph=sess.graph)
            saver.restore(sess, self.checkpoint)
            for i in range(self.vbs.shape[0]):
                predictions = []
                print(i)
                batch = self.vbs[i]
                for j in range(0, batch.shape[0] - 1, 2):
                    mini_batch = batch[j:j + 2]
                    predictions.append(sess.run(output, feed_dict={
                        input_placeholder: mini_batch, phase_placeholder: False
                    }))
                self.display_score.append(np.mean(np.reshape(predictions, (-1,))))

    def processing(self, video_name, sizes):
        self.splitVideo(video_name, sizes)
        self.inference()
