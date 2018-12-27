import numpy as np
import scipy.io as sio
import os
from skimage.measure import compare_ssim as ssim
from PIL import Image
import glob
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
import tensorflow as tf
from data_processor import DataProcessor
from models import Models
import threading



def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

class Utils:
    def entropy(self, img):
        img = np.array(img, np.uint8)
        h = (np.histogram(img, 256)[0] + 1e-10) / (img.shape[0] * img.shape[1])
        return -np.sum(h * np.log2(h))

    def extractInfo(self, data_info, file_name_key='file_name', label_key='dmos_all'):
        mat_file = sio.loadmat(data_info)
        raw_file_list = mat_file[file_name_key]
        raw_dmos = mat_file[label_key]
        file_list, dmos = [], []
        for i in range(raw_file_list.shape[0]):
            file_list.append(raw_file_list[i][0][0])
            dmos.append(raw_dmos[i][0])

        dmos = np.array(dmos, dtype=np.float32)
        raw_max, raw_min = np.max(dmos), np.min(dmos)
        dmos = np.divide(dmos - raw_min, raw_max - raw_min).astype(np.float32)

        return np.array(file_list, np.str), dmos

    def readYuvVideo(self, video_name, sizes=[432, 768]):
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

            return video.astype(np.float32), frame_num

    def buildVideoBlock(self,
                        data_path,
                        data_info='LIVEVIDEOData.mat',
                        sizes=[432, 768],
                        block_size=[64, 64, 64],
                        stride=[64, 64, 64],
                        output_path='videoSlice'):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        video_name_list, dmos = self.extractInfo(data_info)
        dmos = (5 - dmos) / 5

        def getRefName(name):
            name = name.split('_')

            # LIVE
            # pre = name[0][:2] + '1'
            # name[0] = pre

            # IVP
            # if len(name) < 2:
            #     return name[0]
            # elif len(name) == 2:
            #     return '_'.join(name)
            # else:
            #     name = name[:-2]
            #     name = '_'.join(name) + '.yuv'
            #     return name

            # CSIQ
            name = name[:2]
            name.append('ref.yuv')
            return '_'.join(name)

        def getBlock(video, t, w, h, frame_num):
            if t + block_size[0] > frame_num:
                t = frame_num - block_size[0]
            if w + block_size[1] > sizes[0]:
                w = sizes[0] - block_size[1]
            if h + block_size[1] > sizes[1]:
                h = sizes[1] - block_size[2]

            block = video[t:t + block_size[0],
                          w:w + block_size[1],
                          h:h + block_size[2]]
            return block
        def computeBlockScore(ref_block, dis_block, video_score=None):
            psnr_t = psnr(np.mean(ref_block, 0), np.mean(dis_block, 0)) / 100
            psnr_w = psnr(np.mean(ref_block, 1), np.mean(dis_block, 1)) / 100
            psnr_h = psnr(np.mean(ref_block, 2), np.mean(dis_block, 2)) / 100

            ssim_t = ssim(np.mean(ref_block, 0), np.mean(dis_block, 0))
            ssim_w = ssim(np.mean(ref_block, 1), np.mean(dis_block, 1))
            ssim_h = ssim(np.mean(ref_block, 2), np.mean(dis_block, 2))

            return (psnr_t * psnr_w * psnr_h) * video_score
        def run(i):
            video_name = video_name_list[i]
            dis_name = os.path.join(data_path, video_name)
            ref_name = os.path.join(data_path, getRefName(video_name))
            print(dis_name, ref_name)

            dis_video, dis_frame_num = self.readYuvVideo(dis_name, sizes)
            ref_video, ref_frame_num = self.readYuvVideo(ref_name, sizes)
            frame_num = min(dis_frame_num, ref_frame_num)
            count = 0
            for t in range(0, frame_num, stride[0]):
                for w in range(0, sizes[0], stride[1]):
                    for h in range(0, sizes[1], stride[2]):
                        dis_block = getBlock(dis_video, t, w, h, frame_num)
                        ref_block = getBlock(ref_video, t, w, h, frame_num)
                        block_score = computeBlockScore(ref_block, dis_block, dmos[i])
                        save_name = '{}_{}_{}_{}.mat'.format(video_name.split('.')[0], t, w, h)
                        sio.savemat(
                            os.path.join(output_path, save_name),
                            {'label': block_score}
                        )
                        count += 1
            # print(os.path.basename(dis_name))

        threads = []
        max_thread_num = 4

        for i in range(0, len(video_name_list)):
            threads.append(threading.Thread(target=run, args=(i,)))
            threads[-1].start()
            if len(threads) >= max_thread_num:
                for t in threads:
                    t.join()
                threads.clear()

    def inference(self, file_list, checkpoint='model/92400-model'):
        data_processor = DataProcessor()
        data = data_processor.getBlockSet(file_list, 64, shuffle=False, repeate=False)

        # build graph
        model = Models()
        input_placeholder = tf.placeholder(tf.float32, (None, 64, 64, 64))
        phase_placeholder = tf.placeholder(tf.bool, name='phase')
        output = model.createSTNet(input_placeholder, phase_placeholder)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            tf.summary.FileWriter(logdir='inference', graph=sess.graph)
            predictions = []
            try:
                saver.restore(sess, checkpoint)
                while True:
                    batch = sess.run(data)
                    predictions.append(sess.run(output, feed_dict={
                        input_placeholder: batch['block'], phase_placeholder:False
                    }))
            except tf.errors.OutOfRangeError:
                return np.concatenate(predictions, 0)

    def getVideoScore(self, block_path='videoSlice', data_info='LIVEVIDEOData.mat'):
        video_name_list, dmos = self.extractInfo(data_info)

        def calculateMean(block_list):
            score, num = 0, len(block_list)
            for block in block_list:
                raw = sio.loadmat(block)
                score += raw['label'][0][0]
            return score / num

        def calculatePredMean(block_list):
            score = self.inference(block_list)

            return np.mean(score)

        def getBlockList(video_name):
            pre = video_name.split('.')[0]
            block_list = glob.glob(os.path.join(block_path, '{}*.mat'.format(pre)))
            checked_list = []
            for block_name in block_list:
                name = os.path.basename(block_name).split('_')[:-3]
                name = '_'.join(name)
                if name == pre:
                    checked_list.append(block_name)

            return checked_list

        def saveVideoScore():
            video_score = []
            for i, video_name in enumerate(video_name_list):
                video_score.append(calculateMean(getBlockList(video_name)))
                print('Calculate {}'.format(i), end='\r')
            print('\nDone!')
            video_score = np.array(video_score, np.float32)
            print(spearmanr(dmos, video_score)[0])
            sio.savemat('csiq_pred.mat', {'video_score': video_score})

        def evaluate():
            video_score = sio.loadmat('pred.mat')
            video_score = video_score['video_score']
            label = np.reshape(dmos, [-1])
            srcc = []
            for i in range(100):
                index = np.arange(0, 160, 1, dtype=np.uint8)
                np.random.shuffle(index)
                train_index, test_index = index[:128], index[128:]
                train_data, train_label = video_score[train_index], label[train_index]
                test_data, test_label = video_score[test_index], label[test_index]

                svr = SGDRegressor(max_iter=100000)
                svr.fit(train_data, train_label)
                pred = svr.predict(test_data)
                cur = abs(spearmanr(test_label, pred)[0])
                srcc.append(cur)
                print(cur)
            print(np.median(srcc))

        saveVideoScore()
        # evaluate()
if __name__ == '__main__':
    utils = Utils()
    data_path = r'/home/wts/database/CSIQVideo'
    # data_path = r'/home/wts/database/CSIQVideo'
    data_info = r'CSIQData.mat'
    sizes = [480, 832]
    output_path = 'CSIQVB'

    utils.buildVideoBlock(data_path=data_path,
                          data_info=data_info,
                          sizes=sizes,
                          output_path=output_path)
    utils.getVideoScore(block_path=output_path, data_info=data_info)

