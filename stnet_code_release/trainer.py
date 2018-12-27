import tensorflow as tf
from data_processor import DataProcessor
from models import Models
import json
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
import os
import random
import glob
import argparse


class Trainer:
    def __init__(self, block_path, batch_size=256, lr=1e-5, log_path='log', model_path='model', old_checkpoint=None):
        self.batch_size = batch_size
        self.lr = lr
        self.reg_rate = 1e-8
        self.test_interval = 10
        self.save_interval = 100
        self.model_path = model_path
        self.sess = tf.Session()
        self.global_step = 0
        self.train_batch, self.test_batch = self.get_train_and_val_iter(block_path)
        self.models = Models(reg=self.reg_rate)
        self.build_graph()
        self.summary = tf.summary.FileWriter(logdir=log_path, graph=self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=100)
        if old_checkpoint:
            self.global_step = int(os.path.basename(old_checkpoint).split('-')[0])
            self.saver.restore(self.sess, old_checkpoint)
        else:
            self.sess.run(tf.global_variables_initializer())

    def get_train_and_val_iter(self, path):
        block_list = glob.glob(os.path.join(path, '*.mat'))
        random.shuffle(block_list)
        train_list = block_list[:int(round(0.9 * len(block_list)))]
        test_list = block_list[int(round(0.9 * len(block_list))):]
        data_processor = DataProcessor()
        train_batch = data_processor.getBlockSet(train_list, self.batch_size, True, True)
        test_batch = data_processor.getBlockSet(test_list, self.batch_size, True, True)
        return train_batch, test_batch

    def calculateLoss(self, predictions, label):
        loss = tf.losses.mean_squared_error(label, predictions)
        reg_loss = tf.losses.get_regularization_loss()

        return loss + reg_loss

    def build_graph(self):
        input_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None, 64, 64, 64)
        )
        label_placeholder = tf.placeholder(dtype=tf.float32, shape=(None,))
        phase_placeholder = tf.placeholder(dtype=tf.bool, name='phase')

        model_output = self.models.createSTNet(input_placeholder, phase=phase_placeholder)

        loss = self.calculateLoss(model_output, label_placeholder)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

        train_loss_log = tf.summary.scalar('TRAIN_LOSS', loss)
        test_loss_log = tf.summary.scalar('TEST_LOSS', loss)

        self.sess.run(tf.global_variables_initializer())

        tf.add_to_collection('input', input_placeholder)
        tf.add_to_collection('label', label_placeholder)
        tf.add_to_collection('phase', phase_placeholder)
        tf.add_to_collection('output', model_output)
        tf.add_to_collection('loss', loss)
        tf.add_to_collection('train_loss_log', train_loss_log)
        tf.add_to_collection('test_loss_log', test_loss_log)
        tf.add_to_collection('train_step', train_step)

    def calculate_fit_score(self, label, prediction):
        srcc = spearmanr(label, prediction)[0]
        return srcc

    def create_summary_from_numpy(self, value, name):
        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=value)
        self.summary.add_summary(summary, self.global_step)

    def run(self):
        input_op = tf.get_collection('input')[0]
        label_op = tf.get_collection('label')[0]
        phase_op = tf.get_collection('phase')[0]
        output_op = tf.get_collection('output')[0]
        loss_op = tf.get_collection('loss')[0]
        train_step = tf.get_collection('train_step')[0]
        train_loss_log_op = tf.get_collection('train_loss_log')[0]
        test_loss_log_op = tf.get_collection('test_loss_log')[0]

        def train_once():
            batch = self.sess.run(self.train_batch)
            _, output, loss, loss_log = self.sess.run(
                [train_step, output_op, loss_op, train_loss_log_op],
                feed_dict={input_op: batch['block'], label_op: batch['label'], phase_op: True}
            )
            self.summary.add_summary(loss_log, self.global_step)
            # fit_score = self.calculate_fit_score(batch['label'], output)
            # self.create_summary_from_numpy(fit_score, 'TRAIN_SRCC')
            self.summary.flush()
            # print(output)
            print('step {}, loss {:.5f}'.format(self.global_step, loss))

        def test_once():
            batch = self.sess.run(self.test_batch)
            output, loss, loss_log = self.sess.run(
                [output_op, loss_op, test_loss_log_op],
                feed_dict={input_op: batch['block'], label_op: batch['label'], phase_op: False}
            )
            self.summary.add_summary(loss_log, self.global_step)
            # fit_score = self.calculate_fit_score(batch['label'], output)
            # self.create_summary_from_numpy(fit_score, 'TEST_SRCC')
            self.summary.flush()

            print('===========================================')
            print('validation loss {:.5f}'.format(loss))
            print('===========================================')

        while True:
            train_once()
            self.global_step += 1

            if self.global_step % self.test_interval == 0:
                test_once()

            if self.global_step % self.save_interval == 0:
                self.saver.save(self.sess, os.path.join(self.model_path, '{}-model'.format(self.global_step)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--block_path', type=str, default='videoBlock')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--old_checkpoint', type=str, default=None)
    args = parser.parse_args()
    trainer = Trainer(
        block_path=args.block_path,
        batch_size=args.batch_size,
        lr=args.lr,
        old_checkpoint=args.old_checkpoint
    )
    trainer.run()