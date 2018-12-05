import numpy as np

class LogisticRegression:
    def __init__(self, feature_dim, class_num, batch_size, lr, max_epoch):
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.batch_size = batch_size
        self.lr = lr
        self.max_epoch = max_epoch
        self.weight = np.random.randn(feature_dim, class_num)

    def getBatch(self, x, y, shuffle=True):
        # generate batch
        sample_num = x.shape[0]
        index = np.arange(0, sample_num, 1)
        if shuffle:
            np.random.shuffle(index)
        for i in range(0, sample_num, self.batch_size):
            start = i if i + self.batch_size < sample_num else sample_num - self.batch_size
            batch_x = x[index[start:start + self.batch_size]]
            batch_y = y[index[start:start + self.batch_size]]
            yield batch_x, batch_y

    def calculateCrossEntropyLoss(self, prediction, label):
        loss = -np.sum(label * np.log(prediction)) / self.batch_size

        return loss

    def calculateAccuracy(self, predictions, label):
        """accuracy for one-hot label"""
        pred_index = np.argmax(predictions, axis=1)
        true_index = np.argmax(label, axis=1)

        return np.sum(pred_index == true_index) / pred_index.shape[0]

    def sigmoid(self, x):
        e = np.exp(np.matmul(x, self.weight))
        return e / (1 + e)

    def softmax(self, x):
        e = np.exp(np.matmul(x, self.weight))
        return np.divide(e, np.expand_dims(np.sum(e, axis=1), 1))

    def train(self, x, y):
        for i in range(self.max_epoch):
            step = 0
            for batch_x, batch_y in self.getBatch(x, y):
                # forward
                output = self.softmax(batch_x)
                loss = self.calculateCrossEntropyLoss(output, batch_y)
                acc = self.calculateAccuracy(output, batch_y)
                step += 1
                print('Epoch {}, step {}, loss {:.5f}, acc {:.3f}'.format(i, step, loss, acc))

                # backward
                grad_weight = np.matmul(np.transpose(batch_x), batch_y - output) / self.batch_size

                # update
                self.weight += self.lr * grad_weight

    def predict(self, x):
        return self.softmax(x)

