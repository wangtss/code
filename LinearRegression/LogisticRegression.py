import numpy as np

class LogisticRegression:
    def __init__(self, feature_dim, batch_size, lr, max_epoch):
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.lr = lr
        self.max_epoch = max_epoch
        self.weight = np.random.randn(feature_dim, 1)

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
        loss = np.sum(label * np.log(prediction))

        return loss

    def sigmoid(self, x):
        e = np.exp(np.matmul(x, self.weight))
        return e / (1 + e)

    def softmax(self, x):
        e = np.exp(np.matmul(x, self.weight))
        return e / np.sum(e)

    def train(self, x, y):
        for i in range(self.max_epoch):
            step = 0
            for batch_x, batch_y in self.getBatch(x, y):
                # forward
                output = self.softmax(batch_x)
                loss = self.calculateCrossEntropyLoss(output, batch_y)
                step += 1
                print('Epoch {}, step {}, loss {:.5f}'.format(i, step, loss))

                # backward
                grad_weight = np.matmul(np.transpose(batch_x), y - output) / self.batch_size

                # update
                self.weight += self.lr * grad_weight

    def predict(self, x):
        return self.softmax(x)

    