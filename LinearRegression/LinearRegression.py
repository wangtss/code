import numpy as np

class LinearRegression:
    def __init__(self, feature_dim, batch_size, lr, max_epoch):
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.lr = lr
        self.max_epoch = max_epoch
        self.weight = np.random.randn(feature_dim, 1)
        self.bias = 0

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

    def calculateLoss(self, predication, label, requireGrad=True):
        loss = np.mean(np.square(label - predication))
        if requireGrad:
            grad_p = 2 * (label - predication)
            return loss, grad_p
        return loss

    def train(self, x, y):
        for i in range(self.max_epoch):
            step = 0
            for batch_x, batch_y in self.getBatch(x, y):
                # forward
                output = np.matmul(batch_x, self.weight) + self.bias
                loss, grad_output = self.calculateLoss(output, batch_y)
                step += 1
                print('Epoch {}, step {}, loss {:.5f}'.format(i, step, loss))

                # backward
                grad_weight = np.matmul(np.transpose(batch_x), grad_output) / self.batch_size
                grad_bias = np.sum(grad_output) / self.batch_size

                # update
                self.weight += self.lr * grad_weight
                self.bias += self.lr * grad_bias

    def predict(self, x):
        return np.matmul(x, self.weight) + self.bias


if __name__ == '__main__':
    x = np.random.rand(100000, 10)
    weight = np.random.rand(10, 1)
    y = np.matmul(x, weight) + 1
    model = LinearRegression(feature_dim=x.shape[1], batch_size=512, lr=0.1, max_epoch=1000)
    model.train(x, y)






