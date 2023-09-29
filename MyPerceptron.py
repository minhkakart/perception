from random import randint
import numpy

class MyPerceptron:
    def __init__(self, xTrain, yTrain) -> None:
        self.lenXY = len(xTrain)
        if self.lenXY != len(yTrain):
            raise ValueError('Parameters is not valid!')
        self.X = numpy.array(xTrain)
        self.Y = numpy.array([list(map(lambda x: 1 if x == 1 else -1, yTrain))]).T
        self.w = []
        for i in range(len(self.X[0])):
            self.w.append(randint(int(-len(self.X[0])), int(len(self.X[0]))))
        self.w = numpy.array([self.w])
    
    def check(self, x, y):
        # print('check')
        # print(numpy.array([x]).T@self.w)
        a = numpy.array(x)
        res = self.w@a
        return True if res*y >= 0 else False

    def stop(self):
        # print('stop')
        # print(enumerate(self.X))
        for i, v in enumerate(self.X):
            if not self.check(v, self.Y[i]):
                return False, i
        return True, -1

    def fit(self, eta = 0.001):
        count = 0
        while count < 1000:
            check, index = self.stop()
            if check:
                return
            tmp = eta*(numpy.array([self.Y[index]])*numpy.array([self.X[index]]))
            self.w = self.w + tmp
            count += 1
        print(count)
    def predict(self, xTest):
        test = numpy.array(xTest)
        res = (test@self.w.T).T[0]
        return list(map(lambda x: 0 if x < 0 else 1, res))