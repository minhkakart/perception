## rebuild Perceptron
# import numpy
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

raw_data = pandas.read_csv('classification/cars.csv')

data = raw_data.apply(LabelEncoder().fit_transform)
data = data[['buying','maint','doors','persons','lug_boot','safety','acceptability']].values

train, test = train_test_split(data, test_size=0.3, shuffle=True)

xTrain = train[:,:6]
yTrain = train[:,6]
xTest = test[:,:6]
yTest = test[:,6]

perc = Perceptron()
perc.fit(xTrain, yTrain)
a = perc.coef_

yPredict = perc.predict(xTest)

count = 0
for i, v in enumerate(yPredict):
    if v == yTest[i]:
        count += 1

print('Du doan dung {0}/{1} = {2}'.format(count, len(yTest), count/len(yTest)))
# print(perc.coef_)