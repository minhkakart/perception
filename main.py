# import numpy
import pandas
from MyPerceptron import MyPerceptron
# from sklearn.linear_model import Perceptron
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

try:
    raw_data = pandas.read_csv('classification/cars.csv')
except:
    raw_data = pandas.read_csv('cars.csv')

data = raw_data.apply(LabelEncoder().fit_transform)
data = data[['buying','maint','doors','persons','lug_boot','safety','acceptability']].values

train, test = train_test_split(data, test_size=0.3, shuffle=False)

xTrain = train[:,:6]
yTrain = list(train[:,6])
xTest = test[:,:6]
yTest = list(test[:,6])

a = MyPerceptron(xTrain, yTrain)
a.fit(eta=0.01)
yPredict = a.predict(xTest)
count = 0
for i, v in enumerate(yPredict):
    if v == yTest[i]:
        count += 1

print('Du doan dung {0}/{1} = {2}'.format(count, len(yTest), count/len(yTest)))