import pandas
import numpy
import sklearn
from sklearn import linear_model
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

data = pandas.read_csv("winequality-red.csv", sep=";")

predict = "quality"

#attributes used
data = data[["fixed acidity", "residual sugar", "density", "volatile acidity", "alcohol","quality"]]
#shuffles the data
data = sklearn.utils.shuffle(data)

x = numpy.array(data.drop([predict], 1))
y =numpy.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#best model
best = 0
for _ in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    if acc > best:
        best = acc
        print("Accuracy: " + str(acc))
        with open("winemodel.pickle", "wb") as f:
            pickle.dump(linear, f)

#loading in from pickle
pickle_in = open("winemodel.pickle", "rb")
linear = pickle.load(pickle_in)

#prints mx + b
print("Components of linear Equation")
print('Coefficient: ', linear.coef_)
print('Intercept: ', linear.intercept_)

#prediction
predicted= linear.predict(x_test)
for x in range(len(predicted)):
    print(predicted[x], x_test[x], y_test[x])

#final accuracy (not working as intended)
#print("Accuracy: " + str(linear.score(x_test, y_test)))

#plotting data
style.use("classic")
plot = "alcohol"
goal = "quality"
plt.scatter(data[plot], data[goal])
plt.xlabel(plot)
plt.ylabel(goal)
plt.show()
