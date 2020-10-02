from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

clf = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers
# 1
mlp = MLPClassifier()
# 2
kn = KNeighborsClassifier()
# 3
gpc = GaussianProcessClassifier()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)
mlp = mlp.fit(X, Y)
kn = kn.fit(X, Y)
gpc = gpc.fit(X, Y)

prediction = clf.predict([[190, 70, 39]])
predictionMlp = mlp.predict([[190, 70, 39]])
predictionKn = kn.predict([[190, 70, 39]])
predictionGpc = gpc.predict([[190, 70, 39]])

# CHALLENGE compare their reusults and print the best one!

print(prediction)
print(predictionMlp)
print(predictionKn)
print(predictionGpc)
