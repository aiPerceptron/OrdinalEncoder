from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# getting the data

training = np.array(["bye","hello","maybe"]).reshape([-1,1]) # the reshape converts to 2d array with a single column.

testing1 = np.array(["bye", "hello", "hello"]).reshape([-1,1]) 

testing2 = [["hello"],["bye"]]

# choose your transformer

OE = OrdinalEncoder()

# fitting your transformer of choice with the training data

OE.fit(training)

# transform all the data, including the testing data

OE_training = OE.transform(training)
OE_testing1 = OE.transform(testing1)
OE_testing2 = OE.transform(testing2)

# print the results

print(OE_training)
print(OE_testing1)
print(OE_testing2)
