"""
This program is about converting text to numbers in scikitlearn.

NOTE: this is not machine learning, the ordinal encoder will not guess 
If a word not shown in the training data shows up in the testing data, it will not work.

This is a four step process.
"""

from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# Step 1: getting the data

training = np.array(["bye","hello","maybe"]).reshape([-1,1]) # the reshape converts to 2d array with a single column.

testing1 = np.array(["bye", "hello", "hello"]).reshape([-1,1]) 

testing2 = [["hello"],["bye"]]

# Step 2: choose your transformer

OE = OrdinalEncoder()

# Step 3: fitting your transformer of choice with the training data

OE.fit(training)

# Step 4: transform all the data, including the testing data

OE_training = OE.transform(training)
OE_testing1 = OE.transform(testing1)
OE_testing2 = OE.transform(testing2)

# print the results

print(OE_training)
print(OE_testing1)
print(OE_testing2)
