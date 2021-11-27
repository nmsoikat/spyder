import numpy as np
from sklearn import preprocessing

input_data = np.array([
    [1.55,2.22,3.99],
    [4,5,6],
    [7,8,9]
    ])

#BINARIZATION:
output = preprocessing.Binarizer(threshold=(2)).transform(input_data)


#MIN MAX SCALING:
min_max_scale = preprocessing.MinMaxScaler(feature_range=(2,8))
data = min_max_scale.fit_transform(input_data)


#NORMALIZATION:
first_form = preprocessing.normalize(input_data, norm='l1')
second_form = preprocessing.normalize(input_data, norm='l2')

print("1st form:", first_form)
print("2nd form:", second_form)
