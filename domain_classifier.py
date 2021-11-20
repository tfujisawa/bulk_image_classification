import os
import sys
import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

import cv2 as cv

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit

model = VGG19(weights="imagenet", include_top=False, input_shape=(255,255,3))

dir1 = sys.argv[1]
dir2 = sys.argv[2]

#Average pooling of the 5th layer of VGG19
l5 = model.get_layer("block5_pool")
l5ga = GlobalAveragePooling2D()(l5.output)
l5_model = Model(model.inputs, l5ga)

#Read file and extract features
gen = image.ImageDataGenerator(preprocessing_function=preprocess_input) #Don't forget to include this preprocessing function for the VGG models...!
img_gen = gen.flow_from_directory(dir1, target_size=(255,255), batch_size=10, class_mode="sparse", shuffle=False)
vgg_features1 =l5_model.predict(img_gen)

img_gen2 = gen.flow_from_directory(dir2, target_size=(255,255), batch_size=10, class_mode="sparse", shuffle=False)
img_data2 = np.concatenate([img_gen2.next()[0] for i in range(0, img_gen2.samples, 10)])
vgg_features2 = l5_model.predict(img_gen2)

#Random selection
size = 100

n1 = vgg_features1.shape[0]
ind = np.zeros(n1, dtype=np.bool)
ind[np.random.choice(range(n1), size=size, replace=False)] = True
train1 = vgg_features1[ind]

tind = np.random.choice(np.where(np.logical_not(ind))[0], size=size, replace=False)
tind.sort()
test1 = vgg_features1[tind]

n2 = vgg_features2.shape[0]
ind = np.zeros(n2, dtype=np.bool)
ind[np.random.choice(range(n2), size=size, replace=False)] = True
train2 = vgg_features2[ind]

tind = np.random.choice(np.where(np.logical_not(ind))[0], size=size, replace=False)
tind.sort()
test2 = vgg_features2[tind]

##Training
train = np.concatenate((train1, train2))
dlab_tr = np.concatenate((np.zeros(size), np.ones(size)))

svc = LinearSVC(C=1.0, loss="squared_hinge", penalty="l2")
svc.fit(train, dlab_tr)
svc.predict(train)
print ("domain classification accuracy")
print ("train:", np.sum(svc.predict(train) == dlab_tr)/len(dlab_tr))

#Test
test = np.concatenate((test1, test2))
dlab_tes = np.concatenate((np.zeros(size), np.ones(size)))

svc.predict(test)
acc = np.sum(svc.predict(test) == dlab_tes)/len(dlab_tes)
print ("test:", acc, "(da: {0:.3f})".format(2*(1-2*(1-acc))), )

##Five-fold CV
suc = []
kf = KFold(len(train))
for train_i, test_i in kf.split(train, dlab_tr):
    svc.fit(train[train_i], dlab_tr[train_i])
    suc.append(svc.predict(train[test_i])==dlab_tr[test_i])

suc = (np.array(suc)).T
suc = suc[0]
acc = np.sum(suc)/len(suc)
print ("5Fold-CV:", acc, "(da: {0:.3f})".format(2*(1-2*(1-acc))))
