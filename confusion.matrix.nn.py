import os
import sys
import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.layers import Dropout

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

model = VGG19(weights="imagenet", include_top=False, input_shape=(255,255,3))

#Directory for a source dataset
indir = sys.argv[1]

#Load images
l5 = model.get_layer("block5_pool")
l5ga = GlobalAveragePooling2D()(l5.output)
l5_model = Model(model.inputs, l5ga)

gen = image.ImageDataGenerator(preprocessing_function=preprocess_input) #Don't forget to include this preprocessing function for the VGG models...!
img_gen = gen.flow_from_directory(indir, target_size=(255,255), batch_size=10, class_mode="sparse", shuffle=False)
vgg_features =l5_model.predict(img_gen)

img_data = np.concatenate([img_gen.next()[0] for i in range(0, img_gen.samples, 10)])

img_names = [os.path.basename(f) for f in img_gen.filenames]
clab = np.concatenate([img_gen.next()[1] for i in range(0, img_gen.samples, 10)])

img_names = [os.path.basename(f) for f in img_gen.filenames]
img_names2 = ((os.path.basename(f), os.path.dirname(f)) for f in img_gen.filenames)
img_class = dict(img_names2)

class_id = {}
id = 0
for n in [os.path.dirname(f) for f in img_gen.filenames]:
    if not n in class_id:
        class_id[n] = id
        id += 1
print (class_id)
id_class = {v:k for k, v in class_id.items()}

##Directory for a target dataset
indir2 = sys.argv[2]

img_gen2 = gen.flow_from_directory(indir2, target_size=(255,255), batch_size=10, class_mode="sparse", shuffle=False)
img_data2 = np.concatenate([img_gen2.next()[0] for i in range(0, img_gen2.samples, 10)])
vgg_features2 =l5_model.predict(img_gen2)

img_names2 = [os.path.basename(f) for f in img_gen2.filenames]
clab2 = np.concatenate([img_gen2.next()[1] for i in range(0, img_gen2.samples, 10)])

size = int(sys.argv[3])

#Training
clab_img = np.stack((clab, np.array(img_names))).T
train_x, test_x, train_y, test_y = train_test_split(vgg_features, clab_img, test_size=200, train_size=size)

train_y = train_y[:,0].astype(float)
test_img_nam = test_y[:,1]
test_y = test_y[:,0].astype(float)

# train_x, test_x, train_y, test_y = train_test_split(vgg_features, clab, test_size=200, train_size=size)

nclass = 12
input = Input(shape=l5_model.output.shape[1:])
out = Dense(512, activation="relu")(input)
out = Dropout(0.6)(out)
out = Dense(256, activation="relu")(out)
out = Dropout(0.6)(out)
out = Dense(nclass, activation="softmax")(out)

nn = Model(input, out)
nn.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
nn.fit(train_x, train_y, epochs=500, verbose=1, validation_data=(test_x, test_y))

#Output predictions on test sets
pred = nn.predict(test_x)
pred_class = np.argmax(pred, axis=1)
suc = np.argmax(pred, axis=1) == test_y
nam = [id_class[i] for i in test_y]

with open("pred.{0}.N{1}.txt".format(os.path.basename(indir), size), "w") as f:
    f.write("FileName\tTaxa\tTaxaID\tSuccess\t" + "\t".join(class_id.keys()) + "\n" )
    for i, row in enumerate(pred):
        #print("{0}\t{1}\t{2}\t{3}".format(test_img_nam[i], nam[i], int(test_y[i]), suc[i]) + "\t".join(["{0:5f}".format(i) for i in row]) + "\n")
        f.write("{0}\t{1}\t{2}\t{3}\t".format(test_img_nam[i],nam[i], int(test_y[i]), suc[i]) + "\t".join(["{0:5f}".format(i) for i in row]) + "\n")

print ("source confusion matrix:")
print ("-- true \n|\npredicted ")
print (confusion_matrix(test_y, pred_class).T)

#Prediction on a target dataset
pred2 = nn.predict(vgg_features2)
pred_class2 = np.argmax(pred2, axis=1)
suc2 = np.argmax(pred2, axis=1) == clab2
nam2 = [id_class[i] for i in clab2]
img_nam2 = np.array(img_names2)

with open("pred.{0}-{1}.N{2}.txt".format(os.path.basename(indir), os.path.basename(indir2), size), "w") as f:
    f.write("FileName\tTaxa\tTaxaID\tSuccess\t" + "\t".join(class_id.keys()) + "\n" )
    for i, row in enumerate(pred2):
        #print("{0}\t{1}\t{2}\t{3}".format(test_img_nam[i], nam[i], int(test_y[i]), suc[i]) + "\t".join(["{0:5f}".format(i) for i in row]) + "\n")
        f.write("{0}\t{1}\t{2}\t{3}\t".format(img_nam2[i],nam2[i], int(clab2[i]), suc2[i]) + "\t".join(["{0:5f}".format(i) for i in row]) + "\n")

print ("source -> target confusion matrix")
print ("-- true \n|\npredicted ")
print (confusion_matrix(clab2, pred_class2).T)
