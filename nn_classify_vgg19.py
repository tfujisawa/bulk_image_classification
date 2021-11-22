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

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split


if len(sys.argv) > 3:
    size_min, size_max, size_step = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
else:
    size_min, size_max, size_step = 100, 700, 100

model = VGG19(weights="imagenet", include_top=False, input_shape=(255,255,3))

##Directory for a source dataset
dir = sys.argv[1]

##Feature extraction by VGG19-Conv5 model
l5 = model.get_layer("block5_pool")
l5ga = GlobalAveragePooling2D()(l5.output)
l5_model = Model(model.inputs, l5ga)

gen = image.ImageDataGenerator(preprocessing_function=preprocess_input) #Don't forget to include this preprocessing function for the VGG models...!
img_gen = gen.flow_from_directory(dir, target_size=(255,255), batch_size=10, class_mode="sparse", shuffle=False)
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
dir2 = sys.argv[2]

img_gen2 = gen.flow_from_directory(dir2, target_size=(255,255), batch_size=10, class_mode="sparse", shuffle=False)
img_data2 = np.concatenate([img_gen2.next()[0] for i in range(0, img_gen2.samples, 10)])
vgg_features2 =l5_model.predict(img_gen2)

img_names2 = [os.path.basename(f) for f in img_gen2.filenames]
clab2 = np.concatenate([img_gen2.next()[1] for i in range(0, img_gen2.samples, 10)])

res = []
for size in range(size_min, size_max+1, size_step):
    for k in range(1):
        ind = np.zeros(len(clab), dtype=np.bool)

        train_x, test_x, train_y, test_y = train_test_split(vgg_features, clab, test_size=200, train_size=size)

        ##5th convolution layer -> 2 dense layers + last classification layer
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

        print(sum(np.argmax(nn.predict(test_x), axis=1) == test_y)/len(test_y))
        acc1 = sum(np.argmax(nn.predict(test_x), axis=1) == test_y)/len(test_y)

        ##Cf. SVM with RBF
        # svc = SVC(kernel="rbf", gamma="scale", C=100)
        # svc.fit(train_x, train_y)
        # print (sum(svc.predict(test_x) == test_y)/len(test_y))
        # #0.805
        # acc1_svm = sum(svc.predict(test_x) == test_y)/len(test_y)

        out2 = nn.predict(vgg_features2)
        print (sum(np.argmax(out2,axis=1) == clab2)/len(clab2))
        acc2 = sum(np.argmax(out2,axis=1) == clab2)/len(clab2)

        # acc2 = None
        res.append([k, size, acc1, acc2])

for r in res:
    print ("\t".join([dir, *map(str, r)]))

with open("res.acc.{0}-{1}.txt".format(os.path.basename(dir), os.path.basename(dir2)), "w") as f:
    for r in res:
        f.write("\t".join([dir, *map(str, r), "\n"]))

