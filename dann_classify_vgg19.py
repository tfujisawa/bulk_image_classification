import os
import sys
import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import LearningRateScheduler

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

from dann.reversal import *

model = VGG19(weights="imagenet", include_top=False, input_shape=(255,255,3))

##Feature extraction by VGG19-Conv5 model
l5 = model.get_layer("block5_pool")
l5ga = GlobalAveragePooling2D()(l5.output)
l5_model = Model(model.inputs, l5ga)

##Dataset 1.
dir1 = sys.argv[1]

gen = image.ImageDataGenerator(preprocessing_function=preprocess_input) #Don't forget to include this preprocessing function for the VGG models...!
img_gen = gen.flow_from_directory(dir1, target_size=(255,255), batch_size=10, class_mode="sparse", shuffle=False)
vgg_features1 =l5_model.predict(img_gen)

img_data1 = np.concatenate([img_gen.next()[0] for i in range(0, img_gen.samples, 10)]) #10 = batch_size

img_names1 = [os.path.basename(f) for f in img_gen.filenames]
clab1 = np.concatenate([img_gen.next()[1] for i in range(0, img_gen.samples, 10)])

img_names1 = [os.path.basename(f) for f in img_gen.filenames]
img_names1_2 = ((os.path.basename(f), os.path.dirname(f)) for f in img_gen.filenames)
img_class1 = dict(img_names1_2)

#Dataset2
dir2 = sys.argv[2]

img_gen2 = gen.flow_from_directory(dir2, target_size=(255,255), batch_size=10, class_mode="sparse", shuffle=False)
img_data2 = np.concatenate([img_gen2.next()[0] for i in range(0, img_gen2.samples, 10)])
vgg_features2 =l5_model.predict(img_gen2)

img_names2 = [os.path.basename(f) for f in img_gen2.filenames]
clab2 = np.concatenate([img_gen2.next()[1] for i in range(0, img_gen2.samples, 10)])

vgg_features = np.concatenate((vgg_features1, vgg_features2))
clab = np.concatenate((clab1, clab2))
clab = tf.keras.utils.to_categorical(clab)

img_names = img_names1 + img_names2

dlab = np.concatenate((np.zeros_like(clab1), np.ones_like(clab2)))
dlab= tf.keras.utils.to_categorical(dlab)

wc = np.concatenate(( np.ones_like(clab1), np.zeros_like(clab2)))
wd = np.ones_like(wc)

class_id = {}
id = 0
for n in [os.path.dirname(f) for f in img_gen.filenames]:
    if not n in class_id:
        class_id[n] = id
        id += 1
print (class_id)
id_class = {v:k for k, v in class_id.items()}

res1 = []
res2 = []

if len(sys.argv) > 3:
    size_min, size_max, size_step = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    test_size = int(sys.argv[6])
else:
    size_min, size_max, size_step = 300, 800, 100 #LL -> LH
    #size_min, size_max, size_step = 400, 1400, 200 #GH -> LH 
    test_size = 200 

for s in range(size_min, size_max+1, size_step):
    size = s
    for k in range(3):

        ind = np.zeros(len(clab), dtype=np.bool)

        # ind[np.random.randint(0, len(clab), 80)] = True
        ind[np.random.choice(range(len(clab)), size=size, replace=False)] = True

        train_x = vgg_features[ind,]
        train_yc = clab[ind,:]
        train_yd = dlab[ind,:]

        train_wc = wc[ind]
        train_wd = wd[ind]

        tsize = test_size#size//4 #200 for LHQ, 400 for GHQ
        tind = np.random.choice(np.where(np.logical_not(ind))[0], size=tsize, replace=False)
        tind.sort()
        test_x = vgg_features[tind,:]
        test_yc = clab[tind,:]
        test_yd = dlab[tind,:]

        test_wc = wc[tind]
        test_wd = wd[tind]

        ##CNN model
        nclass = 12
        input = Input(shape=l5_model.output.shape[1:])
        out = Dense(512, activation="relu")(input)
        out = Dropout(0.6)(out)
        out = Dense(256, activation="relu")(out)
        out = Dropout(0.6)(out)

        out_class = Dense(nclass, activation="softmax", name="out_class")(out)
        out_domain = Dense(2, activation="softmax", name="out_domain")(out)

        hist0 = TrainingHistory(metric=["loss", "val_loss", "accuracy", "val_accuracy" ], outfile="log0.txt")

        nnc = Model(input, out_class)
        nnc.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
        nnc.fit(train_x, train_yc, epochs=300, verbose=1, validation_data=(test_x, test_yc), sample_weight=train_wc, callbacks=[hist0])

        acc1=sum(np.argmax(nnc.predict(test_x[test_wc==1,:]),1)==np.argmax(test_yc[test_wc==1,:],1))/len(test_yc[test_wc==1,:])
        acc2=sum(np.argmax(nnc.predict(test_x[test_wc==0,:]),1)==np.argmax(test_yc[test_wc==0,:],1))/len(test_yc[test_wc==0,:])

        res1.append([k, size, len(train_yc[train_wc==1,:]), acc1, acc2])

        ##CNN+DANN model
        nclass = 12
        input = Input(shape=l5_model.output.shape[1:])
        out = Dense(512, activation="relu")(input)
        out = Dropout(0.6)(out)
        out = Dense(256, activation="relu")(out)
        out = Dropout(0.6)(out)

        out_class = Dense(nclass, activation="softmax", name="out_class")(out)

        ##Simple reversal layer
        out_domain = ReversalLayer()(out)
        out_domain = Dense(2, activation="softmax", name="out_domain")(out_domain)

        hist = TrainingHistory(metric=["loss", "val_loss", "out_class_loss", "out_domain_loss", "val_out_class_loss",  "val_out_domain_loss", "out_class_accuracy", "out_domain_accuracy", "val_out_class_accuracy", "val_out_domain_accuracy"], outfile="log.txt")

        nncd = Model(input, [out_class, out_domain])

        nncd.compile(optimizer="sgd", loss=["categorical_crossentropy", "binary_crossentropy"],  metrics=["accuracy"], loss_weights=[1.,0.1]) #Tested, 1,0.5 and 1,0.1
        nncd.fit(train_x, [train_yc, train_yd],  epochs=300, verbose=1, validation_data=(test_x, [test_yc, test_yd]), sample_weight=[train_wc, train_wd], callbacks=[hist])

        acc1=sum(np.argmax(nncd.predict(test_x[test_wc==1,:])[0],1)==np.argmax(test_yc[test_wc==1,:],1))/len(test_yc[test_wc==1,:])
        acc2=sum(np.argmax(nncd.predict(test_x[test_wc==0,:])[0],1)==np.argmax(test_yc[test_wc==0,:],1))/len(test_yc[test_wc==0,:])

        res2.append([k, size, len(train_yc[train_wc==1,:]), acc1, acc2])
##############
#uncomment this part to output histories
# hist0.write_file()
# hist.write_file()

#save
with open("res1.acc.{0}-{1}.txt".format(os.path.basename(dir1), os.path.basename(dir2)), "w") as f:
    f.write("k\tN\tn\tacc1\tacc2\n")
    for r in res1:
        f.write("\t".join([str(x) for x in r])+"\n")

with open("res2.acc.{0}-{1}.txt".format(os.path.basename(dir1), os.path.basename(dir2)), "w") as f:
    f.write("k\tN\tn\tacc1\tacc2\n")
    for r in res2:
        f.write("\t".join([str(x) for x in r])+"\n")

