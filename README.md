# Bulk insect image classification with CNNs.

## Dependencies:
	* Python (3.8 or later)
	* Numpy (1.19 or later)
	* TensorFlow (2.5 or later)

A typical command for installing dependency libraries is "pip":

```
pip3 install numpy
pip3 install tensorflow==2.5.0
```

Hoever, this may vary depending on your environment. "conda" is another common command.

## How to download codes:
```
git clone htts://...
```

## Code descriptions:

* nn_classify_vgg19.py: image classification with VGG19 feature extraction + a neural network model.

* dann_classify_vgg19.py: image classification with VGG19 feature extraction +  Domain Adversarial traing of Neural Network (DANN, Ganin et al. 2017).

* domain_classifier.py: dataset classification for quantifying dataset similarity ().


## How to run:
### Classification with CNN
```
python3 nn_classify_vgg19.py [source image folder] [target image folder] [min training images] [max training images] [step] [number of test images]
```
Example: 
```
python3 nn_classify_vgg19.py bulk_images_GH bulk_images_LH 100 900 100 200
```

This command trains a model with images from bulk_images_GH (source) and tests with 200 images from the same source folder. Then, the model predicts all images from bulk_images_LH (target) and the accuracy is recorded. Training/Testing is repeated 10 times with the increasing number of training images between 100 and 900 with 100 intervals.

Make sure that the target folder has identical folder structue with the source folder. 

- Source folder
	- Taxon1
	- Taxon2
	-  ...
- Target folder
	- Taxon1
	- Taxon2
	-  ...
	
### Classification with CNN + DANN
	
```
python3 dann_classify_vgg19.py [source image folder] [target image folder] [min training images] [max training images] [step] [number of test images]
```
Example: 
```
python3 dann_classify_vgg19.py bulk_images_GH bulk_images_LH 100 900 100 200
```

This command trains a DANN model with images from both bulk_images_GH (source) and bulk_images_LH, and predict images from the target.

### Dataset classification 
```
python3 domain_classifier.py [source] [target]

```
Example:
```
python3 domain_classifier.py bulk_images_GH bulk_images_LH
```

This command will do dataset classification from bulk_images_GH and bulk_images_LH.

