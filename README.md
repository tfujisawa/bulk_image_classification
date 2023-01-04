# Bulk insect image classification with CNNs.
Codes for classification of bulk biodiversity images.

For details of this project, please refer to the manuscript:

Image-based taxonomic classification of bulk biodiversity samples using deep learning and domain adaptation

https://www.biorxiv.org/content/10.1101/2021.12.22.473797v1.abstract

Image data sets used in this study are available in Zenodo and Dryad.

https://zenodo.org/record/5823545#.Y7TlbNXP0uW

https://datadryad.org/stash/dataset/doi:10.5061/dryad.05qfttf4f

## Dependencies:
	* Python (3.8 or later)
	* Numpy (1.19 or later)
	* Scikit-learn (0.22 or later)
	* TensorFlow (2.5 or later)

A typical command for installing dependency libraries is "pip":

```
pip3 install numpy
pip3 install tensorflow==2.5.0
...
```

However, this may vary depending on your environment. "conda" is another common command.

## How to download codes:
```
git clone htts://...
```

## Code descriptions:

* **nn_classify_vgg19.py**: image classification with VGG19 feature extraction + a neural network model.

* **dann_classify_vgg19.py**: image classification with VGG19 feature extraction +  Domain Adversarial traing of Neural Network (DANN, Ganin et al. 2017).

* **domain_classifier.py**: dataset classification for quantifying dataset similarity ().

* **confusion_matrix_nn.py**: detailed prediction results of VGG19+NN model.

## How to run:
### Classification with CNN
```
python3 nn_classify_vgg19.py [source image folder] [target image folder] [min training images] [max training images] [step] [number of test images]
```
#### Example:
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
#### Example:
```
python3 dann_classify_vgg19.py bulk_images_GH bulk_images_LH 100 900 100 200
```

This command trains a DANN model with images from both bulk_images_GH (source) and bulk_images_LH, and predict images from the target.

### Dataset classification
```
python3 domain_classifier.py [source] [target]

```
#### Example:
```
python3 domain_classifier.py bulk_images_GH bulk_images_LH
```

This command will do dataset classification from bulk_images_GH and bulk_images_LH.

### Detailed classification results and confusion matrix
```
python3 confusion_matrix_nn.py [source image folder] [target image folder] [number of training images]
```
#### Example:
```
python3 confusion_matrix_nn.py bulk_images_GH bulk_images_LH 600
```
This command trains a NN model with 600 images from bulk_images_GH folder and predict images from bulk_images_LH folder, then detailed results are output to files.


### Workflow of the manuscript

```
#CNN model. Within/between dataset classificiation
python3 nn_classify_vgg19.py LH LH 100 700 100 200 #LH->LH, within-dataset classification

python3 nn_classify_vgg19.py LL LH 50 250 50 50 #LL->LL and LL->LH
python3 nn_classify_vgg19.py GH LH 100 900 100 200 #GH->GH and GH-> LH
python3 nn_classify_vgg19.py GH LL 100 900 100 200 #GH->GH and GH-> LL

#CNN + DANN. Between dataset classification
python3 dann_classify_vgg19.py LL LH 300 800 100 200 #LL -> LH
python3 dann_classify_vgg19.py GH LH 400 1400 200 400 #GH -> LH
python3 dann_classify_vgg19.py GH LL 300 1000 200 400 #GH -> LL

```
