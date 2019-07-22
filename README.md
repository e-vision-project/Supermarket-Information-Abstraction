# Supermarket-Information-Abstraction

This repository contains the classification/cross-validation pipline for extracting the information abstraction level from images taken in a supermarket environment.

The 1st step employs a deep neural network for extracting features from images by means of feature-map from the intermediate layers of a DNN. In our case we took advantage of a ResNet v1 trained on the Imagenet dataset. More information regarding the feature extraction process is provided in https://github.com/MKLab-ITI/intermediate-cnn-features

For the feature extraction procedure please cite:
```
Kordopatis-Zilos, G., Papadopoulos, S., Patras, I., & Kompatsiaris, Y. (2017, January). Near-duplicate video retrieval by aggregating intermediate cnn layers. In International conference on multimedia modeling (pp. 251-263). Springer, Cham.
```

Then, during the second step, the features are used in a matlab script by a linear-kernel Support Vector Machine. Please note that the features of the 1st step should be put in the matlab path as a ".npy" file. The provided version operates on the features from images captured during the on-site visits of the e-Vision project. The employed images/features are not provided so as to preserve participants' privacy.

If you use this code for your research, please cite our paper:
```
Georgiadis, K., Kalaganis, F., Migkotzidis, P., Chatzilari, E., Nikolopoulos, S., & Kompatsiaris, I. (to appear) A computer vision system supporting blind people - The supermarket case. International Conference on Computer Vision Systems
```
