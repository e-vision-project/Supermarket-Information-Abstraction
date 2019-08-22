# Supermarket-Information-Abstraction

This repository contains the classification/validation procedure for extracting features from images using the layer outputs of a MobileNet v2 ANN architecture. Then these features are fed into an SVM classifier (libSVM - matlab implementation) for training/testing. This code was employed to infer the information abstraction level from images taken in a supermarket environment.

The 1st step employs a deep neural network for extracting features from images by means of feature-maps from the layers of a DNN. Here, we employ a MobileNet v2 trained on the Imagenet dataset. The code performs the features extraction from all available layers of the network for all the images in a given folder. For more information see the well documented, feature_extraction.ipynb

Additionally, we provide a matlab script that can operate on the extracted features and feeds them to a Support Vector Machine. Please note that the output features of the 1st step should be put in the matlab path as a ".npy" file. The provided version operates on the features from images captured during the on-site visits of the e-Vision project. The employed images/features are not provided so as to preserve participants' privacy.

Code dependencies:
```
For Tensorflow and Keras please see: https://www.tensorflow.org/tutorials/keras

For the libSVM please refer to: https://www.csie.ntu.edu.tw/~cjlin/libsvm/

For loading .npy files in Matlab please see: https://github.com/kwikteam/npy-matlab
```

If you use this code for your research, please cite our paper:
```
Georgiadis, K., Kalaganis, F., Migkotzidis, P., Chatzilari, E., Nikolopoulos, S., & Kompatsiaris, I. (to appear) A computer vision system supporting blind people - The supermarket case. International Conference on Computer Vision Systems
```

This work is part of project Evision that has been co‐financed by the European Regional Development Fund of the European Union and Greek national funds through the Operational Program Competitiveness, Entrepreneurship and Innovation, under the call RESEARCH – CREATE – INNOVATE (project code: T1EDK-02454).

