import tensorflow as tf
from keras.applications import resnet50, mobilenet_v2
from keras.models import save_model, load_model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
import json
from keras import backend as K

def create_MobileNetV2():
    model = mobilenet_v2.MobileNetV2(include_top=True, weights='imagenet')
    model.load_weights('models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5')
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.save('mobileNetV2_model.h5')

def load_image(path):

    img = image.load_img(path,target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    return x

def save_list(list):
    with open('imagenet_labels_1000.txt','w') as f:
        for item in list:
            f.write("%s\n" % item)
    return list


if __name__ == "__main__":

    # loads the MobileNet model (change the path to the location of your model.h5)
    model = load_model("models/mobileNetV2_model.h5")

    """
    This is used to print the node names of the tf graph. You will need to
    find the name(with the activation function) of the intermediate layer and save it, Unity needs it.
    """
    layers = [layer.output for layer in model.layers]
    print(layers)

    # Displays the model summary, not sure if you are going to need it.
    #model.summary()

    # Get input/output shape
    print("Input shape: ", model.input_shape)
    print("Output shape: ", model.output_shape)

    # Load and preprocess the image
    x_image = load_image("elephant.jpg")

    # ImageNet image preprocessing
    x_image = mobilenet_v2.preprocess_input(x_image)

    labelsList = list()
    with open("imagenet_labels_1000.txt") as f:
        for line in f:
            labelsList.append(line)

    # predict
    predictions = model.predict(x_image)
    max_pred = np.argmax(predictions)
    print(max_pred)
    print("prediction :", labelsList[max_pred])

    # intermediate layer (specify with index, input layer has index zero)
    intermediate_layer_output = K.function([model.layers[0].input],[model.layers[3].output])
    # get output with given input
    layer_output = intermediate_layer_output([x_image])
    # transform to numpy array for flattening
    out = np.asarray(layer_output)
    out = out.flatten()
    print(out.shape)



    #load classes
    #class_idx = json.load(open("imagenet_class_index.json"))
    #labels = [class_idx[str(k)][1] for k in range(len(class_idx))]
    #save_list(labels)