from keras.applications.vgg16 import VGG16
from keras import optimizers
from keras.models import Model
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D, Convolution3D, UpSampling2D
from keras.utils import np_utils
import skimage
from skimage import io, color
from keras.optimizers import SGD
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img

"""
Colorizing a Gray Scale image utilizing LAB format.
L band - Brightness, is the 0 band which will be passed into the network for an input. Range of values: [0,100]
A and B band - Deal with colors. Range of values: [-128, 128] (give or take)

Operating the network: This network is thus far only created to output a correct shape. There are no weights associated
yet. The plan is to train the network to properly adjust the weights to make the output shape the A and B bands.

Training on A and B bands: Getting the right output shape is a pain. The current shape is (75,75, 6). This means the A band
will be placed witin (75, 75, 0) through (75,75,2) and B will be placed in the res (3-5). This way, both bands can be 
discovered and trained for at once.

Note on training: The A and B bands will have to all be divided by 128 to make their values fall between [-1, and 1]. This is so
the output of the network can be mapped to the correct values, as it outputs [-1,1] so the output values can be multiplied by
128 to create the desired values (in theory).
"""


def main():
    # Loading an initial test image
    image = skimage.io.imread('gsun_16b7dcdcc991ff9bdd051443d482c2f7.jpg')

    # Moving the original image into LAB format
    labImage = skimage.color.rgb2lab((image))
    #labImage.reshape(225,225,3)

    # Grabbing just the L band (brightness)
    L_band = labImage[:,:,0]

    # Moving the L_band list into an np array
    L_input = np.zeros((1, 225,225,1))
    L_input[0,:,:,0] = L_band[0:225,0:225]

    #L_input = np.array(L_band, dtype=float)

    #image = np.array(image, dtype=float)
    Desired_Band_A = np.array(labImage[:,:,1], dtype=float)/128.0
    Desired_Band_B = np.array(labImage[:,:,2], dtype=float)/128.0

    # For the labels, must split 225,225,1 into 75,75,6 !!!
    imageLabel = np.zeros((1,75,75,6))

    #Splitting bands A and B into 3 equal sized arrays len=75
    A_List = []


    A_List = splitArray(Desired_Band_A)
    B_List = splitArray(Desired_Band_B)
    
    # Going through each list and creating a 75,75,6 for the two bands
    for i in range(0,5):
        if i < 3:
            imageLabel[0,:,:,i] = np.array(A_List[i], dtype=float)
        else:
            imageLabel[0,:,:,i] = np.array(B_List[i-3],dtype=float)



    model = Sequential()
    model.add(Convolution2D(8, (3, 3), activation='relu', padding='same', strides=2,input_shape=(None,None,1)))
    model.add(Convolution2D(8, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2))) # Adding a pooling of features
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same', strides=2))
    model.add(UpSampling2D(size=(2,2))) # Upsampling to increase size after pooling
    model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D(size=(3,3)))
    model.add(Convolution2D(8,(3,3),activation='relu'))
    model.add(Convolution2D(8,(3,3),activation='relu'))
    model.add(UpSampling2D(size=(2,2)))
    model.add(Convolution2D(8,(3,3),activation='relu'))
    model.add(Convolution2D(16, (3,3), activation='relu'))
    model.add(Convolution2D(8,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(8,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(4,(3,3), activation='relu'))

    model.add(Convolution2D(6,(3,3), activation='tanh'))
    #model.compile(optimizer='rmsprop', loss='mse')
    sgd = SGD(lr=1e-4, momentum=0.9)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    print( model.output_shape)
    model.fit(x=L_input, y=imageLabel, batch_size=1, epochs=100)
#^-----------------------------------------------------------------------------main()
    

def splitArray(arrayToSplit):
    """Function to split an array of [225,225] into 3 equal arrays of [75,75].
    IN: A [225,225] list or np.array
    OUT: A python list of 3 np.arrays of size [75,75]"""

    retList = []
    retList.append(arrayToSplit[0:75, 0:75])
    retList.append(arrayToSplit[75:150, 75:150])
    retList.append(arrayToSplit[150:225, 150:225])
    return retList
#^-----------------------------------------------------------------------------splitArray(arrayToSplit)

#Standard broiler plate to run as main
if __name__ == '__main__':
    main()
