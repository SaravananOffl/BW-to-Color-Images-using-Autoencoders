from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Layer
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense, Input
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.optimizers import Adam
from PIL import Image
import numpy as np
from skimage import io, color
from skimage.transform import resize
import os
import cv2

import tensorflow as tf
from tensorflow.python import keras
'''
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
keras.backend.set_session(tf.Session(config=config))
'''
directory = "images/"
paths = os.listdir(directory)
#np.random.shuffle(paths)
'''
batch_size = 8

length = len(paths)
valPaths = paths[int(0.8*length):]
paths = paths[:int(0.8*length)]

length = len(paths)
print("Training Set : ", length , " Validation Set : ", len(valPaths))
steps_per_epoch = length/batch_size
'''
def random_brightness(image):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    image_HLS = np.array(image_HLS, dtype = np.float64) 
    random_brightness_coefficient = np.random.uniform()+0.5 ## generates value between 0.5 and 1.5
    image_HLS[:,:,1] = image_HLS[:,:,1]*random_brightness_coefficient ## scale pixel values up or down for channel 1(Lightness)
    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return image_RGB

def loadXY(paths, directory):
    x = []
    y = []
    for path in paths :
        img = io.imread(directory + path)
        img = random_brightness(img)
        img = color.rgb2lab(img)
        x.append(np.stack((img[:,:,0],)*3, axis=-1))
        #x.append(img[:,:,0])
        y.append(img[:,:,1:])

    x = np.array(x)/255
    #x = x.reshape(x.shape+(1,))
    y = np.array(y)/128

    return x,y

def ValidationLoader(valPaths, batch_size) :
    L = len(valPaths)

    while True:
        batch_start = 0
        batch_end = batch_size

        #np.random.shuffle(valPaths)

        while batch_start < L :
            limit = np.minimum(batch_end, L)

            vx, vy = loadXY(valPaths[batch_start:limit], "trainingSet/")

            yield (vx,vy)

            batch_start += batch_size
            batch_end += batch_size


def ImageLoader(paths, batch_size):
    L = len(paths)

    while True:
        batch_start = 0
        batch_end = batch_size

        np.random.shuffle(paths)

        while batch_start < L :
            limit = np.minimum(batch_end, L)

            x, y = loadXY(paths[batch_start:limit], "trainingSet/")

            yield (x,y)

            batch_start += batch_size
            batch_end += batch_size


vggModel = VGG16(include_top = False, weights = "imagenet", input_shape = (None,None,3))

transfer_layer = vggModel.get_layer('block5_conv3')
vggModel = Model(inputs=vggModel.input, outputs=transfer_layer.output)
vggModel.trainable = False

newModel = Sequential()
newModel.add(vggModel)

#newModel.add(UpSampling2D((2,2)))
newModel.add(BatchNormalization())
newModel.add(Conv2D(512, (3, 3), padding = 'same', activation = "relu"))
newModel.add(BatchNormalization())
newModel.add(Conv2D(512, (3, 3), padding = 'same',  activation = "relu"))

newModel.add(UpSampling2D((2, 2)))
newModel.add(BatchNormalization())
newModel.add(Conv2D(256, (3, 3), padding = 'same', activation = "relu"))
newModel.add(BatchNormalization())
newModel.add(Conv2D(256, (3, 3), padding = 'same',  activation = "relu"))
newModel.add(BatchNormalization())
newModel.add(Conv2D(256, (3, 3), padding = 'same',  activation = "relu"))
 
newModel.add(UpSampling2D((2, 2)))
newModel.add(BatchNormalization())
newModel.add(Conv2D(128, (3, 3), padding = 'same',  activation = "relu"))
newModel.add(BatchNormalization())
newModel.add(Conv2D(128, (3, 3), padding = 'same',  activation = "relu"))

newModel.add(UpSampling2D((2, 2)))
newModel.add(BatchNormalization())
newModel.add(Conv2D(64, (3, 3), padding = 'same',  activation = "relu"))
newModel.add(BatchNormalization())
newModel.add(Conv2D(64, (3, 3), padding = 'same',  activation = "relu"))

newModel.add(UpSampling2D((2,2)))
newModel.add(Conv2D(2, (3, 3), padding = 'same',  activation = "tanh"))


newModel.summary()

adam = Adam(lr = 0.00001)

newModel.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
newModel.load_weights("weights.hdf5")
'''
filepath="weights2L/colorize-updated-weights-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='min')
callbacks_list = [checkpoint]

newModel.fit_generator(ImageLoader(paths, batch_size), steps_per_epoch=steps_per_epoch, validation_data = ValidationLoader(valPaths, batch_size), validation_steps = len(valPaths)/batch_size,epochs=20, callbacks=callbacks_list)'''
x = []
img = io.imread("k.jpg")
#img = np.stack((img,)*3, axis=-1)
#img = resize(img,(512,1024))
io.imsave("original.jpg", img)
img = color.rgb2lab(img)
x.append(np.stack((img[:,:,0],)*3, axis=-1))
x = np.array(x)
x = x/255
#x = np.expand_dims(x,axis=0)

y = newModel.predict(x)
y = np.squeeze(y)

print(y.shape)

img[:,:,1:] = y * 128
#img[:,:,1] = img[:,:,1] * 1.5
img = np.around(img)
img = color.lab2rgb(img)
print(img.shape)

io.imsave("output.jpg", img)
