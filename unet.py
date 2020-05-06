import os
import skimage
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import cv2
from google.colab.patches import cv2_imshow

from keras.utils import to_categorical

imgs_path = "/Herlev"
class_id = 0
i=0

X = np.empty([917,256,256,3]) # 917 imagens 
y = np.empty([917,256,256,1])


# percorre todos os diretorios da base e carrega as imagens
for f in os.listdir(imgs_path): 
    for img_path in os.listdir(os.path.join(imgs_path,f)):
        if img_path.endswith(".BMP"):
            aux = cv2.resize(cv2.imread(os.path.join(imgs_path,f,img_path)),(256,256))
            X[i, :, :,:] = aux/255.0 # lembrar de normalizar as imagens [0,1]
            #plt.imshow(aux)
            
            
            url = img_path[:-4]
            url = url + "-d.bmp"
            print(img_path)
            print(url)
            #se for ground truth
            aux2 = cv2.resize(cv2.imread(os.path.join(imgs_path,f,url)), (256,256))
            #plt.imshow()
            #pegar só o que tem 128 na primeira camada
            ret,thresh1 = cv2.threshold(aux2[:,:,0],128,255,cv2.THRESH_BINARY)
            y[i, :, :,0] = thresh1/255.0
            #y[i, :, :,1] = thresh1/255.0
            #y[i, :, :,2] = thresh1/255.0
            
            i = i + 1 #só incrementa se já tiver pegado o ground truth
            
            
            #plt.imshow( thresh1)
            #plt.imshow(aux2)
                
            #plt.imshow(thresh)
            
            '''
            figura = plt.figure()
            figura.add_subplot(1,2, 1)
            plt.imshow(np.squeeze(aux))
            plt.xticks([]); plt.yticks([])
            figura.add_subplot(1,2, 2)
            plt.imshow(np.squeeze(aux2))
            plt.xticks([]); plt.yticks([])
            plt.show(block=True)
            print("-----------------------------------------")
            '''

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def unet(pretrained_weights = None,input_size = (256,256,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=2)])
    
    #model.summary()

    if(pretrained_weights):
      model.load_weights(pretrained_weights)

    return model
from keras.preprocessing.image import ImageDataGenerator
  
datagen = ImageDataGenerator(
      # set input mean to 0 over the dataset
      featurewise_center=False,
      # set each sample mean to 0
      samplewise_center=False,
      # divide inputs by std of dataset
      featurewise_std_normalization=False,
      # divide each input by its std
      samplewise_std_normalization=False,
      # apply ZCA whitening
      zca_whitening=False,
      # epsilon for ZCA whitening
      zca_epsilon=1e-010,
      # randomly rotate images in the range (deg 0 to 180)
      rotation_range=0,
      # randomly shift images horizontally
      width_shift_range=0.1,
      # randomly shift images vertically
      height_shift_range=0.1,
      # set range for random shear
      shear_range=0.,
      # set range for random zoom
      zoom_range=0.,
      # set range for random channel shifts
      #channel_shift_range=0.,
      # set mode for filling points outside the input boundaries
      fill_mode='nearest',
      # value used for fill_mode = "constant"
      cval=0.,
      # randomly flip images
      horizontal_flip=True,
      # randomly flip images
      vertical_flip=False,
      # set rescaling factor (applied before any other transformation)
      rescale=None,
      # set function that will be applied on each input
      #preprocessing_function=None,
      # image data format, either "channels_first" or "channels_last"
      #data_format=None,
      # fraction of images reserved for validation (strictly between 0 and 1)
      validation_split=0.0)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(X_train)

from keras.callbacks import ModelCheckpoint
model = unet()
model.load_weights('Unet.h5')


for x in range(20):
  print("-------------------"+str(x)+"-------------------")
  checkpointer = ModelCheckpoint(filepath="Unet.h5", verbose=1, save_best_only=True)
  history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs = 1, validation_data=(X_test, y_test), verbose=1, shuffle=True, callbacks=[checkpointer])
  #model.save_weights('Unet.h5')