import os

from keras.preprocessing import image

from keras.models import Sequential

from keras.layers.core import Dense

from keras.optimizers import SGD

from keras.datasets import mnist
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class Veri_Ogren:
    __instance = None
    @staticmethod
    def getInstance():
        """ Static access method. """
        if Veri_Ogren.__instance == None:
            Veri_Ogren()
        return Veri_Ogren.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if Veri_Ogren.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            self.__ogren()
            Veri_Ogren.__instance = self
    def __ogren(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        """
        path = input("Test resim yolu: ")
        im = Image.open(path).convert('L')
        pixelMap = im.load()
        
        new_tuple = list(X_test[5])
        img = Image.new(im.mode, im.size,'white')
        pixelsNew = im.load()
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                img.putpixel((i, j), int(new_tuple[i][j]))
        
        img.save("out.png")
        
        
        img = Image.new("L",(28,28),'black')
        
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                img.putpixel((i, j), int(X_test[0][((i*28)+j)]))
        
        img.save("deneme.png")
        print(y_test[0])
        
        """
        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)


        self.model = Sequential()
        self.model.add(Dense(input_dim=X_train.shape[1],
                        output_dim = 50,
                        init =   'uniform',
                        activation = 'tanh'))

        from keras.layers.core import Activation
        from keras.layers.core import Dropout

        self.model.add(Dense(50, init='uniform'))
        self.model.add(Activation('tanh'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, init='uniform'))
        self.model.add(Activation('relu'))

        self.model.add(Dense(10, init='uniform'))
        self.model.add(Activation('softmax'))

        from keras.utils.np_utils import to_categorical
        y_train_ohe = to_categorical(y_train)

        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

        self.model.compile(loss = 'categorical_crossentropy',
                      optimizer = sgd)

        self.model.fit(X_train,
                  y_train_ohe,
                  nb_epoch = 50,
                  batch_size = 500,
                  validation_split = 0.1,
                  verbose = 1)



        y_test_predictions = self.model.predict_classes(X_test, verbose = 1)

        correct = np.sum(y_test_predictions ==  y_test)
        print('Test Accuracy: ', correct/float(y_test.shape[0])*100.0, '%')
    def test_image_from_directory(self,path):
        if not os.path.exists(path):
            print("I not find this path")
            return None

        image_array = list()
        for file in os.listdir(path):
            test_image = image.load_img(path+"/"+file,color_mode="grayscale",target_size=(28,28))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image,axis=0)

            #plt.imshow(test_image.reshape(28, 28), cmap='gray', interpolation='none')

            #test_image = test_image.reshape(784)

            new_array = list()
            for a in test_image[0]:
                for b in a:
                    new_array.append(b[0])

            image_array.append(new_array)

        image_array = np.array(image_array)

        result = self.model.predict_classes(image_array, verbose=1)
        # adapt figure size to accomodate 18 subplots
        plt.rcParams['figure.figsize'] = (7, 14)

        figure_evaluation = plt.figure()

        # plot 9 correct predictions
        for i in range(0,len(image_array)):
            plt.subplot(6, 3, i + 1)
            plt.imshow(image_array[i].reshape(28, 28), cmap='gray', interpolation='none')
            plt.title(
                "Predicted: {}".format(result[i]))
            plt.xticks([])
            plt.yticks([])
        plt.show()

        figure_evaluation

        return result

    def test_image_from_only_one(self,path):
        if not os.path.isfile(path):
            print("I not find this path")
            return None

        test_image = image.load_img(path, color_mode="grayscale", target_size=(28, 28))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        new_array = list()
        for a in test_image[0]:
            for b in a:
                new_array.append(b[0])
        image_array = np.array([new_array])

        result = self.model.predict_classes(image_array, verbose=1)

        return result


class Run:
    ogren = Veri_Ogren()
    while(1):
        path = ""
        isdirectoryOrfile = input("dir or file: ")
        if isdirectoryOrfile == "dir":
            path = input("path: ")
            result = ogren.test_image_from_directory(path)
            print(result)
        elif isdirectoryOrfile == "file":
            path = input("path: ")
            result = ogren.test_image_from_only_one(path)
            print(result)
        else:
            print("bye bye")
            break


Run()