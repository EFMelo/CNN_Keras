from keras.preprocessing import image
from keras.datasets import mnist, cifar10
from keras.utils.np_utils import to_categorical
from os import scandir
from numpy import array

class BartAndHomer:
    
    @classmethod
    def load_data(cls, target_size, batch_size):

        """
        Description
        -----------
        Dataset with 196 training samples, 61 validation samples and 12 test samples.
        Classes:
            0 - Bart
            1 - Homer
    
        Parameters
        ----------
        target_size : tuple or list
                     Size of the images.
        batch_size: int
    
        Returns
        -------
        x_train : keras.preprocessing.image.DirectoryIterator
        x_val : keras.preprocessing.image.DirectoryIterator
        x_test : ndarray
        """
        
        cls.__target_size = target_size
    
        # Generating Images
        train_generator = image.ImageDataGenerator(rescale = 1./255, rotation_range=7, horizontal_flip=True, zoom_range=0.2)
        
        val_generator = image.ImageDataGenerator(rescale=1./255)
        
        # Creation of the training and validation dataset
        x_train = train_generator.flow_from_directory('datasets/bart_and_homer/training', target_size=cls.__target_size, 
                                                       batch_size=batch_size, class_mode='binary')
        
        x_val = val_generator.flow_from_directory('datasets/bart_and_homer/validation', target_size=cls.__target_size,
                                                   batch_size=batch_size, class_mode='binary')
        
        # Creating the test dataset
        x_test = cls.__build_test()
        
        return x_train, x_val, x_test
    
    
    @classmethod
    def __build_test(cls):
        
        x = []
        
        for folder in scandir('datasets/bart_and_homer/test'):
            for img in scandir(folder.path):
                data = image.load_img(img.path, target_size=cls.__target_size)
                x.append(image.img_to_array(data) / 255)
        
        x = array(x)
        
        return x


class KerasDataSet:
    
    __x_train, __x_val, __x_test = None, None, None
    __y_train, __y_val, __y_test = None, None, None
    
    
    @classmethod
    def load_data_mnist(cls):
        
        # load dataset
        (cls.__x_train, cls.__y_train), (x, y) = mnist.load_data()
        
        # train data
        cls.__x_train = (cls.__x_train.astype('float16') / 255).reshape(cls.__x_train.shape[0], 28, 28, 1)  # between 0 and 1
        cls.__y_train = to_categorical(cls.__y_train)  # one-hot
        
        # validation and test data
        x = (x.astype('float16') / 255).reshape(x.shape[0], 28, 28, 1)  # between 0 and 1
        cls.__build_val_test_data(x, y)
        
        return cls.__x_train, cls.__y_train, cls.__x_val, cls.__y_val, cls.__x_test, cls.__y_test
    
    
    @classmethod
    def load_data_cifar10(cls):
        
        # load dataset
        (cls.__x_train, cls.__y_train), (x, y) = cifar10.load_data()
        
        # train data
        cls.__x_train = cls.__x_train.astype('float16') / 255  # between 0 and 1
        cls.__y_train = to_categorical(cls.__y_train)  # one-hot
        
        # validation and test data
        x = x.astype('float16') / 255  # between 0 and 1
        cls.__build_val_test_data(x, y)
        
        return cls.__x_train, cls.__y_train, cls.__x_val, cls.__y_val, cls.__x_test, cls.__y_test
    
    
    @classmethod
    def __build_val_test_data(cls, x, y):
        
        # input
        cls.__x_val = x[0:9000]
        cls.__x_test = x[9000:10000]
        
        # output
        y = to_categorical(y)  # one-hot
        cls.__y_val = y[0:9000]
        cls.__y_test = y[9000:10000]