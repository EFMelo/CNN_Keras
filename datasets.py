from keras.preprocessing import image
from keras.datasets import mnist
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


class Mnist:
    
    @classmethod
    def load_data(cls):
        
        # load dataset
        (x_train, y_train), (x, y) = mnist.load_data()
        
        # input
        x_train = (x_train.astype('float16') / 255).reshape(x_train.shape[0], 28, 28, 1)  # between 0 and 1
        x = (x.astype('float16') / 255).reshape(x.shape[0], 28, 28, 1)  # between 0 and 1
        x_val = x[0:9000]
        x_test = x[9000:10000]
        
        # output
        y_train = to_categorical(y_train)  # one-hot
        y = to_categorical(y)  # one-hot
        y_val = y[0:9000]
        y_test = y[9000:10000]
        
        return x_train, y_train, x_val, y_val, x_test, y_test