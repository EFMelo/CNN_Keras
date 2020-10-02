from keras.preprocessing import image
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