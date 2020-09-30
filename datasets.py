from keras.preprocessing import image
from os import scandir
from numpy import array

class BartAndHomer:
    
    @classmethod
    def load_data(cls, target_size, batch_size):
        
        cls.__target_size = target_size
    
        # Generating Images
        train_generator = image.ImageDataGenerator(rescale = 1./255, rotation_range=7, horizontal_flip=True,
                                             zoom_range=0.2)
        
        val_generator = image.ImageDataGenerator(rescale=1./255)
        
        # Creating the dataset
        x_train = train_generator.flow_from_directory('datasets/bart_and_homer/training', target_size=cls.__target_size,
                                                         batch_size=batch_size, class_mode='binary')
        
        x_val = val_generator.flow_from_directory('datasets/bart_and_homer/validation', target_size=cls.__target_size,
                                                     batch_size=batch_size, class_mode='binary')
        
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