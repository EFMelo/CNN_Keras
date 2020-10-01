from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam

class BartAndHomerModel:

    @classmethod
    def build(cls, target_size):
        
        model = Sequential()

        # conv - 32
        model.add(Conv2D(32, (3,3), input_shape=(target_size[0], target_size[1], 3), activation='relu', padding='same'))
        model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
        model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2,2)))

        # conv - 64
        model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2,2)))
        
        # conv - 128
        model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2,2)))
        
        # Flatening
        model.add(Flatten())

        # Dense layers
        model.add(Dense(30, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(30, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))

        # Configing the network
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        return model