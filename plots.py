import matplotlib.pyplot as plt

class Results:
    
    @classmethod
    def loss(cls, epochs, history):
        
        plt.plot(range(epochs), history.history['loss'], label="Training")
        plt.plot(range(epochs), history.history['val_loss'], label="Validation")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
    
    @classmethod
    def accuracy(cls, epochs, history):
        
        plt.plot(range(epochs), history.history['accuracy'], label="Training")
        plt.plot(range(epochs), history.history['val_accuracy'], label="Validation")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()