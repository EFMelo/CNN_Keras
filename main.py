from datasets import BartAndHomer
from model import BartAndHomerModel
from plots import Results

epochs = 30
target_size = (64, 64)

# load dataset
x_train, x_val, x_test = BartAndHomer.load_data(target_size=target_size, batch_size=16)

# building CNN
model = BartAndHomerModel.build(target_size=target_size)

# training
history = model.fit_generator(x_train, steps_per_epoch=196, epochs=epochs, validation_data=x_val, validation_steps=61)

# testing
prediction = model.predict(x_test)

# plotting the loss
Results.loss(epochs, history)

# plotting accuracy
Results.accuracy(epochs, history)