# CNN with Keras

> Using Convolutional Neural Networks (CNNs) from the keras framework.

### Datasets

**Bart and Homer**

Dataset with `196 training` samples, `61 validation` samples and `12 test` samples.

`Objective`: To say the character of the image sent as input.

Outputs:
- `0`: Bart.
- `1`: Homer.

### Some Results

**Bart and Homer**

Using the dataset and training the CNN:

```python
from datasets import BartAndHomer
from models import BartAndHomerModel
from plots import Results

# load dataset
x_train, x_val, x_test = BartAndHomer.load_data(target_size=(64, 64), batch_size=16)

# building and training CNN
model = BartAndHomerModel.build(target_size=(64, 64))
history = model.fit_generator(x_train, steps_per_epoch=196, epochs=15, validation_data=x_val, validation_steps=61)

# CNN prevision
prediction = model.predict(x_test)

# loss and accuracy plotting
Results.loss(epochs, history)
Results.accuracy(epochs, history)
```

<p align="center">
  <img width="384" height="244" src="https://i.imgur.com/lAkxz0X.png">
</p>


### Contact

emeloppgi@gmail.com

[github.com/EFMelo](https://github.com/EFMelo)