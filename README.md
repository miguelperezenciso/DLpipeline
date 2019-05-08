## DLpipeline
### A Guide on Deep Learning for Genomic Prediction: A Keras based guide to implement deep learning
#### M Pérez-Enciso & ML Zingaretti
#### miguel.perez@uab.es, laura.zingaretti@cragenomica.es

If you find this resource useful, please cite:

Pérez-Enciso M, Zingaretti ML, 2019. 
A Guide on Deep Learning for Genomic Prediction. 
submitted

Here we describe some Keras implementation details. Complete code and example data are at https://github.com/miguelperezenciso/DLpipeline. You need to have installed Keras (https://keras.io/) and TensorFlow (https://www.tensorflow.org/), preferably in a computer with GPU architecture. Installing TensorFlow, especially for the GPU architecture, may not be a smooth experience. If unsolved, an alternative is using a docker (i.e., a virtual machine) with all functionalities built-in, or a cloud-based machine already configured. One option is https://github.com/floydhub/dl-docker. 

Implementing Multilayer Perceptrons (MLPs)
In Keras, a MLP is implemented by adding ‘dense’ layers. In the following code, a two layer MLP with 64 and 32 neurons is defined, where the input dimension is 200 (i.e., the number of SNPs):

from keras.models import Sequential
from keras.layers import Dense, Activation

nSNP=200 # no. of SNPs in data
# Instantiate
model = Sequential()
# Add first layer
model.add(Dense(64, input_dim=nSNP))
model.add(Activation('relu'))
# Add second layer
model.add(Dense(32))
model.add(Activation('softplus'))
# Last, output layer with linear activation (default)
model.add(Dense(1))

As is clear from the code, activation functions are ‘relu’ and ‘softplus’ in the first and second layer, respectively.

Implementing Convolutional Neural Networks (CNNs)
The following Keras code illustrates how a convolutional layer with max pooling is applied prior to the MLP described above:

from keras.models import Sequential
from keras.layers import Dense, Activation 
from keras.layers import Flatten, Conv1D, MaxPooling1D

nSNP=200 # no. of SNPs in data
nStride=3 # stride between convolutions
nFilter=32 # no. of convolutions

model = Sequential()
# add convolutional layer
model.add(Conv1D(nFilter, 
kernel_size=3, 
strides=nStride, 		
input_shape=(nSNP,1)))
# add pooling layer: here takes maximum of two consecutive values
model.add(MaxPooling1D(pool_size=2))
# Solutions above are linearized to accommodate a standard layer
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('softplus'))
model.add(Dense(1))

Implementing Generative Networks 
A Keras implementation of GANs can be found at https://github.com/eriklindernoren/Keras-GAN. 

Implementing Recurrent Neural Neiworks (RNNs)
The following model is a simple implementation of 3 layers of LSTM with 256 neurons per layer:
 
from keras.models import Sequential
from keras.layers import Dense, Activation

nSNP=200 # no. of SNPs in data

# Instantiate
model = Sequential()
model.add(LSTM(256,return_sequences=True, input_shape=(None,1), activation=’tanh’))
model.add(Dropout(0.1))
model.add(LSTM(256, return_sequences=True, activation=’tanh’))
model.add(Dropout(0.1))
model.add(LSTM(256, activation=’tanh’))
model.add(Dropout(0.1))
model.add(Dense(units=1))
model.add(Activation(’tanh’))
model.compile(loss=mse, optimizer=adam, metrics=['mae'])

# prints some details
model.summary()

Loss
The loss is a measure of how differences between observed and predicted target variables are quantified. Keras allows three simple metrics to deal with quantitative, binary or multiclass outcome variables: mean squared error, binary cross entropy and multiclass cross entropy, respectively. Several other losses are also possible or can be manually specified. 
Categorical cross-entropy is defined, for M classes, as 
 

where N is the number of observations, ic is an indicator variable taking value 1 if  i-th observation pertains to c-th class and 0 otherwise, and pic is the predicted probability for i-th observation of being of class c. Losses are declared in compiling the model:

# Stochastic Gradient Descent (‘sgd’) as optimization algorithm
# quantitative variable, regression
model.compile(loss='mean_squared_error', optimizer=’sgd’)

# binary classification
model.compile(loss='binary_crossentropy', optimizer=’sgd’)

# multi class classification
model.compile(loss='categorical_crossentropy', optimizer=’sgd’)

When using categorical losses, your targets should be in categorical format. In order to convert integer targets into categorical targets, you can use the Keras utility to_categorical:

from keras.utils import to_categorical
categorical_labels = to_categorical(int_labels, num_classes=None)

See https://keras.io/utils/#to_categorical. 

Activation Functions
In Keras, activation is defined for every Dense layer as

model.add(Activation(‘activation’))

where ‘activation’ can take values ‘sigmoid’, ‘relu’, etc (https://keras.io/activations/). The activation by default in Keras is ‘linear’, i.e., no function.  

Protection against Overfitting
Keras allows implementing early stopping via the callback procedure. The user needs to provide a monitored quantity, say test loss, and the program stops when it stops improving (https://keras.io/callbacks/#earlystopping).

from keras.callbacks import EarlyStopping, Callback

early_stopper = EarlyStopping(monitor='val_loss', 					
min_delta=0.1, 
patience=2, 
verbose=0, 
mode='auto')
model.fit(X_train, y_train, 
										epochs=100, 		
verbose=1, 
validation_data(X_test, y_test), 
		callbacks=[early_stopper])

In Keras, the available regularizers are L1 and L2 norm regularizers, i.e., Eq. 6, Eq.7, which can also be combined in the so called ‘Elastic Net’ procedure, i.e., a mixed L1 and L2 regularization. In Keras, regularizers are applied to either kernels (weights), bias or activity (neuron output) and are specified together with the rest of layer properties, e.g.:

from keras.models import Sequential
from keras.layers import Dense, Activation 
from keras import regularizers

model.add(Dense(64, input_dim=64,
kernel_regularizer=regularizers.l2(0.01), 
activity_regularizer=regularizers.l1(0.01)))

In Keras, different dropout rates can be specified for each layer, after its definition, e.g.:

model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
