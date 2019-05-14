## DLpipeline
### A Guide on Deep Learning for Genomic Prediction: A Keras based pipeline to implement deep learning
#### M Pérez-Enciso & LM Zingaretti
#### miguel.perez@uab.es, laura.zingaretti@cragenomica.es

If you find this resource useful, please cite:

Pérez-Enciso M, Zingaretti LM, 2019. A Guide on Deep Learning for Genomic Prediction. submitted

and possibly

[Bellot P, De Los Campos G, Pérez-Enciso M. 2018. Can Deep Learning Improve Genomic Prediction of Complex Human Traits? Genetics 210:809-819.](https://www.genetics.org/content/210/3/809)

 * * *
 
Implementing DL, despite all its theoretical and computational complexities, is rather easy. This is thanks to Keras API (https://keras.io/) and TensorFlow (https://www.tensorflow.org/), which allow all intricacies to be encapsulated through very simple statements. TensorFlow is a machine-learning library developed by Google. In addition, the machine-learning python library scikit-learn (https://scikit-learn.org) is highly useful. Directly implementing DL in TensorFlow requires some knowledge of DL algorithms, and understanding the philosophy behind tensor (i.e., n-dimensional objects) manipulations. Fortunately, this can be avoided using Keras, a high-level python interface for TensorFlow and other DL libraries. Although alternatives to TensorFlow and Keras exist, we believe these two tools combined are currently the best options: they are simple to use and are well documented. 

Here we describe some Keras implementation details. Complete code is in [jupyter notebook](https://github.com/miguelperezenciso/DLpipeline/blob/master/PDL.ipynb), and example data are [DATA](https://github.com/miguelperezenciso/DLpipeline/tree/master/DATA) folder. To run the script, you need to have installed Keras and TensorFlow, preferably in a computer with GPU architecture. Installing TensorFlow, especially for the GPU architecture, may not be a smooth experience. If unsolved, an alternative is using a docker (i.e., a virtual machine) with all functionalities built-in, or a cloud-based machine already configured. One option is https://github.com/floydhub/dl-docker. 

### Deep Learning Jargon
DL is full of specific terms, here a few of the most relevant ones are defined (just in case).

|Term|Definition|
|:----:|----------|
|**Activation function**|The mathematical function f that produces neuron’s output f(w’x + b).|
|**Backpropagation**|Backpropagation is an efficient algorithm to compute the loss, it propagates the error at the output layer level backward. Then, the gradient of previous layers can be computed easily using the chain rule for derivatives.|
|**Batch**|In SGD, each of the sample partitions within a given epoch|
|**Convolution kernel**|Mathematically, a convolution is a function that can be defined as an ‘integral transform’ between two functions, where one of the functions must be a kernel. The discrete version of the operation is just the weighting sum of several copies of the original function (f) shifting over the kernel.|
|**Convolutional Neural Network (CNN)**|CNNs are an especial case of Neural Networks which uses convolution instead a full matrix multiplication in the hidden layers. A typical CNN is made up of dense fully connected layers and ‘convolutional layers’. The convolution is a type of linear mathematical operation.  |
|**Dropout**|Dropout means that a given percentage of neurons output is set to zero. The percentage is kept constant, but the specific neurons are randomly sampled in every iteration. The goal of dropout is to avoid overfitting.|
|**Early stopping**|An anti-overfitting strategy that consists of stopping the algorithm before it converges.|
|**Epoch**|In SGD and related algorithms, an iteration comprising all batches in a given partition. In the next epoch, a different partition is employed.|
|**Generative Adversarial Network (GAN)**|GANs are based in a simple idea: train two networks simultaneously, the Generator (G), which defines a probability distribution based on the information from the samples, and the Discriminator (D), which distinguishes data produced by G from the real data.|
|**Kernel = Filter = Tensor**|In DL terminology, the kernel is a multidimensional array of weights.|
|**Learning rate**| Specify the speed of gradient update|
|**Loss**|Loss function measures how differences between observed and predicted target variables are quantified.|
|**Neural layer**|‘Neurons’ are arranged in layers, i.e., groups of neurons that take the output of previous group of neurons as input |
|**Neuron**|The basic unit of a DL algorithm. A ‘neuron’ takes as input a list of variable values (x) multiplied by ‘weights’ (w) and, as output, produces a non-linear transformation f(w’x + b) where f is the activation function and b is the bias. Both w and b need to be estimated for each neuron such that the loss is minimized across the whole set of neurons.|
|**Multilayer Perceptron (MLP)**|Multilayer Perceptron Network is one of the most popular DL architectures, which consists of a series of fully connected layers, called input, hidden and output layers. Layers are connected by a directed graph. |
|**Optimizer**|Algorithm to find weights (w and b) that minimize the loss function. Most DL optimizers are based on Stochastic Gradient Descent (SGD).|
|**Pooling**|A pooling function substitutes the output of a network at a certain location with a summary statistic of the neighboring outputs. This is one of the crucial steps on the CNN architecture. The most common pooling operations are maximum, mean, median.|
|**Recurrent Neural Network (RNN)**|RNN architecture considers information from multiple previous layers. Then, in the RNN model, the current hidden layer is a non-linear function of both the previous layer(s) and of the current input (x). The model has memory since the bias term is based on the ‘past’. These networks can be used in temporal -like data structures. |
|**Stochastic Gradient Descent (SGD)**|An optimizing algorithm that consists of randomly partitioning the whole data set in subsets called ‘batches’ or ‘minibatches’ and update the gradient using only that data subset. The next batch is used in next iteration.|
|**Weight regularization**|An excess of parameters (weights, w) may produce the phenomenon called ‘overfitting’, which means that the model adjusts to the observed data very well, but prediction of new unobserved data is very poor. To avoid this, weights are estimated subject to constraints, a strategy called ‘penalization’ or ‘regularization’. The two most frequent regularizations are the L1 and L2 norms, which set restrictions on the sum of absolute values of w (L1) or of the square values (L2)|

### A Generic Keras Pipeline

After uploading, preprocessing and partitioning the dataset, an analysis pipeline in Keras requires of five main steps:
* A model is instantiated: The most usual model is ```Sequential```, which allows adding layers with different properties step by step.
* The architecture is defined: Here, each layer and its properties are defined. For each layer, number of neurons, activation functions, regularization and initialization methods are specified.
* The model is compiled: Optimizer algorithm with associated parameters (e.g., learning rate) and loss function are specified. This step allows us to symbolically define the operations (‘graphs’) to be performed later with actual numbers.
* Training: The model is fitted to the data and parameters are estimated. The number of iterations (‘epochs’) and batch size are specified, input and target variables need to be provided. The input data size must match that defined in step 2.
* Model predictions are validated via cross-validation.

A generic Keras script would look like:

```
# Load modules needed
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# keras items 
from keras.models import Sequential
from keras.layers import Dense, Activation

# Load the dataset as a pandas data frame
# X is a N by nSNP array with SNP genotypes
X = pd.read_csv('DATA/wheat.X', header=None, sep='\s+')
# Y is a N b nTRAIT array with phenotypes
Y = pd.read_csv('DATA/wheat.Y', header=None, sep='\s+')
# The first trait is analyzed
y = Y[0] 

# Data partitioning into train and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# no. of SNPs in data
nSNP = X_train.shape[1] 

# Instantiate model
model = Sequential()

# Add first layer containing 64 neurons
model.add(Dense(64, input_dim=nSNP))
model.add(Activation('relu'))
# Add second layer, with 32 neurons
model.add(Dense(32))
model.add(Activation('softplus'))
# Last, output layer contains one neuron (ie, target is a real numeric value)
model.add(Dense(1))

# Model Compiling 
model.compile(loss='mean_squared_error', optimizer='sgd')

# list some properties of the network
model.summary()

# Training
model.fit(X_train, y_train, epochs=100)

# Cross-validation: get predicted target values
y_hat = model.predict(X_test)

# Computes squared error in prediction
mse_prediction = model.evaluate(X_test, y_test)
```

### Implementing Multilayer Perceptrons (MLPs)
In Keras, a MLP is implemented by adding ‘dense’ layers. In the following code, a two layer MLP with 64 and 32 neurons is defined, where the input dimension is 200 (i.e., the number of SNPs):

```
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
```

As is clear from the code, activation functions are ‘relu’ and ‘softplus’ in the first and second layer, respectively.

### Implementing Convolutional Neural Networks (CNNs)
The following Keras code illustrates how a convolutional layer with max pooling is applied prior to the MLP described above:

```
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
```

### Implementing Recurrent Neural Neiworks (RNNs)
The following model is a simple implementation of 3 layers of LSTM with 256 neurons per layer:
 
```
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
```

### Implementing Generative Networks 
A Keras implementation of GANs can be found at https://github.com/eriklindernoren/Keras-GAN. 

### Activation Functions
In Keras, activation is defined for every Dense layer as

```model.add(Activation(‘activation’))```

where ```‘activation’``` can take values ‘sigmoid’, ‘relu’, etc (https://keras.io/activations/). 
### Loss
The loss is a measure of how differences between observed and predicted target variables are quantified. Keras allows three simple metrics to deal with quantitative, binary or multiclass outcome variables: mean squared error, binary cross entropy and multiclass cross entropy, respectively. Several other losses are also possible or can be manually specified. 

Categorical cross-entropy is defined, for *M* classes, as 
 
&sum;<sub>i=1</sub>&sum;<sub>c=1</sub>&gamma;log(p<sub>ic</sub>), with i=1..N, c=1..M

where *N* is the number of observations, &gamma; is an indicator variable taking value 1 if i-th observation pertains to c-th class and 0 otherwise, and *P* is the predicted probability for i-th observation of being of class c. 

Losses are declared in compiling the model:

```
# Stochastic Gradient Descent (‘sgd’) as optimization algorithm
# quantitative variable, regression
model.compile(loss='mean_squared_error', optimizer=’sgd’)

# binary classification
model.compile(loss='binary_crossentropy', optimizer=’sgd’)

# multi class classification
model.compile(loss='categorical_crossentropy', optimizer=’sgd’)
```

When using categorical losses, your targets should be in categorical format. In order to convert integer targets into categorical targets, you can use the Keras utility ```to_categorical```:

``` 
from keras.utils import to_categorical
categorical_labels = to_categorical(int_labels, num_classes=None)
```

See https://keras.io/utils/#to_categorical. 

The next table shows the most common combinations of Loss Function and Last Layer Activation to different problems. 

|Problem | Last Layer Activation | Loss |
| :-------: | :------: | :-----: |
| Binary classification  | Sigmoid| Binary cross-entropy|
| Multiclass  | Softmax    |Categorical cross-entropy |
| Regression  | Linear     |  MSE  |
| 'Logistic' Regression | Sigmoid  |MSE/Binary cross-entropy|


### Optimizers
One of the most popular numerical algorithms to optimize a loss is the **Gradient Descent**. We can mention three variants of GD: **Batch gradient descent**, which computes the loss function gradient for the whole training data-set , **Stochastic gradient descent (SGD)** which consists of randomly partitioning the whole data set in subsets called ‘batches’ and update the gradient using only a single subset, then the next batch is used for the next iteration and, finally,  **minibatch gradient descent**,  which  is a combination of the two previous methods and it is based on spliting the training dataset into small batches. The gradient is averaged over a small number of samples allowing to reduce noise and code speed acceleration. Numerous optimizers exist and no clear rule on which one is best exist.

 SGD can be outperformed by SGD variants such as:
 
 - **MOMENTUM** accelerates SGD by moving on the relevant direction. The term increases when the gradients are moving in the same direction, and is reduced otherwise. Keras SGD function has the momentum option, which is 0.0 at default. 
 
-  **NESTEROV** is also implemented in keras sgd, being False at default. It is a predictor- corrector algorithm which generally overcomes the Momentum estimator.  It is implemented into two steps: in the predictor stage, the trajectory is linearly extrapolated as in the Moment, but in the second stage, it is corrected resulting on a convergence acceleration. 

These optimizers can be implemented in Keras as:

```
sgd = optimizers.SGD(momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
```

Keras also implements some adaptative optimizers functions as: 
- **Adagrad** allows to control the learning rate considering the occurrence of parameters updates, i.e. the learning rate drops when the frequency of updates increases. It is recommended when data are sparse. 
- **Adadelta** is an extension of the Adagrad which only adapts learning rates basing on a restricted windows (w) of past gradients. 
- **RMSPROP** (Root mean Square Propagation) is also an adaptative learning rate algorithm which combines SGD and Root mean square propagation. Basically, it uses the exponential weighted average instead of individual gradient of w at the backprop state adjusting, at once, the learning rate. It shows a good behavior in Recurrent Neural Networks. 
- **Adam** is an adaptative moment method where a learning rate is maintained for each weight and separately adapted. 
- **Adamax** is a variant of Adam based on infinite norm
- **Nadam** is a combination of Nesterov and Adam algorithms. 

See https://keras.io/optimizers/ 


### Protection against Overfitting
Keras allows implementing **early stopping** via the callback procedure. The user needs to provide a monitored quantity, say test loss, and the program stops when it stops improving (https://keras.io/callbacks/#earlystopping):

```
from keras.callbacks import EarlyStopping, Callback

early_stopper = EarlyStopping(monitor='val_loss', 					
                              min_delta=0.1, 
                              patience=2, 
                              verbose=0, 
                              mode='auto')
		
model.fit(X_train, 
          y_train, 
          epochs=100, 		
          verbose=1, 
          validation_data(X_test, y_test), 
          callbacks=[early_stopper])
```

In Keras, the available **regularizers** are L1 and L2 norm regularizers, which can also be combined in the so called ‘Elastic Net’ procedure, i.e., a mixed L1 and L2 regularization. In Keras, regularizers are applied to either kernels (weights), bias or activity (neuron output) and are specified together with the rest of layer properties, e.g.:

```
from keras.models import Sequential
from keras.layers import Dense, Activation 
from keras import regularizers

model.add(Dense(64, 
                input_dim=64,
                kernel_regularizer=regularizers.l2(0.01), 
                activity_regularizer=regularizers.l1(0.01) ) )
```

In Keras, different **dropout** rates can be specified for each layer, after its definition, e.g.:

```
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
```

### Hyperparameter Optimization
DL is not a single method, it is a heterogenous class of machine learning algorithms that depend on numerous hyperparameters, e.g., number of layers, neurons per layer, dropout rate, activation function and so on. DL optimization does require a general idea of which hyperparameters to optimize, together with a plausible range of values. Optimizing hyperparameter values is perhaps the most daunting task in using DL, which of course need to be done without resorting to the validation datasets!, and has been the topic of multitude of specialized papers. While for certain tasks like image analyses there are some specialized pre-trained networks or general architectures, this is not the case for new problems such as genomic prediction. 

In any realistic scenario, it is impossible to explore the whole space of hyperparameters, and sensible ranges should be chosen a priori. For instance, it is probably unnecessary to go beyond 3 – 5 layers or over say 100 neurons per layer. Testing up to four activation functions should probably capture all expected patterns. Similarly, each of dropout, L1 or L2 regularization does the same job and so only one hyperparameter can be explored. As for the optimization algorithm, we have not found important differences among those for the case of genomic prediction. If you are using CNNs, additional hyperparameters can be tuned, mainly number of filters and kernel width. In our experience with human data ([Bellot et al 2018](https://www.genetics.org/content/210/3/809)), the optimum kernel width was very small (~ three SNPs) but this will likely depend on the extent of linkage disequilibrium between markers and on the genetic architecture of the phenotype.

Once an initial hyperparameter space has been specified, a grid search could be performed if the number of hyperparameters is not very large (say ≤ 4), although a random search is much more efficient ([Goodfellow et al. 2016](https://www.deeplearningbook.org/)). Finally, other sophisticated approaches can be envisaged, such as genetic algorithms. In [Bellot et al 2018](https://www.genetics.org/content/210/3/809), we modified the implementation by Jan Liphardt (https://github.com/jliphard/DeepEvolve). The modified script can be retrieved from https://github.com/paubellot/DeepEvolve and https://github.com/paubellot/DL-Biobank/tree/master/GA. Our recommendation is that the number of generations should be relatively large. If computing time is too large, the data could be split into smaller subsets. In any case, we do recommend some narrow grid / random search to be performed around values suggested by the genetic algorithm. Note that optimizing hyperparameters for all desired marker sets and phenotypes will be unfeasible. We recommend choosing a few hyperparameter combinations that are near-optimum across a range of phenotypes / marker sets and that span a diversity of architectures, e.g., with varying neuron layers.

The next table lists the main DL hyperparameters:

|Hyperparameter|Role | Issues |
| :-------: | :------: | :-----: |
|Optimizer  |Algorithm to optimize the loss function. Most are based in SGD.| Optimization algorithms for training deep models includes some specializations to solve different challenges. |
|Learning rate | Specify the speed of gradient update.|Can result in meandering if too low and in reaching local maxima if too high. |
|Batch size|Determines number of samples in each SGD step.  | Can slow convergence if too small.|
|Number of layers | Controls flexibility to fit the data |The bigger the number, the higher the flexibility but may increase overfitting.|
|Neurons per layer|The bigger the number, the higher the flexibility.|The bigger the number, the higher the flexibility but may increase overfitting and poor training.|
|Convolutinal kernel width*|A larger kernel allows learning more complex patterns.|A larger kernel allows learning more complex patterns.|
|Activation|Makes possible to learn non-linear complex functional mappings between the inputs and response variable|Numerous options. No uniformly best function.|
|Weight regularization|Controls overfitting.|Decreasing the weight regularization allows the model to fit the training data better.|
|Dropout|Controls overfitting.|A higher dropout helps to reduce overfitting.|
