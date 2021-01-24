from keras.models import Sequential 
from keras.layers import Dense
from keras import optimizers
from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from genetic_algorithm import *
from plot_tools import *

class Swish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return (sigmoid(x) * x)

get_custom_objects().update({'swish': Swish(swish)})

def create_neural_network(nodes_per_layer,hidden_layers,learning_rate):
    model = Sequential()
    activation_func = 'swish'
    
    #create input layer with 6 nodes and first hidden layer
    model.add(Dense(units=nodes_per_layer, activation=activation_func, input_dim=6))
    # create the remaining hidden layers
    for i in range(hidden_layers-1):
        model.add(Dense(units=nodes_per_layer, activation=activation_func))
    # create the output layer with two nodes, using a softmax function
    model.add(Dense(units=2, activation='softmax'))
    
    optimizer = optimizers.Adam(lr=learning_rate,epsilon=0.0001)
    model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
               metrics=['accuracy'])
    
    return model

def train_network(nodes, layers, learning, loss, species,x,y):
    #print(individual)
    #nodes = int(individual[0])
    #layers = int(individual[1])
    #learning = int(individual[2])
    nodes = int(nodes)
    layers = int(layers)
    #print(nodes)
    model = create_neural_network(nodes,layers,learning)
    # this stops training if there's no improvement in 50 epochs
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    history = model.fit(x, y, epochs=500, batch_size=32,validation_split=0.2,shuffle=True,callbacks=[es],verbose=0)
        
    min_loss = min(history.history['val_loss'])
    return min_loss
