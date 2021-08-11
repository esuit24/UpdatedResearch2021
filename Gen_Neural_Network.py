#imports
import tensorflow as tf
import numpy as np

#creates model
def set_model():
    model = tf.keras.models.Sequential() #feed forward
    return model

#trains model based on various parameters
def train(model, num_inputs, factor, num_labels, x_train, y_train):
    #set layers
    #input layer: 100 different x values to represent the function
    model.add(tf.keras.layers.Input(num_inputs))
    #hidden layers (3):
    model.add(tf.keras.layers.Dense(num_inputs*factor, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(num_inputs*factor, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(num_inputs*factor, activation = tf.nn.relu))
    #output layer:
    model.add(tf.keras.layers.Dense(num_labels,))

    #optimize
    model.compile(optimizer = 'adam',
                  loss = 'mean_squared_error',
                  metrics = [tf.keras.metrics.RootMeanSquaredError()])
    model.fit(x_train, y_train, epochs = 100)

#tests model based on answers and model
def test(model, x_test, y_test):
    results = model.evaluate(x = x_test, y = y_test)
    return results
    
#saves model
def save(model, model_name):
    model.save(model_name)
