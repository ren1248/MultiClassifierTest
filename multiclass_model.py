from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def define_dense_model_single_layer(input_length, activation_f='sigmoid', output_length=1):
    model = keras.Sequential([
        layers.Input(shape=(input_length,)),
        layers.Dense(output_length, activation=activation_f)
    ])
    return model

def define_dense_model_with_hidden_layer(input_length, 
                                         activation_func_array=['relu','sigmoid'],
                                         hidden_layer_size=10,
                                         output_length=1):
    model = keras.Sequential([
        layers.Input(shape=(input_length,)),
        layers.Dense(hidden_layer_size, activation=activation_func_array[0]),
        layers.Dense(output_length, activation=activation_func_array[1])
    ])
    return model

def get_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255
    return (x_train, y_train), (x_test, y_test)

def fit_mnist_model(x_train, y_train, model, epochs=2, batch_size=2):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

def evaluate_mnist_model(x_test, y_test, model):
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, accuracy

if __name__ == "__main__":
    # Test cases
    model_single_layer = define_dense_model_single_layer(43, activation_f='sigmoid', output_length=1)
    assert len(model_single_layer.layers) == 2, "model should have 2 layers"
    
    model_hidden_layer = define_dense_model_with_hidden_layer(43, activation_func_array=['relu','sigmoid'], hidden_layer_size=11, output_length=1)
    assert len(model_hidden_layer.layers) == 3, "model should have 3 layers"
