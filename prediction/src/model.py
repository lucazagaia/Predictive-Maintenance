import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

def build_hdea_model(input_shape):
    """
    CNN-based model with two output heads: RUL mean and log variance
    """
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv1D(32, kernel_size=5, activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(64, kernel_size=3, activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)

    rul_mean = layers.Dense(1, name="rul_mean")(x)
    rul_log_var = layers.Dense(1, name="rul_log_var")(x)

    model = models.Model(inputs=inputs, outputs=[rul_mean, rul_log_var])
    return model

def nll_loss(y_true, y_pred_mean, y_pred_log_var):
    """
    Negative log likelihood loss (aleatoric uncertainty)
    """
    precision = K.exp(-y_pred_log_var)
    return K.mean(precision * K.square(y_true - y_pred_mean) + y_pred_log_var)


def build_rul_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, kernel_size=5, activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(64, kernel_size=3, activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(1, name='rul')(x)

    model = models.Model(inputs=inputs, outputs=output)
    return model

def build_stronger_rul_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv1D(32, kernel_size=5, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    output = layers.Dense(1)(x)
    
    model = models.Model(inputs, output)
    return model