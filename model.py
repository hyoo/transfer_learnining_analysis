import math
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras import backend as K


def build_model(feature_shapes, input_features, params):
    input_models = {}
    dropout_rate = params.get('dropout_rate')
    for fea_type, shape in feature_shapes.items():
        base_type = fea_type.split('.')[0]
        if base_type in ['cell', 'drug']:
            box = build_feature_model(input_shape=shape, name=fea_type,
                                      dense_layers=params.get('dense_feature_layers'),
                                      attention=params.get('attention', False),
                                      dropout_rate=dropout_rate)
            input_models[fea_type] = box

    inputs = []
    encoded_inputs = []
    for fea_name, fea_type in input_features.items():
        shape = feature_shapes[fea_type]
        fea_input = Input(shape, name='input.' + fea_name)
        inputs.append(fea_input)
        if fea_type in input_models:
            input_model = input_models[fea_type]
            encoded = input_model(fea_input)
        else:
            encoded = fea_input
        encoded_inputs.append(encoded)

    merged = concatenate(encoded_inputs)

    h = merged
    for i, layer in enumerate(params.get('dense')):
        x = h
        if i == 0 and params.get('attention', False):
            a = Dense(layer, activation='relu')(h)
            a = BatchNormalization()(a)
            b = Attention(layer)(a)
            h = tf.keras.layers.multiply([b, a])
        else:
            h = Dense(layer, activation=params.get('activation'))(h)
        if dropout_rate > 0:
            h = PermanentDropout(dropout_rate)(h)
        if params.get('residual'):
            try:
                h = tf.keras.layers.add([h, x])
            except ValueError:
                pass
    output = Dense(1)(h)

    return Model(inputs, output)


def build_feature_model(input_shape,
                        name='',
                        dense_layers=[1000, 1000, 1000],
                        activation='relu',
                        residual=False,
                        attention=False,
                        dropout_rate=0):
    x_input = Input(shape=input_shape)
    h = x_input
    for i, layer in enumerate(dense_layers):
        x = h
        if i == 0 and attention:
            a = Dense(layer, activation='relu')(h)
            a = BatchNormalization()(a)
            b = Attention(layer)(a)
            h = tf.keras.layers.multiply([b, a])
        else:
            h = Dense(layer, activation=activation)(h)
        if dropout_rate > 0:
            h = PermanentDropout(dropout_rate)(h)
        if residual:
            try:
                h = tf.keras.layers.add([h, x])
            except ValueError:
                pass
    model = Model(x_input, h, name=name)
    return model


class PermanentDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super(PermanentDropout, self).__init__(rate, **kwargs)
        self.uses_learning_phase = False

    def call(self, x, mask=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(x)
            x = K.dropout(x, self.rate, noise_shape)
        return x


def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def mae(y_true, y_pred):
    return tf.keras.metrics.mean_absolute_error(y_true, y_pred)


class Attention(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1].value, self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, V):
        Q = tf.keras.backend.dot(V, self.kernel)
        Q = Q * V
        Q = Q / math.sqrt(self.output_dim)
        Q = tf.keras.activations.softmax(Q)
        return Q

    def compute_output_shape(self, input_shape):
        return input_shape
