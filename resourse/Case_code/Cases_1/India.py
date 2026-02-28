import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras.api as api
from keras import Model, layers, optimizers

def data_extract(dir_name, file_name, attributes):
    script_path = globals().get('__file__')
    if script_path is not None:
        script_dir = os.path.dirname(os.path.abspath(script_path))
        path = os.path.join(dir_name, file_name)
        relative_path = os.path.join(script_dir, '../../', path)
        file_path = os.path.abspath(relative_path)
    else:
        file_path = os.path.join(os.getcwd(), dir_name, file_name)
    
    df = pd.read_excel(file_path)
    df = df.fillna(value=0)
    df_data = df[f'{attributes[0]}']

    for j in attributes[1:]:
        other_data = df[f'{j}']
        df_data = pd.concat([df_data, other_data], axis=1)

    return pd.DataFrame(df_data)


def get_data(dir_name, country_list, num, attributes):
    data_dict = {}

    for n, i in enumerate(country_list):
        # data是一个Dateframe对象
        data = data_extract(dir_name, i, attributes)[:num]
        data_dict[i] = data

    return data_dict


def max_country_num(data, attribute_index):
    max_value = list(data[list(data.keys())[0]].max())[attribute_index]
    max_country = list(data.keys())[attribute_index]
    for i in list(data.keys())[1:]:
        country_value = list(data[i].max())[attribute_index]
        if country_value >= max_value:
            max_country = i
            max_value = country_value
    return max_country


def data_format(data, num_steps, target_day, attributes_len, regre_len, test_ratio=0.25, output_shape=False,
                Dense_outputs=False):

    X = np.array([data[i: i + num_steps] for i in range(len(data) - num_steps - target_day + 1)]).reshape(-1, num_steps,
                                                                                                          attributes_len)
    y = np.array([data[i + num_steps: i + num_steps + target_day] for i in
                  range(len(data) - num_steps - target_day + 1)]).reshape(-1, target_day, attributes_len)

    train_size = int((len(X) - target_day + 1) * (1.0 - test_ratio))
    train_X, test_X = X[:train_size], X[train_size:]
    train_y, test_y = y[:train_size, :, :regre_len], y[train_size:, :, :regre_len]

    if output_shape:
        print(train_X.shape)
        print(train_y.shape)

    if Dense_outputs:
        train_X = train_X.reshape(len(train_X), num_steps)
        train_y = train_y.reshape(len(train_y), target_day)
        test_X = test_X.reshape(len(test_X), num_steps)
        test_y = test_y.reshape(len(test_y), target_day)

    return train_X, train_y, test_X, test_y


def normalization(data, data_scaler):
    data_dict = {}
    key_list = list(data.keys())
    for n, i in enumerate(key_list):
        scaled_data = data_scaler.transform(data[i])
        data_dict[n] = scaled_data

    return data_dict


def training_normalization(data, suptitle, title, data_label, visualization=True, max_figure_num=2, block=False):

    data_scaler = MinMaxScaler()
    feature_scaler = MinMaxScaler(feature_range=(0, 1))

    max_country = max_country_num(data, attribute_index=0)
    max_data = data[max_country]
    data_scaler.fit(max_data)
    feature_scaler.fit(pd.DataFrame(max_data[attributes[0]]))

    data_dict = normalization(data, data_scaler)

    if visualization:
        data_visualization(data_dict, suptitle, title, data_label, max_figure_num=max_figure_num, block=block)

    return data_dict, data_scaler, feature_scaler


def mean_ape(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / (y_true))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def country_smooth(data, attributes_num, ratio_list):
    '''
    :param data : Dataframe object
    :param attributes_num: List object
    :param ratio_list:List object
    '''
    for i in attributes_num:
        smo_fea = np.array([])
        feature = data.iloc[:, i]
        for j in range(len(feature)):
            if j <= len(feature) - len(ratio_list):
                value = np.sum(feature[j:j + len(ratio_list)] * ratio_list)
                smo_fea = np.append(smo_fea, value)
            else:
                back = smo_fea[-(j - (len(feature) - len(ratio_list))):]
                forward = feature[j:]
                arr = np.append(forward, back)
                value = np.sum(arr * ratio_list)
                smo_fea = np.append(smo_fea, value)
        data.iloc[:, i] = smo_fea

    long = min(data.count())

    return data.iloc[:long, :]


def data_visualization(data, suptitle, title, data_label, cols=4, rows=4, max_figure_num=2, block=False):

    country_list = list(data.keys())
    drawing_num = rows * cols
    figure_num = min(int(np.ceil(len(country_list) / drawing_num)), max_figure_num)

    for j in range(figure_num):
        plt.close()
        country = country_list[j * drawing_num:(j + 1) * drawing_num]

        cols = min(cols, len(country))
        rows = int(np.ceil(len(country) / cols))

        figure, axs = plt.subplots(rows, cols, dpi=120, constrained_layout=True)
        figure.suptitle(f'{suptitle}')

        a = Axes
        if type(axs) == a:
            axs = np.array(axs).reshape(rows, cols)
        axs = axs.reshape(rows, cols)
        for n, i in enumerate(country):
            sample_data = np.array(data[i])[:, 0]
            ax = axs[n // cols, n % cols]
            ax.set_title(f'{i}th----{title}')
            ax.plot(sample_data, 'b', label=f'{data_label}')
            ax.legend()

        if block:
            plt.show()
        else:
            plt.show(block=block)
            plt.pause(1)
            plt.close()


class L_C(api.Model):
    def __init__(self):
        super().__init__()
        self.cov_1 = api.layers.Conv1D(48, kernel_size=3, padding='same', activation='relu')
        self.cov_2 = api.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')
        self.maxpooling_1 = api.layers.MaxPool1D()
        self.maxpooling_2 = api.layers.MaxPool1D()

        self.lstm_1 = api.layers.LSTM(32, return_sequences=True)
        self.dense = api.layers.Dense(1)
        self.flatten = api.layers.Flatten()

    def call(self, inputs):
        x = self.lstm_1(inputs)
        x = self.cov_1(x)
        x = self.maxpooling_1(x)
        x = self.cov_2(x)
        x = self.maxpooling_2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x



class DSC_L(Model):
    def __init__(self, target_day, regre_len):
        super().__init__()
        self.scov_1 = layers.SeparableConv1D(32, kernel_size=3, padding='same', activation=hard_swish)
        self.maxpooling_1 = layers.MaxPool1D()

        self.scov_2 = layers.SeparableConv1D(16, kernel_size=3, padding='same', activation=hard_swish)
        self.maxpooling_2 = layers.MaxPool1D()

        self.lstm = layers.LSTM(16, return_sequences=True)

        self.shape_fitter = api.layers.SeparableConv1D(target_day, kernel_size=1)
        self.dense = layers.Dense(regre_len)

    def call(self, inputs):
        x = self.scov_1(inputs)
        x = self.maxpooling_1(x)

        x = self.scov_2(x)
        x = self.maxpooling_2(x)

        x = self.lstm(x)

        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.shape_fitter(x)
        x = tf.transpose(x, perm=[0, 2, 1])

        x = self.dense(x)
        return x


class DSC_GRU(Model):
    def __init__(self, target_day, regre_len):
        super().__init__()
        self.scov_1 = layers.SeparableConv1D(32, kernel_size=3, padding='same', activation=hard_swish)
        self.maxpooling_1 = layers.MaxPool1D()

        self.scov_2 = layers.SeparableConv1D(16, kernel_size=3, padding='same', activation=hard_swish)
        self.maxpooling_2 = layers.MaxPool1D()

        self.gru_1 = layers.GRU(16, return_sequences=True)

        self.shape_fitter = api.layers.SeparableConv1D(target_day, kernel_size=1)
        self.dense = layers.Dense(regre_len)

    def call(self, inputs):
        x = self.scov_1(inputs)
        x = self.maxpooling_1(x)

        x = self.scov_2(x)
        x = self.maxpooling_2(x)

        x = self.gru_1(x)

        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.shape_fitter(x)
        x = tf.transpose(x, perm=[0, 2, 1])

        x = self.dense(x)
        return x


class C_G(Model):
    def __init__(self, target_day, regre_len):
        super().__init__()
        self.cov_1 = layers.Conv1D(32, kernel_size=3, padding='same', activation=hard_swish)
        self.maxpooling_1 = layers.MaxPool1D()

        self.cov_2 = layers.Conv1D(16, kernel_size=3, padding='same', activation=hard_swish)
        self.maxpooling_2 = layers.MaxPool1D()

        self.gru_1 = layers.GRU(16, return_sequences=True)

        self.shape_fitter = api.layers.Conv1D(target_day, kernel_size=1)
        self.dense = layers.Dense(regre_len)

    def call(self, inputs):
        x = self.cov_1(inputs)
        x = self.maxpooling_1(x)

        x = self.cov_2(x)
        x = self.maxpooling_2(x)

        x = self.gru_1(x)

        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.shape_fitter(x)
        x = tf.transpose(x, perm=[0, 2, 1])

        x = self.dense(x)
        return x


class C_L(Model):
    def __init__(self, target_day, regre_len):
        super().__init__()
        self.cov_1 = layers.Conv1D(32, kernel_size=3, padding='same', activation=hard_swish)
        self.maxpooling_1 = layers.MaxPool1D()

        self.cov_2 = layers.Conv1D(16, kernel_size=3, padding='same', activation=hard_swish)
        self.maxpooling_2 = layers.MaxPool1D()

        self.lstm = layers.LSTM(16, return_sequences=True)

        self.shape_fitter = api.layers.Conv1D(target_day, kernel_size=1)
        self.dense = layers.Dense(regre_len)

    def call(self, inputs):
        x = self.cov_1(inputs)
        x = self.maxpooling_1(x)

        x = self.cov_2(x)
        x = self.maxpooling_2(x)

        x = self.lstm(x)

        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.shape_fitter(x)
        x = tf.transpose(x, perm=[0, 2, 1])

        x = self.dense(x)
        return x


class Splitting(api.Model):
    def __init__(self, num_heads):
        super(Splitting, self).__init__()
        self.num_heads = num_heads

    def apart(self, x):
        _ = []
        for i in range(self.num_heads):
            tensor = x[:, i::self.num_heads, :]
            # Check if tensor is empty, replace with previous tensor slice
            if tensor.shape[1] == 0:
                tensor = x[:, i - 1::self.num_heads, :]

            _.append(tensor)

        return _

    def back_padding(self, inputs):
        # inputs: (tensor1, tensor2, ... , tensorN), 4D tensors
        time_step_list = []
        for i in inputs:
            time_step_list.append(i.shape[1])
        min_len = min(time_step_list)
        padding_index = []
        for i in range(len(time_step_list)):
            if time_step_list[i] == min_len:
                padding_index.append(i)
        if len(padding_index) == len(time_step_list):
            return inputs
        else:
            for i in range(len(padding_index)):
                i_padding_index = padding_index[i]
                tensor_padding = tf.repeat(
                    tf.reshape(inputs[i_padding_index][:, -1, :], shape=(-1, 1, inputs[i_padding_index].shape[-1])),
                    repeats=1, axis=1)
                inputs[i_padding_index] = tf.concat([inputs[i_padding_index], tensor_padding], axis=1)
            return inputs


    def call(self, inputs):
        x = self.apart(inputs)
        x = self.back_padding(x)
        return x


class FeedForwardNetwork(api.Model):
    def __init__(self, dff_size, output_feature_len):
        super(FeedForwardNetwork, self).__init__()
        self.dense1 = api.layers.Dense(dff_size)
        self.dense2 = api.layers.Dense(output_feature_len)

    def call(self, x):
        x = self.dense1(x)
        x = hard_swish(x)
        x = self.dense2(x)
        return x


class Custom_DSC(api.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1, strides=1, padding='valid', activation='relu'):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.strides = strides

        self.sep_cov1d = api.layers.SeparableConv1D(filters=self.filters, kernel_size=self.kernel_size,
                                                    dilation_rate=self.dilation_rate, strides=self.strides,
                                                    padding='valid')

        if activation == 'relu':
            self.activation = api.activations.relu
        elif activation == 'leaky_relu':
            self.activation = leaky_relu
        else:
            self.activation = hard_swish

    def conv_padding(self, inputs):
        if self.padding == 'causal':
            num_padding = self.dilation_rate * (self.kernel_size - 1)
            tensor_padding = tf.repeat(
                tf.reshape(inputs[:, 0, :], shape=(-1, 1, inputs.shape[-1])),
                repeats=num_padding, axis=1)
            return tf.concat([tensor_padding, inputs], axis=1)

        elif self.padding == 'back':
            num_padding = self.dilation_rate * (self.kernel_size - 1)
            tensor_padding = tf.repeat(
                tf.reshape(inputs[:, -1, :], shape=(-1, 1, inputs.shape[-1])),
                repeats=num_padding, axis=1)
            return tf.concat([tensor_padding, inputs], axis=1)

        else:
            return inputs

    def call(self, inputs):
        padded_inputs = self.conv_padding(inputs)
        outputs = self.sep_cov1d(padded_inputs)
        outputs = self.activation(outputs)
        return outputs


class Resi(api.Model):
    def __init__(self, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
        self.avg_pooling = api.layers.AveragePooling1D(self.pool_size)

    def odd_to_even(self, inputs):
        time_steps = inputs.shape[1]
        if time_steps % 2 != 0:
            inputs = tf.concat([inputs,
                                tf.reshape(inputs[:, -1, :], shape=(-1, 1, inputs.shape[-1]))],
                               axis=1)
            return inputs
        else:
            return inputs

    def call(self, inputs):
        padding_inputs = self.odd_to_even(inputs)
        return self.avg_pooling(padding_inputs)


class CDSC_inner_base(api.Model):
    def __init__(self, filters, kernel_size, dilation_rate, activations, padding):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.activations = activations

        self.cov1d_1 = Custom_DSC(self.filters, self.kernel_size, dilation_rate=1, activation=self.activations,
                                  padding=padding, )  # Default padding = back
        self.pooling = api.layers.MaxPool1D()
        self.cov1d_3 = Custom_DSC(self.filters, self.kernel_size, dilation_rate=1, activation=self.activations,
                                  padding=padding, )  # Default padding = back


    def odd_to_even(self, inputs):
        time_steps = inputs.shape[1]
        if time_steps % 2 != 0:
            inputs = tf.concat([inputs,
                                tf.reshape(inputs[:, -1, :], shape=(-1, 1, inputs.shape[-1]))],
                               axis=1)
            return inputs
        else:
            return inputs

    def call(self, inputs):
        _1 = self.cov1d_1(inputs)
        _1 = self.odd_to_even(_1)
        _1 = self.pooling(_1)
        _1 = self.cov1d_3(_1)
        return _1


class CDSC_inner_block(api.Model):
    def __init__(self, filters, kernel_size, activations, padding):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activations = activations

        self.inner0 = CDSC_inner_base(self.filters, self.kernel_size, dilation_rate=1, activations=self.activations,
                                     padding=padding)
        self.inner1 = CDSC_inner_base(self.filters, self.kernel_size, dilation_rate=1, activations=self.activations,
                                     padding=padding)
        self.inner2 = CDSC_inner_base(self.filters, self.kernel_size, dilation_rate=1, activations=self.activations,
                                     padding=padding)

    def call(self, inputs):
        _0, _1, _2 = inputs
        _0 = self.inner0(_0)
        _1 = self.inner1(_1)
        _2 = self.inner2(_2)
        return [_0, _1, _2]


class CDSC_interactive(api.Model):
    def __init__(self, filters, kernel_size, interactive_dilation, activations, padding):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.interactive_dilation = interactive_dilation
        self.activations = activations
        self.padding = padding

        self.cov_dilation = Custom_DSC(self.filters, self.kernel_size, self.interactive_dilation,
                                       activation=self.activations, padding=self.padding[0])
        self.pooling = api.layers.MaxPool1D()

        self.cov1d_1 = Custom_DSC(self.filters, self.kernel_size, activation=self.activations,
                                  padding=self.padding[1])
        self.cov1d_2 = Custom_DSC(self.filters, self.kernel_size, activation=self.activations,
                                  padding=self.padding[1])
        self.cov1d_3 = Custom_DSC(self.filters, self.kernel_size, activation=self.activations,
                                  padding=self.padding[1])

    def odd_to_even(self, inputs):
        time_steps = inputs.shape[1]
        if time_steps % 2 != 0:
            inputs = tf.concat([inputs,
                                tf.reshape(inputs[:, -1, :], shape=(-1, 1, inputs.shape[-1]))],
                               axis=1)
            return inputs
        else:
            return inputs

    def call(self, inputs):
        _ = self.cov_dilation(inputs)
        _ = self.odd_to_even(_)
        _ = self.pooling(_)
        _1 = self.cov1d_1(_)
        _2 = self.cov1d_2(_)
        _3 = self.cov1d_3(_)
        return [_1, _2, _3]


class CDSC_block(api.Model):
    def __init__(self, filters, kernel_size, interactive_dilation, num_heads, activations,
                 padding):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.interactive_dilation = interactive_dilation
        self.num_heads = num_heads
        self.activations = activations

        self.dsc_inner = CDSC_inner_block(self.filters, self.kernel_size, self.activations,
                                         padding=padding[0])
        self.interactive = CDSC_interactive(self.filters, self.kernel_size, self.interactive_dilation,
                                           self.activations, padding=padding[1])

    def call(self, inputs):
        # Intra-learning
        _0, _1, _2 = self.dsc_inner(inputs)

        # Interactive learning
        __0, __1, __2 = inputs
        __ = tf.concat([__0, __1, __2], axis=1)
        __0, __1, __2 = self.interactive(__)

        # Add together
        _0 += __0
        _1 += __1
        _2 += __2

        return [_0, _1, _2]


class CDSC_encoder(api.Model):
    def __init__(self, filters, kernel_size, num_heads, x_len, padding, activations,
                 pool_size):
        super().__init__()
        self.filters_0 = filters[0]
        self.filters_1 = filters[1]
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.x_len = x_len
        self.activations = activations
        self.pool_size = pool_size
        self.interactive_dilation_1 = int(np.ceil(self.x_len / self.num_heads))
        self.interactive_dilation_2 = int(np.ceil(self.interactive_dilation_1 / self.pool_size))

        self.splitor = Splitting(self.num_heads)
        self.block_1 = CDSC_block(self.filters_0, self.kernel_size, self.interactive_dilation_1,
                                 self.num_heads,  activations=self.activations,
                                 padding=padding)
        self.block_2 = CDSC_block(self.filters_1, self.kernel_size, self.interactive_dilation_2,
                                 self.num_heads,  self.activations, padding=padding)

        self.resi_1 = Resi(pool_size=self.pool_size)
        self.resi_2 = Resi(pool_size=self.pool_size)

    # Merge and rearrange convolution heads into one tensor, Input: [tensor1, tensor2, ... , tensorN], Output: tensor(B,T,D)
    def relignment(self, inputs):

        _0, _1, _2 = inputs

        tensor_list = []
        for i in range(_0.shape[1]):
            tensor = tf.concat([_0[:, i:i + 1, :], _1[:, i:i + 1, :], _2[:, i:i + 1, :]], axis=1)
            tensor_list.append(tensor)

        # Concatenate shapes to restore
        outputs = tensor_list[0]
        for i in range(1, len(tensor_list)):
            outputs = tf.concat([outputs, tensor_list[i]], axis=1)

        return outputs

    def call(self, inputs):
        # inputs (B,T,D)
        # Split convolution heads, Input tensor: (B,T,D), Output: [tensor1, tensor2, ... , tensorN]
        _ = self.splitor(inputs)
        _ = self.block_1(_)

        # resi_1
        r_1 = self.resi_1(inputs)
        resi_head_1 = self.splitor(r_1)  # [tensor1, tensor2, ... , tensorN]

        _0 = _[0] + resi_head_1[0]  # (B,T1,D1)
        _1 = _[1] + resi_head_1[1]  # (B,T1,D1)
        _2 = _[2] + resi_head_1[2]  # (B,T1,D1)

        # block_2
        _ = self.block_2([_0, _1, _2])  # [tensor1, tensor2, ... , tensorN]

        # resi_2
        r_2 = self.resi_2(r_1)  # (B,T2,D2)
        resi_head_2 = self.splitor(r_2)  # [tensor1, tensor2, ... , tensorN]
        _0 = _[0] + resi_head_2[0]  # (B,T2,D2)
        _1 = _[1] + resi_head_2[1]  # (B,T2,D2)
        _2 = _[2] + resi_head_2[2]  # (B,T2,D2)

        # tensor reshape
        relign_outputs = self.relignment([_0, _1, _2])

        return relign_outputs


class CDSC_decoder(api.Model):
    def __init__(self, dff_size, y_len, output_feature_len):
        super().__init__()
        self.dff_size = dff_size
        self.y_len = y_len
        self.output_feature_len = output_feature_len
        self.shape_fiter = api.layers.SeparableConv1D(filters=self.y_len, kernel_size=1, padding='valid')

        if y_len == 1:
            self.ffn = FeedForwardNetwork(dff_size=self.dff_size, output_feature_len=self.output_feature_len)
        else:
            self.dense1 = api.layers.Dense(self.dff_size)
            dense_num = (y_len - 1) // 4  # Excluding the last step
            remainder = y_len - 4 * dense_num
            self.dense_dict = {}
            n = 0
            for i in range(dense_num):
                self.dense_dict[i] = api.layers.Dense(self.output_feature_len)  # Each dense connected with same step size
                n += 1
            if remainder != 0:
                self.dense_dict[n] = api.layers.Dense(self.output_feature_len)
            self.dense_dict[n + 1] = api.layers.Dense(self.output_feature_len)  # Last step connected to a separate dense

    def call(self, inputs):
        # inputs (B,L,D)
        # x (B,D,L)
        x = tf.transpose(inputs, perm=[0, 2, 1])
        x = self.shape_fiter(x)
        # x  (B,L,D)
        x = tf.transpose(x, perm=[0, 2, 1])
        if self.y_len == 1:
            x = self.ffn(x)
        else:
            x = self.dense1(x)
            keys = list(self.dense_dict.keys())
            tensor_list = []
            for n, i in enumerate(keys):
                if n < len(keys) - 2:
                    tensor_list.append(self.dense_dict[i](x[:, 4 * n:4 * (n + 1), :]))
                elif n == len(keys) - 2:
                    tensor = self.dense_dict[i](x[:, 4 * n:self.y_len - 1, :])
                    tensor_list.append(tensor)
                else:
                    tensor_list.append(self.dense_dict[i](x[:, self.y_len - 1:, :]))
            x = tf.concat(tensor_list, axis=1)

        return x


class CDSC_net(api.Model):
    def __init__(self, filters, kernel_size, num_heads, x_len, dff_size, y_len, output_feature_len,
                 padding, activations, pool_size):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.x_len = x_len
        self.padding = padding
        self.interactive_dilation = np.ceil(self.x_len / self.num_heads)

        self.dff_size = dff_size  # 前馈网络中间层单元数
        self.y_len = y_len  # 预测时间步长
        self.output_feature_len = output_feature_len  # 输出特征数

        self.activations = activations
        self.pool_size = pool_size

        self.encoder = CDSC_encoder(self.filters, self.kernel_size, self.num_heads,
                                   self.x_len, self.padding, self.activations, self.pool_size)
        self.decoder = CDSC_decoder(self.dff_size, self.y_len, self.output_feature_len)

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x


def single_step_prdition(model, data_scaler, train_X, train_y, test_X, test_y, org_data, regre_len, block = True):
    pred_label = model.predict(train_X)
    pred_label = data_scaler.inverse_transform(pred_label.reshape(-1, regre_len))
    label = data_scaler.inverse_transform(train_y.reshape(-1, regre_len))
    plt.plot(pred_label, label='pred')
    plt.plot(label, label='true')
    plt.title('Prediction')
    plt.legend()

    if block:
        plt.show()
    else:
        plt.show(block=block)
        plt.pause(1)
        plt.close()

    pred = model.predict(test_X)
    pred = data_scaler.inverse_transform(pred.reshape(-1, regre_len))
    y = data_scaler.inverse_transform(test_y.reshape(-1, regre_len))
    plt.plot(pred, label='pred')
    plt.plot(y, label='true')
    plt.title('Prediction')
    plt.legend()

    if block:
        plt.show()
    else:
        plt.show(block=block)
        plt.pause(1)
        plt.close()

    print('Mape：', mean_ape(y, pred)*100, '%')

    org_x = np.arange(1, len(org_data) + 1)
    pred_x = np.arange(len(org_x) + 1 - len(pred), len(org_data) + 1)

    plt.plot(org_x, org_data, label='true')
    plt.plot(pred_x, pred, label='pred')
    plt.axvline(len(org_data) - len(pred) + 1, linestyle=(0, (5, 2, 1, 2)),
                linewidth=2)
    plt.legend()

    if block:
        plt.show()
    else:
        plt.show(block=block)
        plt.pause(1)
        plt.close()

    true_data = np.concatenate([label, y], axis=0)
    pred_data = np.concatenate([pred_label, pred], axis=0)
    return true_data, pred_data, pred, y, mean_ape(y, pred)



if __name__ == "__main__":

    '''
    ================================================
                                        1.Data process
    ===============================================
    '''

    num = 229
    attributes = ['Cumulative_recovered']
    file = ['india']
    file_name = [i + '.xlsx' for i in file]
    dir_name = os.path.join('datasets', 'Case_1')
    data = get_data(dir_name, file_name, num, attributes, )

    # ============ Visualization =============#

    cols = 1
    rows = 1
    max_figure_num = 1
    block = False

    training_suptitle = 'Training data'
    training_figure_title = 'Training_data'
    training_label = 'Training_data'
    data_visualization(data, suptitle=training_suptitle, title=training_figure_title,
                       data_label=training_label, max_figure_num=max_figure_num)

    scaled_training_suptitle = 'Scaled_training data'
    scaled_training_figure_title = 'Scaled_training_data'
    scaled_training_label = 'Scaled_training_data'
    scaled_training_data, data_scaler, feature_scaler = training_normalization(data,
                                                                               suptitle=scaled_training_suptitle,
                                                                               title=scaled_training_figure_title,
                                                                               data_label=scaled_training_label,
                                                                               max_figure_num=max_figure_num)

    '''
    ===========================================
                                         2.Models
    ==========================================
    '''

    hard_swish = api.activations.hard_swish
    leaky_relu = api.activations.leaky_relu

    regre_len = 1  # feature length
    num_steps = 7  # widow size
    target_day = 1  # prediction step
    attributes_len = len(attributes)

    train_X, train_y, test_X, test_y = data_format(scaled_training_data[0], num_steps, target_day, attributes_len,
                                                   regre_len, test_ratio=0.414)

    epoch = 200

    rmse_list = np.array([])
    mape_list = np.array([])

    run_times = 10

    filters = [32, 16]
    kernel_size = 3
    num_heads = 3
    dff_size = 16
    activations = 'no_relu'
    padding = ['back', [False, 'back']]

    for i in range(run_times):
        # train_y = train_y.reshape(-1,target_day)
        # base_model = LSTM_CNN()
        # model = CNN_LSTM()

        model = CDSC_net(filters, kernel_size, num_heads, x_len=num_steps, dff_size=dff_size,
                         y_len=target_day, output_feature_len=regre_len,
                         padding=padding, activations='hardswish',
                         pool_size=2)

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.002),
            loss='mse',
        )

        history = model.fit(train_X, train_y, epochs=epoch, batch_size=32, )

        training_loss = history.history['loss']
        plt.plot(training_loss, label='training_loss')
        plt.title('Training')
        plt.legend()

        if block:
            plt.show()
        else:
            plt.show(block=block)
            plt.pause(1)
            plt.close()

        org_data = data[file_name[0]][attributes[:1]]

        excel_location = r'C:\Users\HelloWorld\Desktop\data'
        excel_name = f'data_{i}.xlsx'
        excel_path = os.path.join(excel_location, excel_name)
        writer = pd.ExcelWriter(excel_path, engine='openpyxl')

        true_data, pred_data, pred_on_test, label_on_test, mean_ape_value = single_step_prdition(model, data_scaler, train_X, train_y, test_X, test_y, org_data, regre_len, block=block)

        mape_list = np.append(mape_list, mean_ape_value)

        data_to_excel_whole = np.concatenate([true_data, pred_data], axis=-1)
        data_to_excel_whole = pd.DataFrame(data_to_excel_whole, columns=['label(on train and test)',
                                                                         'predicted_value(on train and test)'])
        data_to_excel_whole.to_excel(writer, sheet_name='Sheet1')
        data_to_excel_test = np.concatenate([label_on_test, pred_on_test], axis=-1)
        data_to_excel_test = pd.DataFrame(data_to_excel_test, columns=['label(on test)',
                                                                       'predicted_value(on test)'])
        data_to_excel_test.to_excel(writer, sheet_name='Sheet2')

        if_cumulative_data = True
        if if_cumulative_data:
            cumulative_ratio_list = np.ones(7)/7
            new_cases = data_to_excel_whole.diff().dropna()
            new_cases_other = data_to_excel_whole.diff().dropna()
            original_cases_label = new_cases.iloc[:, :1]
            original_cases_label.columns = ['original daily cases']
            original_predicted_value = new_cases.iloc[:, 1:]
            original_predicted_value.columns = ['original predicted value']
            country_smooth(new_cases_other, [0, 1], cumulative_ratio_list)
            new_cases_other.columns = ['smoothed daily cases', 'smoothed predicted value']
            final_data = pd.concat([original_cases_label, original_predicted_value, new_cases_other], axis=1)
            final_data.to_excel(writer, sheet_name='Sheet3')

        writer.close()

    '''
    
    ===========================================
                                        3.Results
    ===========================================

    '''

    mape_list = np.append(mape_list, np.mean(mape_list))*100

    figure, axs = plt.subplots(1, 2, constrained_layout=True)
    figure.suptitle('Metric')

    x = np.arange(1, len(mape_list))
    x = np.append(x, 'average')

    ax_2 = axs[1]
    ax_2.set_title('mape')
    bar_2 = ax_2.bar(x, mape_list)
    ax_2.bar_label(bar_2, fmt='%.3f')

    plt.show()
