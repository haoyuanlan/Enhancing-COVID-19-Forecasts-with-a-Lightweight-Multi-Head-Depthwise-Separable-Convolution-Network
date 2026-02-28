
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import keras.api as api
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

# Activation functions
hard_swish = api.activations.hard_swish
leaky_relu = api.activations.leaky_relu

def get_flops(model, batch_size=1, input_shape=None):
    """
    Calculate FLOPs for Keras model.
    Args:
        model: model instance
        batch_size: batch size
        input_shape: input shape (excluding batch dimension). If provided, use directly; otherwise try to get from model attributes.
    Returns:
        float_ops: total floating point operations
    """

    if input_shape is not None:
        # Construct dummy input using provided input_shape (assuming single input)
        dummy_input = tf.zeros((batch_size,) + input_shape, dtype=tf.float32)
    else:
        # Attempt to get input shape from model attributes (compatible with standard Keras)
        try:
            if hasattr(model, 'inputs') and model.inputs:
                dummy_inputs = []
                for input_layer in model.inputs:
                    shape = input_layer.shape.as_list()
                    shape[0] = batch_size
                    dummy_inputs.append(tf.zeros(shape, dtype=input_layer.dtype))
                dummy_input = dummy_inputs if len(dummy_inputs) > 1 else dummy_inputs[0]
            else:
                input_shapes = model.input_shape
                if not isinstance(input_shapes, list):
                    input_shapes = [input_shapes]
                dummy_inputs = []
                for shape in input_shapes:
                    shape_list = list(shape)
                    shape_list[0] = batch_size
                    dummy_inputs.append(tf.zeros(shape_list, dtype=tf.float32))
                dummy_input = dummy_inputs if len(dummy_inputs) > 1 else dummy_inputs[0]
        except AttributeError:
            raise ValueError("Unable to get input shape from model, please provide input_shape parameter explicitly")

    # get concrete function
    real_model = tf.function(model).get_concrete_function(dummy_input)

    # Freeze graph (remove training-related operations), return GraphDef
    frozen_func, frozen_graph_def = convert_variables_to_constants_v2_as_graph(real_model)

    # Import GraphDef into a new Graph
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(frozen_graph_def, name="")
        # Use tf.profiler to count floating point operations
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops_stat = tf.compat.v1.profiler.profile(
            graph=graph,
            run_meta=run_meta,
            cmd="op",
            options=opts
        )
        return flops_stat.total_float_ops


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


x_len = 7  # window size
y_len = 1  # prediction step
output_feature_len = 1

filters = [16, 8]
kernel_size = 3
num_heads = 3
dff_size = 16
activations = 'no_relu'
padding = ['back', [False, 'back']]

model = CDSC_net(filters, kernel_size, num_heads, x_len, dff_size, y_len, output_feature_len,
                padding, activations='hardswish', pool_size=2)

x = np.zeros((32, x_len, output_feature_len))

model(x)

model.summary()

flops = get_flops(model, batch_size=32, input_shape=(x_len, output_feature_len))
print(f"FLOPs: {flops:.2e}")
print(f"GFLOPs: {flops / 1e9:.3f}")
