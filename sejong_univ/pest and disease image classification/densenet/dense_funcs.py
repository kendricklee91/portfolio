from dense_packages import *

LRELU = LeakyReLU(alpha = 0.01)
PRELU = PReLU()

def relu(x):
    return Activation('relu')(x)

def lrelu(x):
    return Activation(LRELU)(x) # Leaky ReLU

def prelu(x):
    return Activation(PRELU)(x) # Parametric ReLU

def dropout(x, dropout_rate):
    return Dropout(dropout_rate)(x)

def bn(x, weight_decay):
    return BatchNormalization(mode = 0, axis = -1, gamma_regularizer = l2(weight_decay), beta_regularizer = l2(weight_decay))(x) # 데이터 포맷에서 채널이 마지막이면 axis = -1, 제일 첫 번째 위치이면 axis = 1

def relu_bn(x, weight_decay):
    return relu(bn(x, weight_decay))

def lrelu_bn(x, weight_decay):
    return lrelu(bn(x, weight_decay))

def prelu_bn(x, weight_decay):
    return prelu(bn(x, weight_decay))

def convolution(x, n_filter, img_size, weight_decay):
    return Convolution2D(n_filter, img_size, img_size, init = 'he_uniform', border_mode = 'same', W_regularizer = l2(weight_decay))(x) # border_mode = padding
    #return Conv2D(n_filter, (img_size, img_size), kernel_initializer= 'he_uniform', padding = 'same', kernel_regularizer = l2(weight_decay), data_format = 'channels_last')(x)
    #return Conv2D(n_filter, img_size, img_size, padding = 'same', data_format = 'channels_last', kernel_initializer = 'he_uniform', kernel_regularizer = l2(weight_decay))(x)

def convolution_block(img_input, n_filter, bottleneck = False, dropout_rate = None, weight_decay = 0):
    #x = relu_bn(img_input, weight_decay)
    x = lrelu_bn(img_input, weight_decay)
    #x = prelu_bn(img_input, weight_decay)

    if bottleneck:
        x = convolution(x = x, n_filter = n_filter * 4, img_size = 1, weight_decay = weight_decay)
        x = dropout(x = x, dropout_rate = dropout_rate)
        x = relu_bn(x = x, weight_decay = weight_decay)
        #x = lrelu_bn(x = x, weight_decay = weight_decay)
        #x = prelu_bn(x = x, weight_decay = weight_decay)

    x = convolution(x = x, n_filter = n_filter, img_size = 3, weight_decay = weight_decay) # 3x3 convolution layer 16개
    return dropout(x = x, dropout_rate = dropout_rate)

def transition_block(img_input, n_filter, compression = 1.0, dropout_rate = None, weight_decay = 0):
    #x = relu_bn(img_input, weight_decay = weight_decay)
    x = lrelu_bn(img_input, weight_decay = weight_decay)
    #x = prelu_bn(img_input, weight_decay = weight_decay)

    x = convolution(x = x, n_filter = int(n_filter * compression), img_size = 1, weight_decay = weight_decay)
    x = dropout(x = x, dropout_rate = dropout_rate)
    return AveragePooling2D(pool_size = (2, 2), strides = (2, 2))(x)

def dense_block(x, n_layers, n_filters, growth_rate, bottleneck = False, dropout_rate = None, weight_decay = 0, grow_n_filters = True):
    for i in range(n_layers):
        a = convolution_block(img_input = x, n_filter = growth_rate, bottleneck = bottleneck, dropout_rate = dropout_rate, weight_decay = weight_decay)
        x = merge([x, a], mode = 'concat', concat_axis = -1)

        if grow_n_filters:
            n_filters += growth_rate # n_filer + growth_rate
    return x, n_layers

# DenseNet Model
def densenet(n_classes, img_input, depth = 40, n_dense_block = 3, growth_rate = 12, n_filter = -1, n_layers_per_block = 1,
             bottleneck = False, reduction = False, dropout_rate = None, weight_decay = 0, activation = 'softmax'):

    assert (depth - 4) % 3 == 0, "Depth must be 3N + 4"

    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0

    if n_layers_per_block == 1:
        count = int((depth - 4) / 3) # 12
        n_layers = [count for _ in range(n_dense_block)] # n_layers = [12, 12, 12]
        final_n_layer = count # 12

    if bottleneck:
        n_layers = [int(layer // 2) for layer in n_layers] # n_layers = [6, 6, 6]

    if n_filter <= 0:
        n_filter = 2 * growth_rate # n_filter = 24

    compression = 1.0 - reduction

    x = convolution(x = img_input, n_filter = n_filter, img_size = 3, weight_decay = weight_decay) # 3 x 3 convolution

    # dense block 2 개 (n_dense_block idx : 0, 1)
    for block_idx in range(n_dense_block - 1):
        x, n_filter = dense_block(x = x, n_layers = n_layers[block_idx], n_filters = n_filter, growth_rate = growth_rate, bottleneck = bottleneck, dropout_rate = dropout_rate, weight_decay = weight_decay)
        x = transition_block(img_input = x, n_filter = n_filter, compression = compression, dropout_rate = dropout_rate, weight_decay = weight_decay)
        n_filter = int(n_filter * compression)


    x, n_filter = dense_block(x = x, n_layers = final_n_layer, n_filters = n_filter, growth_rate = growth_rate, bottleneck = bottleneck, dropout_rate = dropout_rate, weight_decay = weight_decay)

    #x = relu_bn(x = x, weight_decay = weight_decay)
    #x = lrelu_bn(x = x, weight_decay = weight_decay)
    #x = prelu_bn(x = x, weight_decay = weight_decay)

    x = GlobalAveragePooling2D()(x)
    return Dense(n_classes, activation = activation, W_regularizer = l2(weight_decay), b_regularizer = l2(weight_decay))(x)