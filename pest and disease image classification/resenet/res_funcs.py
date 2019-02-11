from res_packages import *

LRELU = LeakyReLU(alpha = 0.01)

def identity_block(img_input, kernel_size, filters):
    filters1, filters2, filters3 = filters

    x = Conv2D(filters1, (1, 1))(img_input)
    x = BatchNormalization(axis = -1)(x)
    x = Activation('relu')(x)
    #x = Activation(LRELU)(x)

    x = Conv2D(filters2, kernel_size, padding = 'same')(x)
    x = BatchNormalization(axis = -1)(x)
    x = Activation('relu')(x)
    #x = Activation(LRELU)(x)

    x = Conv2D(filters3, (1, 1))(x)
    x = BatchNormalization(axis = -1)(x)
    x = layers.add([x, img_input])
    x = Activation('relu')(x)
    #x = Activation(LRELU)(x)
    return x

def convolution_block(img_input, kernel_size, filters, strides = (2, 2)):
    filters1, filters2, filters3 = filters

    x = Conv2D(filters1, (1, 1), strides = strides)(img_input)
    x = BatchNormalization(axis = -1)(x)
    x = Activation('relu')(x)
    #x = Activation(LRELU)(x)

    x = Conv2D(filters2, kernel_size, padding = 'same')(x)
    x = BatchNormalization(axis = -1)(x)
    x = Activation('relu')(x)
    #x = Activation(LRELU)(x)

    x = Conv2D(filters3, (1, 1))(x)
    x = BatchNormalization(axis = -1)(x)

    shortcut = Conv2D(filters3, (1, 1), strides = strides)(img_input)
    shortcut = BatchNormalization(axis = -1)(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    #x = Activation(LRELU)(x)
    return x

def resnet50(n_classes, img_input, include_top = True, pooling = None, input_tensor = None):
    x = Conv2D(64, (7, 7), strides = (2, 2))(img_input)
    x = MaxPooling2D((3, 3), strides = (2, 2))(x)

    # Conv2_x
    x = convolution_block(img_input = x, kernel_size = 3, filters = [64, 64, 256], strides = (1, 1))
    x = identity_block(img_input = x, kernel_size = 3, filters = [64, 64, 256])
    x = identity_block(img_input = x, kernel_size = 3, filters = [64, 64, 256])

    # Conv3_x
    x = convolution_block(img_input = x, kernel_size = 3, filters = [128, 128, 512])
    x = identity_block(img_input = x, kernel_size = 3, filters = [128, 128, 512])
    x = identity_block(img_input = x, kernel_size = 3, filters = [128, 128, 512])
    x = identity_block(img_input = x, kernel_size = 3, filters = [128, 128, 512])

    # Conv4_x
    x = convolution_block(img_input = x, kernel_size = 3, filters = [256, 256, 1024])
    x = identity_block(img_input = x, kernel_size = 3, filters = [256, 256, 1024])
    x = identity_block(img_input = x, kernel_size = 3, filters = [256, 256, 1024])
    x = identity_block(img_input = x, kernel_size = 3, filters = [256, 256, 1024])
    x = identity_block(img_input = x, kernel_size = 3, filters = [256, 256, 1024])
    x = identity_block(img_input = x, kernel_size = 3, filters = [256, 256, 1024])

    # Conv5_x
    x = convolution_block(img_input = x, kernel_size = 3, filters = [512, 512, 2048])
    x = identity_block(img_input = x, kernel_size = 3, filters = [512, 512, 2048])
    x = identity_block(img_input = x, kernel_size = 3, filters = [512, 512, 2048])

    x = AveragePooling2D((7, 7))(x)

    if include_top:
        x = Dense(n_classes, activation = 'softmax')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x)
    return model