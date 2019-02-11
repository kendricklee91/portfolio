def VGG16(n_classes = None, img_input = None, dropout_rate = 0.0):

    # Block 1
    x = Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(img_input)
    x = Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)

    # Block 2
    x = Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
    x = Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)

    # Block 3
    x = Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
    x = Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
    x = Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)

    # Block 4
    x = Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
    x = Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
    x = Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)

    # Block 5
    x = Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
    x = Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
    x = Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)

    # Classification block
    x = Flatten()(x)
    x = Dense(units = 4096, activation = 'relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(units = 4096, activation = 'relu')(x)
    x = Dropout(dropout_rate)(x)
    return Dense(n_classes, activation = 'softmax')(x)
