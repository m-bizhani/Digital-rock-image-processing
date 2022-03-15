import tensorflow as tf
from tensorflow.keras import layers, Input, Model, callbacks, optimizers, losses
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

N = 48      ##Initial number of features 

#--------------------------------------------------------------------------------#
def DUB(x):
    conv1a = layers.Conv2D(N*2, (3,3), strides=(1,1), padding='same')(x)
    conv1a = layers.ReLU()(conv1a)

    conv2a = layers.Conv2D(N*2, (3,3), strides=(1,1), padding='same')(conv1a)
    conv2a = layers.ReLU()(conv2a)

    conv3a = layers.Conv2D(N*2, (3,3), strides=(1,1), padding='same')(conv2a)
    conv3a = layers.ReLU()(conv3a)

    add1a = layers.Add()([conv1a, conv3a])
    
    conv1b = layers.Conv2D(N*4, (3,3), strides=(2,2), padding='same')(add1a)
    # conv1b = layers.PReLU()(conv1b)

    conv2b = layers.Conv2D(N*4, (3,3), strides=(1,1), padding='same')(conv1b)
    conv2b = layers.PReLU()(conv2b)

    add1b = layers.Add()([conv1b, conv2b])
    
    conv1c = layers.Conv2D(N*8, (3,3), strides=(2,2), padding='same')(add1b)
    # conv1c = layers.PReLU()(conv1c)

    conv2c = layers.Conv2D(N*8, (3,3), strides=(1,1), padding='same')(conv1c)
    conv2c  = layers.PReLU()(conv2c)

    add1c = layers.Add()([conv1c, conv2c])

    conv3c = layers.Conv2D(N*16, (1,1), strides=(1,1), padding='same')(add1c)

    upsample1 = tf.nn.depth_to_space(conv3c,2)

    conc1b = layers.Concatenate()([add1b, upsample1])

    conv1d = layers.Conv2D(N*4, (1,1), strides=(1,1), padding='same')(conc1b)
    # conv1d = layers.PReLU()(conv1d)

    conv2d = layers.Conv2D(N*4, (3,3), strides=(1,1), padding='same')(conv1d)
    conv2d = layers.PReLU()(conv2d)

    conv3d = layers.Conv2D(N*4, (3,3), strides=(1,1), padding='same')(conv2d)
    conv3d = layers.PReLU()(conv3d)

    add3d = layers.Add()([conv2d, conv3d])

    conv4d = layers.Conv2D(N*8, (1,1), strides=(1,1), padding='same')(add3d)
    # conv4d = layers.PReLU()(conv4d)

    upsample2 = tf.nn.depth_to_space(conv4d,2)

    conc1a = layers.Concatenate()([add1a, upsample2])

    conv1e = layers.Conv2D(N*2, (1,1), strides=(1,1), padding='same')(conc1a)
    # conv1e = layers.PReLU()(conv1e)

    conv2e = layers.Conv2D(N*2, (3,3), strides=(1,1), padding='same')(conv1e)
    conv2e = layers.PReLU()(conv2e)

    conv3e = layers.Conv2D(N*2, (3,3), strides=(1,1), padding='same')(conv2e)
    conv3e = layers.PReLU()(conv3e)
    
    conv4e = layers.Conv2D(N*2, (3,3), strides=(1,1), padding='same')(conv3e)
    conv4e = layers.PReLU()(conv4e)
    
    add1e = layers.Add()([conv2e, conv4e])
    conv5e = layers.Conv2D(N*2, (3,3), strides=(1,1), padding='same')(add1e)
    
    conv5e = layers.PReLU()(conv5e)
    add2e = layers.Add()([conv5e, conv1a])

    return add2e
#--------------------------------------------------------------------------------#
def reconstruction(x):

    l1 =  layers.Conv2D(N*2, (3,3), strides=(1,1), padding='same')(x)
    l1 = layers.PReLU()(l1)
    l1 = layers.Conv2D(N*2, (3,3), strides=(1,1), padding='same')(l1)
    l1 = layers.PReLU()(l1)

    s1 = layers.Add()([x, l1])

    l2 =  layers.Conv2D(N*2, (3,3), strides=(1,1), padding='same')(s1)
    l2 = layers.PReLU()(l2)
    l2 = layers.Conv2D(N*2, (3,3), strides=(1,1), padding='same')(l2)
    l2 = layers.PReLU()(l2)

    s2 = layers.Add()([s1, l2])

    l3 =  layers.Conv2D(N*2, (3,3), strides=(1,1), padding='same')(s2)
    l3 = layers.PReLU()(l3)
    l3 = layers.Conv2D(N*2, (3,3), strides=(1,1), padding='same')(l3)
    l3 = layers.PReLU()(l3)

    s3 = layers.Add()([s2, l3])

    l4 =  layers.Conv2D(N*2, (3,3), strides=(1,1), padding='same')(s3)
    l4 = layers.PReLU()(l4)
    l4 = layers.Conv2D(N*2, (3,3), strides=(1,1), padding='same')(l4)
    l4 = layers.PReLU()(l4)

    s3 = layers.Add()([s3, l4])

    l5 = layers.Conv2D(N*2, (3,3), strides=(1,1), padding='same')(s3)

    return l5

#--------------------------------------------------------------------------------#
def get_model(input_shape):

    model_input = Input(shape=input_shape)
#     pad = tf.keras.layers.ZeroPadding2D((2,1))(model_input)
    conv1 = layers.Conv2D(N, 3, padding='same')(model_input)
    conv1 = layers.PReLU()(conv1)

    conv2 = layers.Conv2D(N, 3, strides=(2,2), padding='same')(conv1)
    conv2 = layers.PReLU()(conv2)

    ###Down-up-sample block
    DUB1 = DUB(conv2)
    DUB2 = DUB(DUB1)
#     DUB3 = DUB(DUB2)
#     DUB4 = DUB(DUB3)

    r1 = reconstruction(DUB1)
    r2 = reconstruction(DUB2)
#     r3 = reconstruction(DUB3)
#     r4 = reconstruction(DUB4)

    c1 = layers.Concatenate()([r1, r2])

    c2 = layers.Conv2D(N*2, (1,1), strides=(1,1), padding='same')(c1)

    c3 = layers.Conv2D(N*2, (1,1), strides=(1,1), padding='same')(c2)
    c3 = layers.PReLU()(c3)
    c4= layers.Conv2D(N*2, (1,1), strides=(1,1), padding='same')(c3)
    c4 = layers.PReLU()(c4)

    a1 = layers.Add()([c3, c4])

    upsample3  = tf.nn.depth_to_space(a1 ,2)

    c5= layers.Conv2D(N//2, (1,1), strides=(1,1), padding='same')(upsample3)
    c5 = layers.PReLU()(c5)

    model_output = layers.Conv2D(1, (3,3), padding='same')(c5)
    return Model(model_input, model_output)


if __name__ == '__main__':
    model  = get_model(input_shape=(128, 128, 1))
    print(model.summary())