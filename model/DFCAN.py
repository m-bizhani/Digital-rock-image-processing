#%%
import tensorflow as tf
from tensorflow.keras import layers, activations, Input, Model, utils
from math import log

def FCAB(x):
    x_skip = x
    
    x = layers.Conv2D(64, 3, strides=1, padding='same')(x)
    x = activations.gelu(x)
    x = layers.Conv2D(64, 3, strides=1, padding='same')(x)
    x = activations.gelu(x)
    x_2 = x
    x = layers.Lambda(lambda v: tf.cast(tf.signal.rfft2d(v), tf.float32))(x)
    x = layers.Conv2D(64, 3, strides=1, padding='same')(x)
    x = activations.relu(x)
    s = x.shape
    x = layers.GlobalAvgPool2D()(x)
#     x = layers.GlobalMaxPool2D()(x)
#     x = layers.GlobalAvgPool2D(keepdims=True)(x)
    x = layers.Reshape((1, 1, s[-1]))(x)
    # print(x.shape)
    x = layers.Conv2D(4, 1, strides=1, padding='same')(x)
    x = activations.relu(x)
    x = layers.Conv2D(64, 1, strides=1, padding='same')(x)
    x = activations.sigmoid(x)
    
    x = layers.Multiply()([x, x_2])
    x = layers.Add()([x, x_skip])
    
    return x

def res_blocks(x, num_res = 4):
    x_skip = x
    
    for _ in range(num_res):
        x = FCAB(x)
    x = layers.Add()([x, x_skip])
    
    return x

def pixel_shuffle(x, scale=4):
    
    for _ in range(int(log(scale, 2))):
        x = tf.nn.depth_to_space(x, block_size=2)
        
    return x

def DFCAN(input_shape, num_res_blocks = 4, num_res = 4, scale=4):
    x_in = Input(shape = input_shape)
    
    x = layers.Conv2D(64, 3, strides=1, padding='same')(x_in)
    x = activations.gelu(x)
    
    for _ in range(num_res_blocks):
        x = res_blocks(x, num_res = num_res)
    
    x = layers.Conv2D(256, 3, strides=1, padding='same')(x)
    x = activations.gelu(x)
    x = pixel_shuffle(x, scale=scale)
    x = layers.Conv2D(1, 3, strides=1, padding='same')(x)
    x_out = activations.sigmoid(x)
    
    return Model(x_in, x_out)

if __name__ == '__main__':
    model = DFCAN()
    print(model.summary())

# %%
