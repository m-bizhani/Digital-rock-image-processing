import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, Lambda, Input

def res_block(x, f, res_block_scaling=None):

  x_skip = x
  x = Conv2D(f, 3, strides=(1,1), padding='same', activation='relu')(x)
  x = Conv2D(f, 3, strides=(1,1), padding='same')(x)

  if res_block_scaling:
    x = Lambda(lambda t: t*res_block_scaling)(x)
  x = Add()([x, x_skip])
  return x


def edsr(scale, f, num_of_res_blocks=8, res_block_scaling=None):

  x_in = Input(shape=(None,None,1))
  x_skip = x = Conv2D(f, 3, strides=(1,1), padding='same')(x_in)

  for i in range(num_of_res_blocks):
    x = res_block(x, f, res_block_scaling)

  x = Conv2D(f, 3, strides=(1,1), padding='same')(x)
  x = Add()([x, x_skip])

  if scale ==2:
    x = Conv2D(f * (scale ** 2), 3, padding='same')(x)
    x = tf.nn.depth_to_space(x, scale)

  if scale ==3:
    x = Conv2D(f * (scale ** 2), 3, padding='same')(x)
    x = tf.nn.depth_to_space(x, scale)

  if scale ==4:
    x = Conv2D(f * (2 ** 2), 3, padding='same')(x)
    x = tf.nn.depth_to_space(x, 2)
    x = Conv2D(f * (2 ** 2), 3, padding='same')(x)
    x = tf.nn.depth_to_space(x, 2)

  
  x_out = Conv2D(1, 3, strides=(1,1), padding='same')(x)
  
  model = tf.keras.Model(inputs=x_in, outputs=x_out, name='EDSR')

  return model
