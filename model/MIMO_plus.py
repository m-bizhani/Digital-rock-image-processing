#%%
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

def BasicConv(x, f, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
    
    if bias and norm:
        bias = False
        
    padding = kernel_size //2
    
    if transpose:
        padding = (kernel_size // 2) - 1
        # x = layers.ZeroPadding2D(padding=(padding,padding))(x)
        x = layers.Conv2DTranspose(f, kernel_size, padding='same', strides=stride, use_bias=bias)(x)
    else:
        # x = layers.ZeroPadding2D(padding=(padding,padding))(x)
        x = layers.Conv2D(f, kernel_size, padding='same', strides=stride, use_bias=bias)(x)
    if norm:
        x = layers.BatchNormalization()(x)
    if relu:
        x = layers.ReLU()(x)
        
    return x

def ResBlock(x, f):
    
    x = BasicConv(x, f, kernel_size=3, stride=1, relu=True)
    x = BasicConv(x, f, kernel_size=3, stride=1, relu=False)
    
    return x


def EBlock(x, f, num_res=8):
    
    for _ in range(num_res):
        x = ResBlock(x, f)
        
    return x

def DBlock(x, f, num_res=8):
    
    for _ in range(num_res):
        x = ResBlock(x, f)
        
    return x


def AFF(x1, x2, x4, f):
    x = layers.Concatenate()([x1, x2, x4])
    x = BasicConv(x, f, kernel_size=1, stride=1, relu=True)
    x = BasicConv(x, f, kernel_size=3, stride=1, relu=False)
    
    return x

def SCM(x, f):
    x_in = x
    x_in = BasicConv(x_in, f//4, kernel_size=3, stride=1, relu=True)
    x_in = BasicConv(x_in, f // 2, kernel_size=1, stride=1, relu=True)
    x_in = BasicConv(x_in, f // 2, kernel_size=3, stride=1, relu=True)
    x_in = BasicConv(x_in, f-3, kernel_size=1, stride=1, relu=True)

    x_o = layers.Concatenate()([x_in, x])
    x_o = BasicConv(x_o, f, kernel_size=1, stride=1, relu=False)
    return x_o

def FAM(x1, x2, f):
    x_in = layers.Multiply()([x1, x2])
    x = BasicConv(x_in, f, kernel_size=3, stride=1, relu=False)
    x_o = layers.Add()([x, x1])

    return x_o

def get_model(input_shape, f=32, num_res = 20):
    x_in = Input(shape=input_shape)
    
    x_2 = layers.AveragePooling2D(pool_size=(2,2))(x_in)
    x_4 = layers.AveragePooling2D(pool_size=(2,2))(x_2)
    
    z2 = SCM(x_2, f * 2)
    z4 = SCM(x_4, f * 4)
#     print(z4.shape)
    
    
    x_ = BasicConv(x_in, f, kernel_size=3, relu=True, stride=1)
    res1 = EBlock(x_, f, num_res)
    
    z = BasicConv(res1, f*2, kernel_size=3, relu=True, stride=2)
    z = FAM(z, z2, f*2)
    res2 = EBlock(z, f*2, num_res)

    z = BasicConv(res2, f*4, kernel_size=3, relu=True, stride=2)
    z = FAM(z, z4, f*4)
    z = EBlock(z, f*4, num_res)
    
    z12 = layers.AveragePooling2D(pool_size=(2,2))(res1)
    z21 = layers.UpSampling2D(size = (2,2))(res2)
    z42 = layers.UpSampling2D(size = (2,2))(z)
    z41 = layers.UpSampling2D(size = (2,2))(z42)
    
    
    res2 = AFF(z12, res2, z42, f *2)
    res1 = AFF(res1, z21, z41, f )
    
    res2 = layers.SpatialDropout2D(rate=0.1)(res2)
    res1 = layers.SpatialDropout2D(rate=0.1)(res1)

    z = DBlock(z, f * 4, num_res)
    z_ = BasicConv(z, 1, kernel_size=3, relu=False, stride=1)
    z = BasicConv(z, f*2, kernel_size=4, relu=True, stride=2, transpose=True)
    # out1 = layers.Add()([z_, x_4])
#     print(z.shape, res2.shape)
    z = layers.Concatenate()([z, res2])
    z =  BasicConv(z, f * 2, kernel_size=1, relu=True, stride=1)
    z = DBlock(z, f * 2, num_res)
    z_ = BasicConv(z, 1, kernel_size=3, relu=False, stride=1)
    z =  BasicConv(z, f, kernel_size=4, relu=True, stride=2, transpose=True)  
    # out2 = layers.Add()([z_, x_2])
#     print(z.shape, res1.shape)
    z = layers.Concatenate()([z, res1])
    z = BasicConv(z, f , kernel_size=1, relu=True, stride=1)
    z = DBlock(z, f, num_res)
    z = BasicConv(z, 1, kernel_size=3, relu=False, stride=1)
    out3 = layers.Add()([z, x_in])
    
    return Model(inputs = x_in, outputs = out3)


if __name__ == '__main__':
    model = get_model(input_shape=(None, None, 1))
    print(model.summary())

# %%
