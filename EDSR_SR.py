#%%
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from model.EDSR import edsr 
from tensorflow.keras.optimizers import Adam
from common.losses import *
from common.lr_scheduler import *
from common.metrics import *
from functools import wraps
import time

tf.keras.backend.clear_session()

##TODO load train and test data


model = edsr(scale=2, f=256, num_of_res_blocks=16, res_block_scaling=None)
print(model.summary())





content_loss_22 = ContentLoss(criterion='l2', output_layer=22, before_act=True) #Output layer 22 or 54
content_loss_54 = ContentLoss(criterion='l2', output_layer=54, before_act=True) #Output layer 22 or 54
pixel_loss_l1 = PixelLoss(criterion='l1')                                       ###MeanAbsoluteError
pixel_loss_l2 = PixelLoss(criterion='l2')                                       ###MeanSquaredError
ssim_loss = SSIM_loss()                                                         ###Weighted l1, l2 and SSIM loss

loss_weights = [0.2, 1.0, 1.0]

def loss_func(y_true, y_pred):
    """Content loss function based on VGG19 model"""
    c_loss = content_loss_22(y_true, y_pred)
    l1 = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    l2 = tf.keras.losses.mean_squared_error(y_true, y_pred)
    total_loss = loss_weights[0]*c_loss + loss_weights[1]*l1 + loss_weights[2]*l2
    
    return total_loss

    
C = [
    tf.keras.callbacks.CSVLogger('Outputlog.csv', append=True), 
    tf.keras.callbacks.ModelCheckpoint('EDSR_ssmlos.h5', save_best_only=True, verbose=1), 
    tf.keras.callbacks.EarlyStopping(monitor='val_PSNR', patience=30, verbose =1)
    ]


gpu = str(tf.test.gpu_device_name())

def execuation_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        end = time.time()
        print(f'\n Total execuation time for {func.__name__} is {end-start}s')
        return results
    return wrapper

@execuation_time
def train(epoch=1):
    if gpu:
        print('GPU training')
        with tf.device(gpu):
            model.fit(train_ds, epochs = epoch, validation_data = val_ds, callbacks = C)
            
    else:
        print('CPU training')
        model.fit(train_ds, epochs = epoch, validation_data = val_ds, callbacks = C)
        
    return model

model = train()


with tf.device(gpu):
    model.evaluate(test_ds, batch_size=16)