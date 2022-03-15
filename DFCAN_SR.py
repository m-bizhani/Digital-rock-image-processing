import tensorflow as tf
tf.keras.backend.clear_session()

import os
from model.DFCAN import DFCAN 
from tensorflow.keras.optimizers import Adam
from common.losses import *
from common.lr_scheduler import *
from common.metrics import *
from functools import wraps
import time
from tensorflow.keras.models import load_model


gpu = tf.test.gpu_device_name()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

print(tf.__version__)


##TODO load train and test data in tf.data.Dataset form 

##Instantiate the model with right parameters
model = DFCAN(input_shape = (128,128,1), num_res_blocks = 4, num_res = 4, scale=2)
print(model.summary())



content_loss_22 = ContentLoss(criterion='l2', output_layer=22, before_act=True) #Output layer 22 or 54
content_loss_54 = ContentLoss(criterion='l2', output_layer=54, before_act=True) #Output layer 22 or 54
pixel_loss_l1 = PixelLoss(criterion='l1')                                       ###MeanAbsoluteError
pixel_loss_l2 = PixelLoss(criterion='l2')                                       ###MeanSquaredError
ssim_loss = SSIM_loss()                                                         ###Weighted l1, l2 and SSIM loss

loss_weights = [0.2, 1.0, 1.0]

def loss_func(y_true, y_pred):
    """Content loss based on VGG19"""
    c_loss = content_loss_22(y_true, y_pred)
    l1 = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    l2 = tf.keras.losses.mean_squared_error(y_true, y_pred)
    total_loss = loss_weights[0]*c_loss + loss_weights[1]*l1 + loss_weights[2]*l2
    return total_loss


learning_rate_fn = MultiStepLR(1e-4, [10000, 20000, 40000, 50000], 0.5)

model.compile(optimizer = Adam(learning_rate=learning_rate_fn), 
            loss = ssim_loss, 
            metrics = [PSNR, ssim, mssim, 'mse', 'mae'])


C = [
    tf.keras.callbacks.CSVLogger('DFCAN-ssim-l2.csv', append=True), 
    tf.keras.callbacks.ModelCheckpoint('DFCAN-ssim-l2.h5', save_best_only=True, verbose=1), 
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, verbose =1)
    ]



###Train the model from scratch
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
def train(epoch=100):
    if gpu:
        print('GPU training')
        with tf.device(gpu):
            model.fit(train_ds, epochs = epoch, validation_data = val_ds, callbacks = C)
            
    else:
        print('CPU training')
        model.fit(train_ds, epochs = epoch, validation_data = val_ds, callbacks = C)
        
    return model

model = train()


##Evlautate the model

with tf.device(gpu):
    model.evaluate(test_ds, batch_size=16)

