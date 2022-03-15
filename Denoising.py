import tensorflow as tf
import os

from model.DIDN import get_model
from tensorflow.keras.optimizers import Adam
from common.losses import *
from common.lr_scheduler import *
from common.metrics import *
from functools import wraps
import time

tf.keras.backend.clear_session()

gpu = tf.test.gpu_device_name()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

print(tf.__version__)


#TODO load data
def _get_dataset():
    """Function that returns tf.data.Dataset objects
    Train, val, and test datasets
    """
    
    pass


model = get_model(input_shape=(128, 128, 1))
print(model.summary())


content_loss_22 = ContentLoss(criterion='l2', output_layer=22, before_act=True) #Output layer 22 or 54
content_loss_54 = ContentLoss(criterion='l2', output_layer=54, before_act=True) #Output layer 22 or 54
pixel_loss_l1 = PixelLoss(criterion='l1')                                       ###MeanAbsoluteError
pixel_loss_l2 = PixelLoss(criterion='l2')                                       ###MeanSquaredError
ssim_loss = SSIM_loss()                                                         ###Weighted l1, l2 and SSIM loss

loss_weights = [1.0, 0.2, 0.2]

def loss_func(y_true, y_pred):
    """Content loss from VGG19 model"""
    c_loss = content_loss_22(y_true, y_pred)
    l1 = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    l2 = tf.keras.losses.mean_squared_error(y_true, y_pred)
    total_loss = loss_weights[0]*c_loss + loss_weights[1]*l1 + loss_weights[2]*l2
    return total_loss


learning_rate_fn = MultiStepLR(1e-4, [5000, 10000, 15000, 30000, 50000], 0.5)
optim = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

model.compile(optimizer = Adam(learning_rate=learning_rate_fn), 
            loss = pixel_loss_l1, 
            metrics = [PSNR, ssim, mssim, 'mse', 'mae'])


C = [
    tf.keras.callbacks.CSVLogger('DIDN_l1.csv', append=True), 
    tf.keras.callbacks.ModelCheckpoint('DIDN_l1.h5', save_best_only=True, verbose=1), 
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose =1)
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
def train(epoch=200):
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
    I_pred = model.evaluate(test_ds, batch_size = 16)
    print(I_pred.shape)