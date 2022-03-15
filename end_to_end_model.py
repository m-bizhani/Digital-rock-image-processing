import tensorflow as tf
tf.keras.backend.clear_session()

import os
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, Input
from common.losses import *
from common.metrics import *
from common.lr_scheduler import *
from model_load import get_combined_models

gpu = tf.test.gpu_device_name()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU:3'

print(tf.__version__)



img_size = (128,128,1)

with tf.device(gpu):
    didn, mimo, dfcan, _ = get_combined_models(load_individual_models=True)

didn.trainable = True
mimo.trainable = True
dfcan.trainable = True



def get_combined_model(input_shape):
    x_in = Input(input_shape)
    
    x1 = didn(x_in)
    x2 = mimo(x1)
    x3 = dfcan(x2)
    
    return Model(inputs=x_in, outputs = [x1, x2, x3])


with tf.device(gpu):
    model = get_combined_model(input_shape = img_size)
    print(model.summary())
    
    
pixel_loss_l1 = PixelLoss(criterion='l1')    ###MeanAbsoluteError
pixel_loss_l2 = PixelLoss(criterion='l2')    ###MeanSquaredError
ssim_loss = SSIM_loss()                      ###Weighted l1, l2 and SSIM loss

loss_weights = [0.2, 1.0, 1.0]



learning_rate_fn = MultiStepLR(1e-4, [10000, 20000, 40000, 50000], 0.5)

model.compile(optimizer = Adam(learning_rate=learning_rate_fn), 
            loss = [pixel_loss_l1, pixel_loss_l1, ssim_loss], loss_weights =[1.0,1.0,1.0], 
            metrics = [PSNR, ssim, mssim, 'mse', 'mae'])


C = [
    callbacks.CSVLogger('end_to_end_pretrain.csv', append=True), 
    callbacks.ModelCheckpoint('end_to_end_pretrain_weights.h5', save_best_only=True, verbose=1, save_weights_only=True), 
    callbacks.EarlyStopping(monitor='val_loss', patience=40, verbose =1)
    ]


if gpu:
    print('Training on GPU')
    
    with tf.device(gpu):
        model.fit(x = X_train,
                y = [Y1, Y2, Y3],                     ##Note, Y1, Y2 and Y3 are the three outputs of the model
                epochs=100,
                batch_size = 4,
                validation_split = 0.1,
                verbose=1, 
                callbacks=C)

else:
    print('Training on CPU')
    model.fit(x = X_train,
            y = [Y1, Y2, Y3],                     ##Note, Y1, Y2 and Y3 are the three outputs of the model
            epochs=100,
            batch_size = 4,
            validation_split = 0.1,
            verbose=1, 
            callbacks=C)



