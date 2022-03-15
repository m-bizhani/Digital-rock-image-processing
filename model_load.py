from tensorflow.keras.models import load_model 
import tensorflow as tf


def PSNR(y_true, y_pred):
        max_pixel = 1.0
        return tf.image.psnr(y_true, y_pred, max_val =max_pixel)

def ssim(y_true, y_pred):
        max_val = 1.0
        return tf.image.ssim(y_true, y_pred, max_val = max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)

def mssim(y_true, y_pred):
        max_val = 1.0
        return tf.image.ssim_multiscale(
                        y_true, y_pred, max_val = max_val, filter_size=7,
                        filter_sigma=1.5, k1=0.01, k2=0.03)

def SSIM_loss():
        def SSIMLoss(y_true, y_pred):
                # l1 = tf.keras.losses.mean_absolute_error(y_true, y_pred)
                ss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val = 1.0))
                l2 = tf.keras.losses.mean_absolute_error(y_true, y_pred)
                return ss + 0.1*l2
        return SSIMLoss

def get_combined_models(path = './weights/', load_individual_models = False):
        
        didn = load_model(path + 'DIDN_l1.h5', 
                        custom_objects={'PSNR': PSNR, 'ssim':ssim, 'mssim': mssim, 'tf': tf} ,
                        compile=False)
        
        for layer in didn.layers:
                layer._name = layer._name + str("_1")

        didn._name = 'didn'
        mimo = load_model(path + 'MIMO_l1.h5', 
                        custom_objects={'PSNR': PSNR, 'ssim':ssim, 'mssim': mssim, 'tf': tf} ,
                        compile=False)
        mimo._name = 'mimo'
        for layer in mimo.layers:
                layer._name = layer._name + str("_2")

        dfcan = load_model(path + 'DFCAN-ssim-l2.h5', 
                        custom_objects={'PSNR': PSNR, 'ssim':ssim, 'mssim': mssim, 'tf': tf, 'SSIM_loss': SSIM_loss} ,
                        compile=False)
        dfcan._name = 'dfcan'
        
        for layer in dfcan.layers:
                layer._name = layer._name + str("_3")
        model  = tf.keras.Sequential([didn, mimo, dfcan])
#         model.trainable = False
        

        if load_individual_models:
                return didn, mimo, dfcan, model
        else:
                return model


if __name__ == '__main__':
        model = get_combined_models()
        print(model.summary())
