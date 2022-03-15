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
                    y_true, y_pred, max_val = max_val, filter_size=8,
                    filter_sigma=1.5, k1=0.01, k2=0.03)
