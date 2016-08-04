import numpy as np

def rgb_to_luminance(image):
    luminance = 0.2126*image[:,:,0] + 0.7152*image[:,:,1] + 0.0722*image[:,:,2]
    luminance /= 255.0
    return np.expand_dims(luminance, axis=3)

def downscale(image, factor):
    return image[::factor, ::factor, :]
