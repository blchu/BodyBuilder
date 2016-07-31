def rgb_to_luminance(image):
    luminance = 0.2126*image[:,:,0] + 0.7152*image[:,:,1] + 0.0722*image[:,:,2]
    luminance /= 255.0
    return luminance

def downscale(image, factor):
    return image[::factor, ::factor]
