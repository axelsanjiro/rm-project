import mahotas
import numpy as np

def extract_zernike(image):
    # Zernike moments butuh format uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
        
    radius = 21 # Radius lingkaran yang mencakup karakter
    moments = mahotas.features.zernike_moments(image, radius)
    return moments