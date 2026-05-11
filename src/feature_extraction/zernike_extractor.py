# import mahotas
# import numpy as np

# def extract_zernike(image):
#     # Zernike moments butuh format uint8
#     if image.dtype != np.uint8:
#         image = (image * 255).astype(np.uint8)
        
#     radius = 21 # Radius lingkaran yang mencakup karakter
#     moments = mahotas.features.zernike_moments(image, radius)
#     return moments

import numpy as np
import cv2
import math

def radial_polynomial(n, m, r):
    """Menghitung polinomial radial Zernike."""
    R = np.zeros_like(r, dtype=np.float64)
    for s in range(int((n - abs(m)) / 2) + 1):
        c = ((-1)**s * math.factorial(n - s)) / \
            (math.factorial(s) * \
            math.factorial(int((n + abs(m)) / 2) - s) * \
            math.factorial(int((n - abs(m)) / 2) - s))
        R += c * (r ** (n - 2 * s))
    return R

def extract_zernike(img, radius=32, degree=8):
    """
    Mengekstrak Zernike Moments dari gambar grayscale.
    Hanya menggunakan Numpy dan Math murni tanpa Mahotas.
    """
    # Pastikan gambar berukuran 64x64 agar jari-jari lingkaran (radius=32) pas
    img = cv2.resize(img, (radius * 2, radius * 2))
    img = img.astype(np.float64) / 255.0 # Normalisasi
    
    # Membuat grid koordinat kartesian (X, Y)
    x = np.arange(-radius, radius)
    y = np.arange(-radius, radius)
    X, Y = np.meshgrid(x, y)
    
    # Konversi ke koordinat polar (R, Theta)
    R = np.sqrt(X**2 + Y**2) / radius
    Theta = np.arctan2(Y, X)
    
    # Zernike hanya dihitung di dalam unit circle (jari-jari <= 1)
    mask = R <= 1.0
    R_mask = R[mask]
    Theta_mask = Theta[mask]
    img_mask = img[mask]
    
    features = []
    
    # Menghitung momen berdasarkan degree (Orde)
    for n in range(degree + 1):
        for m in range(n + 1):
            if (n - m) % 2 == 0:  # Syarat Zernike
                R_nm = radial_polynomial(n, m, R_mask)
                
                # Fungsi basis Zernike (Real dan Imajiner)
                V_nm_real = R_nm * np.cos(m * Theta_mask)
                V_nm_imag = R_nm * np.sin(m * Theta_mask)
                
                # Proyeksi gambar ke dalam fungsi basis
                real_part = np.sum(img_mask * V_nm_real)
                imag_part = np.sum(img_mask * V_nm_imag)
                
                # Hitung Magnitude (Amplitudo) agar kebal rotasi
                c = (n + 1) / np.pi
                magnitude = c * np.sqrt(real_part**2 + imag_part**2)
                features.append(magnitude)
                
    return np.array(features, dtype=np.float32)