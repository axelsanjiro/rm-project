import cv2
import numpy as np

def extract_gabor(image):
    gabor_features = []
    ksize = 5
    
    # Membagi gambar menjadi grid 4x4 untuk mempertahankan informasi spasial
    grid_x, grid_y = 4, 4
    h, w = image.shape
    cell_h, cell_w = h // grid_y, w // grid_x
    
    # PERBAIKAN: Gunakan 6 orientasi sudut
    thetas = [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]
    
    for theta in thetas:
        kernel = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(image, -1, kernel)
        
        # Hitung statistik lokal (mean dan varians) per KOTAK, bukan seluruh gambar
        for i in range(grid_y):
            for j in range(grid_x):
                cell = filtered[i*cell_h : (i+1)*cell_h, j*cell_w : (j+1)*cell_w]
                gabor_features.extend([cell.mean(), cell.var()])
                
    return np.array(gabor_features)