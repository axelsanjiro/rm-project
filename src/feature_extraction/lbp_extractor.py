import numpy as np
from skimage.feature import local_binary_pattern

def extract_lbp(image):
    radius = 1
    n_points = 8 * radius
    
    # Ekstraksi matriks tekstur LBP
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    
    # PERBAIKAN: Membagi gambar menjadi grid 8x8 sel
    grid_x, grid_y = 8, 8
    h, w = image.shape
    cell_h, cell_w = h // grid_y, w // grid_x
    
    histograms = []
    for i in range(grid_y):
        for j in range(grid_x):
            # Ambil potongan area (cell)
            cell = lbp[i*cell_h : (i+1)*cell_h, j*cell_w : (j+1)*cell_w]
            
            # Hitung histogram KHUSUS untuk potongan ini
            hist, _ = np.histogram(cell, density=True, bins=n_bins, range=(0, n_bins))
            histograms.extend(hist)
            
    # Hasil akhir adalah gabungan histogram dari 64 kotak
    return np.array(histograms)