from skimage.feature import hog

def extract_hog(image):
    # Parameter disesuaikan untuk citra 64x64
    features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return features