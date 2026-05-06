import os
import cv2
import numpy as np
import glob
import shutil

def preprocess_image(image_path, target_size=(64, 64)):
    """
    Membaca dan memproses satu gambar sesuai dengan metodologi paper.
    """
    # 1. Read Image & Grayscale Conversion
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    # 2. Binarization (Otsu's Thresholding)
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Resizing
    resized_img = cv2.resize(binary_img, target_size, interpolation=cv2.INTER_AREA)
    
    # 4. Normalization
    normalized_img = resized_img.astype(np.float32) / 255.0
    
    return normalized_img

def process_and_save_dataset(raw_dir, processed_dir, target_size=(64, 64)):
    """
    Memproses dataset berdasarkan struktur folder Training, Testing, dan Validation.
    """
    # Definisi struktur folder dari gambar dataset
    subsets = {
        'Training': ('training_words', 'training_labels.csv'),
        'Testing': ('testing_words', 'testing_labels.csv'),
        'Validation': ('validation_words', 'validation_labels.csv')
    }
    
    total_images = 0
    print("=== Memulai Preprocessing Dataset ===")
    
    for subset_name, (words_folder, csv_file) in subsets.items():
        # Path Sumber (Raw)
        raw_subset_dir = os.path.join(raw_dir, subset_name, words_folder)
        raw_csv_path = os.path.join(raw_dir, subset_name, csv_file)
        
        # Path Tujuan (Processed)
        processed_subset_dir = os.path.join(processed_dir, subset_name, words_folder)
        processed_csv_path = os.path.join(processed_dir, subset_name, csv_file)
        
        # Buat folder tujuan jika belum ada
        if not os.path.exists(processed_subset_dir):
            os.makedirs(processed_subset_dir)
            
        # Copy file CSV ke folder processed agar label tidak hilang
        if os.path.exists(raw_csv_path):
            shutil.copy2(raw_csv_path, processed_csv_path)
            print(f"[*] Berhasil menyalin {csv_file}")
        
        if not os.path.exists(raw_subset_dir):
            print(f"[!] Folder {raw_subset_dir} tidak ditemukan, lewati...")
            continue
            
        # Mengambil semua gambar .jpg / .png di dalam folder words
        image_files = glob.glob(os.path.join(raw_subset_dir, "*.jpg")) + glob.glob(os.path.join(raw_subset_dir, "*.png"))
        
        print(f"Memproses folder {subset_name}... ({len(image_files)} gambar)")
        
        for img_path in image_files:
            processed_img = preprocess_image(img_path, target_size)
            
            if processed_img is not None:
                # Kembalikan ke format 0-255 untuk disimpan sebagai gambar
                img_to_save = (processed_img * 255).astype(np.uint8)
                
                filename = os.path.basename(img_path)
                save_path = os.path.join(processed_subset_dir, filename)
                cv2.imwrite(save_path, img_to_save)
                total_images += 1
                
    print(f"\nPreprocessing Selesai! Total {total_images} gambar berhasil diproses.")
    print(f"Dataset hasil preprocessing dapat dilihat di: {processed_dir}")

if __name__ == "__main__":
    # Menyesuaikan dengan struktur folder barumu
    # Asumsi script dijalankan dari sejajar dengan folder 'dataset'
    RAW_DATA_DIR = "dataset/raw"  
    PROCESSED_DATA_DIR = "dataset/processed"
    
    process_and_save_dataset(RAW_DATA_DIR, PROCESSED_DATA_DIR)