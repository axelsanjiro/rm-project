import os
import pandas as pd
import cv2
import numpy as np

def augment_training_data():
    print("=== Memulai Data Augmentation (Rotasi +7 dan -7 derajat) ===")
    
    # Path ke data latih RAW
    train_dir = "dataset/raw/Training/training_words"
    csv_path = "dataset/raw/Training/training_labels.csv"
    
    if not os.path.exists(csv_path):
        print(f"[!] File {csv_path} tidak ditemukan.")
        return

    # Baca file CSV original
    df = pd.read_csv(csv_path)
    new_rows = []
    
    angles = [7, -7]
    total_augmented = 0
    
    for index, row in df.iterrows():
        img_name = row['IMAGE']
        label = row['GENERIC_NAME']
        img_path = os.path.join(train_dir, img_name)
        
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        for angle in angles:
            # Buat nama file baru
            name, ext = os.path.splitext(img_name)
            new_img_name = f"{name}_rot{angle}{ext}"
            new_img_path = os.path.join(train_dir, new_img_name)
            
            # Cek agar tidak menduplikasi jika script dijalankan 2x
            if not os.path.exists(new_img_path):
                # Matriks rotasi
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                # Lakukan rotasi dengan background putih (255,255,255)
                rotated = cv2.warpAffine(img, M, (w, h), borderValue=(255,255,255))
                
                # Simpan gambar baru
                cv2.imwrite(new_img_path, rotated)
                
                # Tambahkan ke daftar CSV baru
                new_rows.append({'IMAGE': new_img_name, 'WORD': label})
                total_augmented += 1
                
    if new_rows:
        # Gabungkan data lama dengan data baru
        df_new = pd.DataFrame(new_rows)
        df_combined = pd.concat([df, df_new], ignore_index=True)
        
        # Timpa file CSV lama dengan yang baru
        df_combined.to_csv(csv_path, index=False)
        print(f"[*] Selesai! Berhasil menambahkan {total_augmented} gambar augmentasi.")
        print(f"[*] Jumlah total data latih sekarang: {len(df_combined)} gambar.")
    else:
        print("[*] Tidak ada gambar baru yang ditambahkan (mungkin sudah pernah diaugmentasi).")

if __name__ == "__main__":
    augment_training_data()