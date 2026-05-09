import os
import glob
import pandas as pd

def revert_augmentation():
    print("=== Memulai Proses Pembersihan Data Augmentasi ===")
    
    # Kita harus membersihkan di folder RAW maupun PROCESSED
    dirs_to_clean = [
        "dataset/raw/Training",
        "dataset/processed/Training"
    ]
    
    for base_dir in dirs_to_clean:
        img_dir = os.path.join(base_dir, "training_words")
        csv_path = os.path.join(base_dir, "training_labels.csv")
        
        print(f"\n[*] Memeriksa direktori: {base_dir}")
        
        # 1. Hapus file gambar yang memiliki kata '_rot' di namanya
        if os.path.exists(img_dir):
            aug_images = glob.glob(os.path.join(img_dir, "*_rot*.jpg")) + glob.glob(os.path.join(img_dir, "*_rot*.png"))
            count = 0
            for img in aug_images:
                try:
                    os.remove(img)
                    count += 1
                except Exception as e:
                    print(f"Gagal menghapus {img}: {e}")
            print(f"    -> Berhasil menghapus {count} gambar fisik rotasi.")
            
        # 2. Hapus baris data augmentasi dari file CSV
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            awal_len = len(df)
            
            # Cari nama kolom gambar secara dinamis
            img_col = [col for col in df.columns if 'IMAGE' in col.upper()][0]
            
            # Buang baris yang nama gambarnya mengandung '_rot'
            df_clean = df[~df[img_col].str.contains('_rot', na=False)]
            
            # Timpa/Save file CSV yang sudah bersih
            df_clean.to_csv(csv_path, index=False)
            akhir_len = len(df_clean)
            
            print(f"    -> Berhasil menghapus {awal_len - akhir_len} baris data dari CSV.")
            print(f"    -> Sisa data original di CSV: {akhir_len} baris.")

    print("\n=== Pembersihan Selesai! ===")

if __name__ == "__main__":
    revert_augmentation()