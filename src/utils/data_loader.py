import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import metode ekstraksi fitur yang sudah jalan
from src.feature_extraction.hog_extractor import extract_hog
from src.feature_extraction.lbp_extractor import extract_lbp
from src.feature_extraction.gabor_extractor import extract_gabor

def extract_features(image_path, feature_name):
    """Membaca gambar grayscale dan menerapkan ekstraksi fitur."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Normalisasi kembali ke 0-1 jika metode memerlukannya
    img_normalized = img.astype(np.float32) / 255.0

    if feature_name == 'HOG':
        return extract_hog(img)
    elif feature_name == 'LBP':
        return extract_lbp(img)
    elif feature_name == 'Gabor':
        return extract_gabor(img_normalized)
    elif feature_name == 'Fusion':
        # MENGGABUNGKAN KETIGA FITUR (Feature Fusion)
        hog_feat = extract_hog(img)
        lbp_feat = extract_lbp(img)
        gabor_feat = extract_gabor(img_normalized)
        
        # Concatenate array menjadi satu vektor panjang
        fusion_feat = np.concatenate([hog_feat, lbp_feat, gabor_feat])
        return fusion_feat
    else:
        raise ValueError(f"Metode {feature_name} tidak dikenali/di-skip.")

def load_and_combine_data(processed_dir, feature_name):
    """
    Menggabungkan seluruh dataset (Train, Test, Val) menjadi satu kesatuan 
    dan mengekstrak fiturnya.
    """
    subsets = {
        'Training': ('training_words', 'training_labels.csv'),
        'Testing': ('testing_words', 'testing_labels.csv'),
        'Validation': ('validation_words', 'validation_labels.csv')
    }
    
    X_all = []
    y_all = []
    
    print(f"[*] Menggabungkan data & mengekstraksi fitur {feature_name}...")
    
    for subset_name, (words_folder, csv_file) in subsets.items():
        csv_path = os.path.join(processed_dir, subset_name, csv_file)
        img_folder = os.path.join(processed_dir, subset_name, words_folder)
        
        if not os.path.exists(csv_path) or not os.path.exists(img_folder):
            continue
            
        df = pd.read_csv(csv_path)
        
        img_col = [col for col in df.columns if 'IMAGE' in col.upper()][0]
        label_col = [col for col in df.columns if 'MEDICINE' in col.upper()][0]
        
        for index, row in df.iterrows():
            img_name = row[img_col]
            label = row[label_col]
            
            img_path = os.path.join(img_folder, img_name)
            
            if os.path.exists(img_path):
                features = extract_features(img_path, feature_name)
                if features is not None:
                    X_all.append(features)
                    y_all.append(label)
                    
    return np.array(X_all), np.array(y_all)

def get_train_test_split(processed_dir, feature_name, test_size=0.2, random_state=42):
    """
    Mengembalikan data latih (80%) dan data uji (20%) yang valid untuk melatih model.
    """
    X_all, y_all = load_and_combine_data(processed_dir, feature_name)
    
    # Membagi ulang rasio menjadi 80:20
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=random_state, stratify=y_all
    )
    
    print(f"[+] Total Data: {len(X_all)} | Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test