import time
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # <-- TAMBAHAN IMPORT PCA
from src.utils.data_loader import get_train_test_split

def run_fusion_experiments():
    print("=== Memulai Eksperimen Feature Fusion (HOG + LBP + Gabor) ===")
    processed_data_dir = "dataset/processed"
    
    # 1. Load Data Gabungan
    print("\n[*] Mengekstraksi dan menggabungkan fitur (Ini akan memakan waktu)...")
    X_train, X_test, y_train, y_test = get_train_test_split(processed_data_dir, 'Fusion')
    
    # 2. Standardisasi Fitur (Sangat penting sebelum PCA)
    print("[*] Melakukan Standardisasi Skala Fitur (StandardScaler)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 3. PCA (Principal Component Analysis)
    print("[*] Menerapkan PCA untuk mereduksi dimensi dan menghilangkan noise...")
    # Menyimpan 95% varians informasi, membuang 5% noise
    pca = PCA(n_components=0.95, random_state=42) 
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    print(f"[*] Dimensi fitur setelah PCA berhasil dipadatkan menjadi: {X_train.shape[1]} komponen!")

    # Sedikit melebarkan pencarian C pada SVM untuk potensi akurasi maksimal
    models = {
        'SVM': (SVC(random_state=42), {'C': [0.1, 1, 10, 100], 'kernel': ['rbf']}),
        'KNN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}),
        'Random Forest': (RandomForestClassifier(random_state=42), {'n_estimators': [50, 100, 200], 'max_depth': [None, 20]})
    }

    for model_name, (model, param_grid) in models.items():
        print(f"\n--- Melatih {model_name} dengan fitur Fusion + PCA ---")
        start_time = time.time()
        
        grid = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        preds = grid.predict(X_test)
        training_time = time.time() - start_time
        
        print(f"[{model_name}] Best Params  : {grid.best_params_}")
        print(f"[{model_name}] Accuracy     : {accuracy_score(y_test, preds):.4f}")
        print(f"[{model_name}] F1-Macro     : {f1_score(y_test, preds, average='macro'):.4f}")
        print(f"[{model_name}] Waktu Proses : {training_time:.2f} detik")

if __name__ == "__main__":
    run_fusion_experiments()