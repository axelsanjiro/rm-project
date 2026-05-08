import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from src.utils.data_loader import get_train_test_split

def run_knn_experiments():
    print("=== Memulai Eksperimen Klasifikasi KNN ===")
    
    feature_methods = ['HOG', 'LBP', 'Gabor']
    
    # Grid parameter untuk KNN sesuai metodologi paper
    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    }

    processed_data_dir = "dataset/processed"

    for feature in feature_methods:
        print(f"\n--- Melatih KNN menggunakan fitur: {feature} ---")
        
        # 1. Load & Split Data (80:20) secara on-the-fly
        X_train, X_test, y_train, y_test = get_train_test_split(processed_data_dir, feature)
        
        # 2. Standardisasi Skala Fitur (KECUALI HOG)
        if feature != 'HOG':
            print("[*] Melakukan Standardisasi Skala Fitur (StandardScaler)...")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            print("[*] Fitur HOG melewati StandardScaler (sudah ternormalisasi secara internal)...")
        
        start_time = time.time()
        
        # 3. Inisialisasi dan Tuning Hyperparameter
        print(f"[*] Sedang mencari parameter terbaik dan melatih model...")
        knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
        knn_grid.fit(X_train, y_train)
        
        # 4. Prediksi dan Evaluasi
        knn_preds = knn_grid.predict(X_test)
        training_time = time.time() - start_time
        
        print(f"[{feature}] Best Params  : {knn_grid.best_params_}")
        print(f"[{feature}] Accuracy     : {accuracy_score(y_test, knn_preds):.4f}")
        print(f"[{feature}] F1-Macro     : {f1_score(y_test, knn_preds, average='macro'):.4f}")
        print(f"[{feature}] Waktu Proses : {training_time:.2f} detik")
        print(f"[{feature}] Confusion Matrix:\n{confusion_matrix(y_test, knn_preds)}\n")

if __name__ == "__main__":
    run_knn_experiments()