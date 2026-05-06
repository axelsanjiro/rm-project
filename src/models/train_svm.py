import time
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src.utils.data_loader import get_train_test_split

def run_svm_experiments():
    print("=== Memulai Eksperimen Klasifikasi SVM ===")
    
    # Zernike di-skip sementara sesuai kesepakatan
    feature_methods = ['HOG', 'LBP', 'Gabor']
    
    # Grid parameter untuk SVM sesuai metodologi paper
    svm_param_grid = {
        'C': [0.1, 1, 10], 
        'kernel': ['linear', 'rbf']
    }

    processed_data_dir = "dataset/processed"

    for feature in feature_methods:
        print(f"\n--- Melatih SVM menggunakan fitur: {feature} ---")
        
        # 1. Load & Split Data (80:20) secara on-the-fly
        X_train, X_test, y_train, y_test = get_train_test_split(processed_data_dir, feature)
        
        start_time = time.time()
        
        # 2. Inisialisasi dan Tuning Hyperparameter
        print(f"[*] Sedang mencari parameter terbaik dan melatih model...")
        svm_grid = GridSearchCV(SVC(random_state=42), svm_param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
        svm_grid.fit(X_train, y_train)
        
        # 3. Prediksi dan Evaluasi
        svm_preds = svm_grid.predict(X_test)
        training_time = time.time() - start_time
        
        print(f"[{feature}] Best Params  : {svm_grid.best_params_}")
        print(f"[{feature}] Accuracy     : {accuracy_score(y_test, svm_preds):.4f}")
        print(f"[{feature}] F1-Macro     : {f1_score(y_test, svm_preds, average='macro'):.4f}")
        print(f"[{feature}] Waktu Proses : {training_time:.2f} detik")
        print(f"[{feature}] Confusion Matrix:\n{confusion_matrix(y_test, svm_preds)}\n")

if __name__ == "__main__":
    run_svm_experiments()