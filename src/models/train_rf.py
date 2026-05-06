import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src.utils.data_loader import get_train_test_split

def run_rf_experiments():
    print("=== Memulai Eksperimen Klasifikasi Random Forest ===")
    
    feature_methods = ['HOG', 'LBP', 'Gabor']
    
    # Grid parameter untuk Random Forest sesuai metodologi paper
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20]
    }

    processed_data_dir = "dataset/processed"

    for feature in feature_methods:
        print(f"\n--- Melatih Random Forest menggunakan fitur: {feature} ---")
        
        X_train, X_test, y_train, y_test = get_train_test_split(processed_data_dir, feature)
        
        start_time = time.time()
        
        print(f"[*] Sedang mencari parameter terbaik dan melatih model...")
        rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        
        rf_preds = rf_grid.predict(X_test)
        training_time = time.time() - start_time
        
        print(f"[{feature}] Best Params  : {rf_grid.best_params_}")
        print(f"[{feature}] Accuracy     : {accuracy_score(y_test, rf_preds):.4f}")
        print(f"[{feature}] F1-Macro     : {f1_score(y_test, rf_preds, average='macro'):.4f}")
        print(f"[{feature}] Waktu Proses : {training_time:.2f} detik")
        print(f"[{feature}] Confusion Matrix:\n{confusion_matrix(y_test, rf_preds)}\n")

if __name__ == "__main__":
    run_rf_experiments()