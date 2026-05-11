import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.utils.data_loader import get_train_test_split

def run_rf_experiments():
    print("=== Eksperimen Random Forest untuk Fitur Individu + PCA ===")
    processed_data_dir = "dataset/processed"
    features_to_test = ['HOG', 'LBP', 'Gabor', 'Zernike']  # <-- TAMBAHAN FITUR ZERNIKE
    
    for feat in features_to_test:
        print(f"\n[*] Mengekstraksi fitur {feat}...")
        X_train, X_test, y_train, y_test = get_train_test_split(processed_data_dir, feat)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        pca = PCA(n_components=0.95, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        
        print(f"[*] Dimensi {feat} setelah PCA: {X_train.shape[1]} komponen")

        model = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 20]}

        print(f"--- Melatih Random Forest dengan {feat} ---")
        start_time = time.time()
        
        grid = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        preds = grid.predict(X_test)
        training_time = time.time() - start_time
        
        print(f"[RF - {feat}] Best Params  : {grid.best_params_}")
        print(f"[RF - {feat}] Accuracy     : {accuracy_score(y_test, preds):.4f}")
        print(f"[RF - {feat}] F1-Macro     : {f1_score(y_test, preds, average='macro'):.4f}")
        print(f"[RF - {feat}] Waktu Proses : {training_time:.2f} detik")

if __name__ == "__main__":
    run_rf_experiments()