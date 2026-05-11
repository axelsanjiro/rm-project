import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.utils.data_loader import get_train_test_split

def run_knn_experiments():
    print("=== Eksperimen KNN untuk Fitur Individu + PCA ===")
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

        model = KNeighborsClassifier()
        param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}

        print(f"--- Melatih KNN dengan {feat} ---")
        start_time = time.time()
        
        grid = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        preds = grid.predict(X_test)
        training_time = time.time() - start_time
        
        print(f"[KNN - {feat}] Best Params  : {grid.best_params_}")
        print(f"[KNN - {feat}] Accuracy     : {accuracy_score(y_test, preds):.4f}")
        print(f"[KNN - {feat}] F1-Macro     : {f1_score(y_test, preds, average='macro'):.4f}")
        print(f"[KNN - {feat}] Waktu Proses : {training_time:.2f} detik")

if __name__ == "__main__":
    run_knn_experiments()