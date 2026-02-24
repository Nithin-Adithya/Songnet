import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os

# Set Memory Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
def evaluate_models():
    print("Loading data for benchmarking...")
    try:
        # Use mmap_mode to save memory if file is large
        X = np.load('X.npy', mmap_mode='r') 
        y = np.load('y.npy')
    except FileNotFoundError:
        print("Data files not found. Please run preprocess.py first.")
        return

    # Split data (Same seed as train.py to ensure fair test set evaluation)
    # Note: We need the test set indices.
    # Stratified split 70/20/10
    indices = np.arange(len(y))
    _, idx_temp, _, y_temp = train_test_split(indices, y, test_size=0.3, random_state=42, stratify=y)
    _, idx_test, _, y_test = train_test_split(idx_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)
    
    X_test = X[idx_test]
    # For ML models we need train set too
    idx_train, _, y_train, _ = train_test_split(indices, y, test_size=0.3, random_state=42, stratify=y)
    X_train = X[idx_train]

    print("Data loaded. Computing baseline features (Global Average Pooling)...")
    # Compute mean over time for ML baselines: (N, Time, Mel) -> (N, Mel)
    X_train_flat = np.mean(X_train, axis=1)
    X_test_flat = np.mean(X_test, axis=1)
    
    # Scale features
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)

    results = {}

    # 1. Dummy Classifier (Random Baseline)
    print("Evaluating Random Baseline...")
    dummy = DummyClassifier(strategy='stratified', random_state=42)
    dummy.fit(X_train_flat, y_train)
    results['Random'] = accuracy_score(y_test, dummy.predict(X_test_flat))

    # 2. KNN
    print("Evaluating KNN...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_flat, y_train)
    results['KNN'] = accuracy_score(y_test, knn.predict(X_test_flat))

    # 3. Logistic Regression
    print("Evaluating Logistic Regression...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_flat, y_train)
    results['Logistic Regression'] = accuracy_score(y_test, lr.predict(X_test_flat))

    # 4. SVM
    print("Evaluating SVM...")
    svm = SVC()
    svm.fit(X_train_flat, y_train)
    results['SVM'] = accuracy_score(y_test, svm.predict(X_test_flat))
    
    # 5. SongNet (CNN+RNN)
    print("Evaluating SongNet...")
    if os.path.exists('best_model.h5'):
        try:
            from model import SelfAttention
            model = tf.keras.models.load_model('best_model.h5', custom_objects={'SelfAttention': SelfAttention})
            # Evaluate directly on X_test (Raw MFCCs)
            y_test_hot = tf.keras.utils.to_categorical(y_test, num_classes=8)
            loss, acc = model.evaluate(X_test, y_test_hot, verbose=0)
            results['SongNet (CNN+RNN)'] = acc
        except Exception as e:
            print(f"Could not load/evaluate SongNet: {e}")
            results['SongNet (CNN+RNN)'] = "N/A"
    else:
        results['SongNet (CNN+RNN)'] = "Not Trained"

    print("\n\n## Model Comparison")
    print("| Model | Accuracy |")
    print("|-------|----------|")
    for name, acc in results.items():
        if isinstance(acc, float):
            print(f"| {name} | {acc*100:.1f}% |")
        else:
            print(f"| {name} | {acc} |")

if __name__ == "__main__":
    evaluate_models()
