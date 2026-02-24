import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from model import create_songnet_model

# Constants
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig('training_history.png')
    plt.close()

def train():
    if not os.path.exists('X.npy') or not os.path.exists('y.npy'):
        print("Error: Processed data not found. Run preprocess.py first.")
        return

    print("Loading data...")
    X = np.load('X.npy')
    y = np.load('y.npy')
    classes = np.load('classes.npy')
    
    # Split data: 70% Train, 20% Val, 10% Test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)
    
    print(f"Train shape: {X_train.shape}")
    print(f"Val shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")
    
    # One-hot encoding
    y_train_hot = tf.keras.utils.to_categorical(y_train, num_classes=len(classes))
    y_val_hot = tf.keras.utils.to_categorical(y_val, num_classes=len(classes))
    y_test_hot = tf.keras.utils.to_categorical(y_test, num_classes=len(classes))
    
    # Create Model
    input_shape = (X_train.shape[1], X_train.shape[2]) # (Time, n_mels)
    # Using Phase 2 upgrades: GRU + Attention
    model = create_songnet_model(input_shape, len(classes), rnn_type='gru', use_attention=True)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # Callbacks
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train
    print("Starting training...")
    history = model.fit(
        X_train, y_train_hot,
        validation_data=(X_val, y_val_hot),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, reduce_lr, early_stop]
    )
    
    plot_history(history)
    
    # Evaluate on Test Set
    print("Evaluating on Test Set...")
    loss, acc = model.evaluate(X_test, y_test_hot)
    print(f"Test Accuracy: {acc:.4f}")
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=classes))
    
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    train()
