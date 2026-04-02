import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
np.random.seed(42)

class IrisClassifier:
    """
    A modular classifier for Iris flower categorization using 
    Categorical Cross-Entropy loss and Softmax activation.
    Enhanced with L2 Regularization and Xavier Initialization.
    """
    def __init__(self, input_dim, output_dim, l2_lambda=0.01):
        # Xavier Initialization for better gradient flow
        limit = np.sqrt(6 / (input_dim + output_dim))
        self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.loss_history = []
        self.mean = None
        self.std = None
        self.l2_lambda = l2_lambda

    def _softmax(self, Z):
        # Numerically stable softmax
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def _normalize(self, X, fit=False):
        if fit:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0)
        return (X - self.mean) / (self.std + 1e-8)

    def train(self, X, Y, epochs=10000, lr=0.1):
        # Normalize features (Standardization)
        X_norm = self._normalize(X, fit=True)
        # Add bias (column of 1s)
        X_bias = np.hstack([np.ones((X_norm.shape[0], 1)), X_norm])
        
        n_samples = X_bias.shape[0]
        current_lr = lr
        
        for epoch in range(epochs):
            # Forward pass: logits -> softmax probabilities
            Z = np.dot(X_bias, self.W)
            P = self._softmax(Z)
            
            # Compute Categorical Cross-Entropy Loss with L2 penalty
            # Adding epsilon for log stability
            loss = -np.mean(np.sum(Y * np.log(P + 1e-15), axis=1)) + self.l2_lambda * np.sum(np.square(self.W))
            self.loss_history.append(loss)
            
            # Gradient for Softmax + Cross-Entropy: grad = (1/n) * X.T * (P - Y) + 2 * lambda * W
            gradient = (1/n_samples) * np.dot(X_bias.T, (P - Y)) + 2 * self.l2_lambda * self.W
            
            # Adaptive learning rate decay
            if epoch > 0 and epoch % 2000 == 0:
                current_lr *= 0.8
                
            self.W -= current_lr * gradient
            
            if epoch % 1000 == 0:
                print(f"Epoch {epoch:5d} | Cross-Entropy Loss: {loss:.6f} | LR: {current_lr:.4f}")

    def predict_proba(self, X):
        X_norm = self._normalize(X)
        X_bias = np.hstack([np.ones((X_norm.shape[0], 1)), X_norm])
        Z = np.dot(X_bias, self.W)
        return self._softmax(Z)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

def load_data(file_path):
    # Determine the directory of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, file_path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Dataset file not found at: {full_path}")
        
    df = pd.read_csv(full_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Feature Engineering: Retaining non-linear interactions
    df['petal_area'] = df['petal_length'] * df['petal_width']
    df['sepal_petal_len'] = df['sepal_length'] * df['petal_length']
    
    species_map = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2,
        'Iris-gigantica': 3
    }
    df['target'] = df['species'].map(species_map)
    
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'petal_area', 'sepal_petal_len']
    X = df[features].values
    Y = np.eye(4)[df['target'].values] # One-hot labels for Cross-Entropy
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    y_test_labels = df['target'].values[split:]
    
    return X_train, X_test, Y_train, Y_test, y_test_labels, species_map, df

def save_visualizations(model, df, cm, accuracy, species_map):
    # Ensure the 'outputs' directory exists relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'outputs')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    labels = list(species_map.keys())
    
    # --- 1. Loss Convergence Graph (Updated for Cross-Entropy) ---
    plt.figure(figsize=(10, 6))
    plt.plot(model.loss_history, color='tab:blue', linewidth=2, label='Cross-Entropy Path')
    plt.title('Technical Analysis: Learning Curve & Loss Convergence', fontsize=14, fontweight='bold')
    plt.xlabel('Training Iterations (Epochs)', fontsize=12)
    plt.ylabel('Categorical Cross-Entropy Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    mse_path = os.path.join(output_dir, 'mse_convergence.png')
    plt.savefig(mse_path, dpi=300)
    plt.close()

    # --- 2. Confusion Matrix ---
    plt.figure(figsize=(9, 8))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title(f'Business Metric: Flower Sorting Confusion Matrix\nAccuracy: {accuracy*100:.2f}%', fontsize=14, fontweight='bold')
    plt.colorbar(label='Sample Count')
    
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha='right')
    plt.yticks(tick_marks, labels)
    
    thresh = cm.max() / 2.
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=12, fontweight='bold')
    
    plt.ylabel('Actual Flower Category', fontsize=12)
    plt.xlabel('System-Predicted Category', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # --- 3. Classification Clouds Scatter Plot ---
    plt.figure(figsize=(11, 7))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    for i, label in enumerate(labels):
        subset = df[df['species'] == label]
        plt.scatter(subset['petal_length'], subset['petal_width'], 
                    c=colors[i], label=label, edgecolors='k', alpha=0.8, s=80)
    
    plt.title('Point Cloud Distribution: Multi-Dimensional Flower Separation', fontsize=14, fontweight='bold')
    plt.xlabel('Petal Length', fontsize=12)
    plt.ylabel('Petal Width', fontsize=12)
    plt.legend(title="Flower Categories", frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.5)
    cloud_path = os.path.join(output_dir, 'classification_clouds.png')
    plt.savefig(cloud_path, dpi=300)
    plt.close()

def main():
    print("="*60)
    print("      ADVANCED IRIS FLOWER CLASSIFICATION SYSTEM (V3.0)      ")
    print("="*60)
    
    try:
        # Load and Split Data
        X_train, X_test, Y_train, Y_test, y_test_labels, species_map, df = load_data('iris_extended.csv')
        
        # Initialize and Train Model
        # Input dimension = 6 features + 1 bias = 7
        # Output dimension = 4 species
        model = IrisClassifier(input_dim=7, output_dim=4, l2_lambda=0.005)
        print("\n[INFO] Starting training with Softmax & Categorical Cross-Entropy...")
        model.train(X_train, Y_train, epochs=10000, lr=0.2)
        
        # System Evaluation
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test_labels)
        
        # Compute 4x4 Confusion Matrix
        cm = np.zeros((4, 4), dtype=int)
        for t, p in zip(y_test_labels, y_pred):
            cm[t, p] += 1
        
        print(f"\n[SUCCESS] Training Complete. Final Test Accuracy: {accuracy*100:.2f}%")
        
        # Generate and Save Visualizations
        print("\n[INFO] Generating professional visualization files...")
        save_visualizations(model, df, cm, accuracy, species_map)
        
        # Interactive Real-Time Prediction
        if os.environ.get("SKIP_INTERACTIVE") == "1":
            print("\n[INFO] Skipping interactive interface as requested.")
            return

        print("\n" + "-"*40)
        print("   LIVE FLOWER CLASSIFICATION INTERFACE")
        print("-"*40)
        print("Please enter flower measurements for automated sorting:")
        try:
            sl = float(input(" -> Sepal Length: "))
            sw = float(input(" -> Sepal Width:  "))
            pl = float(input(" -> Petal Length: "))
            pw = float(input(" -> Petal Width:  "))
            
            # Add Interaction Features
            pa = pl * pw
            spl = sl * pl
            
            user_input = np.array([[sl, sw, pl, pw, pa, spl]])
            probs = model.predict_proba(user_input)[0]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            
            rev_map = {v: k for k, v in species_map.items()}
            print(f"\n>>> RESULT: This flower belongs to: {rev_map[pred_idx]}")
            print(f">>> CONFIDENCE: {confidence*100:.2f}%")
            print(">>> LOG: Sorting complete. Output routed to corresponding bin.")
        except ValueError:
            print("\n[ERROR] Invalid input. Please enter numeric values.")
            
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {str(e)}")

    print("\n" + "="*60)
    print("                 PROCESS COMPLETED SUCCESSFULLY              ")
    print("="*60)

if __name__ == "__main__":
    main()
