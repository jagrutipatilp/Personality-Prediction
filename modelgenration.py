import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Function to load and preprocess the dataset
def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)  # Remove missing values
    return df

# Function to vectorize text data
def vectorize_text(X):
    vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 words
    X_transformed = vectorizer.fit_transform(X)
    return X_transformed, vectorizer

# Function to train and save the model
def train_and_save_model(X_train, y_train, model_filename, vectorizer_filename, vectorizer):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the trained model and vectorizer
    joblib.dump(model, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)
    
    print(f"Model saved to {model_filename}")
    print(f"Vectorizer saved to {vectorizer_filename}")
    
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix visualization
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return y_pred

# Function to visualize feature importance
def plot_feature_importance(model, vectorizer):
    feature_importances = model.feature_importances_
    features = vectorizer.get_feature_names_out()
    
    # Get top 20 features for better visualization
    top_indices = np.argsort(feature_importances)[-20:]
    top_features = np.array(features)[top_indices]
    top_importances = feature_importances[top_indices]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features)
    plt.title("Top 20 Feature Importances")
    plt.show()

# Main execution
if __name__ == "__main__":
    csv_path = os.path.join('.', 'mbti.csv')  # Adjusted to current directory
    df = load_and_preprocess_data(csv_path)
    
    X = df['posts']
    y = df['type']
    
    X_transformed, vectorizer = vectorize_text(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
    
    model_filename = "personality_model.pkl"
    vectorizer_filename = "personality_model_vectorizer.pkl"
    
    model = train_and_save_model(X_train, y_train, model_filename, vectorizer_filename, vectorizer)
    evaluate_model(model, X_test, y_test)
    plot_feature_importance(model, vectorizer)
