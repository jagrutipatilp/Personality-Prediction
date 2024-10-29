import joblib
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# MBTI types and their descriptions
mbti_descriptions = {
    'INFJ': 'Insightful, idealistic, and empathetic. They seek meaning and connection in all that they do.',
    'ENTP': 'Enthusiastic, inventive, and analytical. They enjoy exploring new ideas and challenging the status quo.',
    'INTP': 'Curious, logical, and independent. They enjoy solving complex problems and understanding abstract concepts.',
    'INTJ': 'Strategic, determined, and analytical. They excel at planning and executing long-term goals.',
    'ENTJ': 'Confident, efficient, and decisive. They are natural leaders who thrive in strategic and high-stakes environments.',
    'ENFJ': 'Charismatic, compassionate, and inspiring. They are driven by a desire to help others and make a positive impact.',
    'INFP': 'Creative, idealistic, and empathetic. They are guided by their values and seek to understand others.',
    'ENFP': 'Enthusiastic, imaginative, and sociable. They enjoy exploring possibilities and connecting with people.',
    'ISFP': 'Artistic, sensitive, and spontaneous. They enjoy experiencing the world through their senses and expressing their creativity.',
    'ISTP': 'Practical, logical, and adventurous. They enjoy hands-on activities and solving immediate problems.',
    'ISFJ': 'Caring, meticulous, and reliable. They are dedicated to helping others and ensuring that their needs are met.',
    'ISTJ': 'Responsible, practical, and detail-oriented. They value order and structure and excel at organizing and managing tasks.',
    'ESTP': 'Energetic, action-oriented, and pragmatic. They thrive in dynamic environments and enjoy taking risks.',
    'ESFP': 'Spontaneous, outgoing, and fun-loving. They enjoy being the center of attention and living in the moment.',
    'ESTJ': 'Organized, assertive, and efficient. They are natural leaders who focus on getting things done and ensuring rules are followed.',
    'ESFJ': 'Warm, outgoing, and sociable. They value harmony and work hard to create a positive environment for others.'
}

# Mapping for MBTI scales
mbti_scales = {
    'Introversion (I) – Extroversion (E)': {'I': 1, 'E': -1},
    'Intuition (N) – Sensing (S)': {'N': 1, 'S': -1},
    'Thinking (T) – Feeling (F)': {'T': 1, 'F': -1},
    'Judging (J) – Perceiving (P)': {'J': 1, 'P': -1}
}

trait_mapping = {
    'INFJ': 'Insightful', 'ENTP': 'Innovative', 'INTP': 'Analytical', 'INTJ': 'Strategic',
    'ENTJ': 'Assertive', 'ENFJ': 'Charismatic', 'INFP': 'Idealistic', 'ENFP': 'Enthusiastic',
    'ISFP': 'Artistic', 'ISTP': 'Practical', 'ISFJ': 'Nurturing', 'ISTJ': 'Responsible',
    'ESTP': 'Adventurous', 'ESFP': 'Spontaneous', 'ESTJ': 'Organized', 'ESFJ': 'Supportive'
}

# Function to load the saved model and vectorizer
def load_model_and_vectorizer(model_filename, vectorizer_filename):
    model = joblib.load(model_filename)
    vectorizer = joblib.load(vectorizer_filename)
    return model, vectorizer

# Function to predict the personality probabilities
def predict_personality_proba(text, model, vectorizer):
    text_transformed = vectorizer.transform([text])
    probabilities = model.predict_proba(text_transformed)[0]
    return probabilities

# Function to visualize personality traits with a pie chart
def visualize_personality_traits(probabilities, classes):
    # Filter out zero probabilities
    filtered_indices = [i for i, p in enumerate(probabilities) if p > 0]
    filtered_probabilities = [probabilities[i] for i in filtered_indices]
    filtered_classes = [classes[i] for i in filtered_indices]
    
    # Use a colormap for diverse colors
    colors = cm.get_cmap('tab20', len(filtered_classes)).colors
    
    plt.figure(figsize=(10, 8))
    plt.pie(
        filtered_probabilities,
        labels=filtered_classes,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        pctdistance=0.85,  # Distance of percentage labels from center
        labeldistance=1.2,  # Distance of labels from center
        textprops={'fontsize': 10}  # Font size for labels
    )
    plt.title("Personality Trait")
    plt.show()


def visualize_mbti_scales(mbti_type, scales):
    # Ensure that `mbti_type` and `scales` are lists or arrays
    if isinstance(mbti_type, str):
        mbti_type = [mbti_type]
        scales = [scales]
    
    # Generate a list of colors for each bar
    colors = plt.get_cmap('tab20', len(mbti_type)).colors
    
    plt.figure(figsize=(15, 6))
    # Create the bar chart with different colors for each bar
    plt.barh(scales, mbti_type, color=colors)
    
    plt.xlabel('Score', fontsize=10)
    plt.title('MBTI Personality Scales', fontsize=10)
    
    # Adjust the size of tick labels for better readability
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.show()


# Function to predict the top personality types
def predict_top_personalities(text, model, vectorizer, top_n=3):
    text_transformed = vectorizer.transform([text])
    probabilities = model.predict_proba(text_transformed)[0]
    
    # Get indices of top_n probabilities
    top_indices = np.argsort(probabilities)[-top_n:][::-1]
    
    # Get top probabilities and their corresponding classes
    top_probabilities = probabilities[top_indices]
    top_classes = model.classes_[top_indices]
    
    return top_probabilities, top_classes

# Main execution
if __name__ == "__main__":
    model_filename = "personality_model.pkl"
    vectorizer_filename = "personality_model_vectorizer.pkl"
    
    # Load the saved model and vectorizer
    model, vectorizer = load_model_and_vectorizer(model_filename, vectorizer_filename)
    
    # Input string to predict
    input_text = "I am an introverted and creative person who enjoys deep conversations and reflecting on ideas. In my free time, I enjoy hiking and exploring new outdoor destinations. I'm an avid reader and enjoy staying up-to-date on the latest industry trends and publications. I'm a passionate music lover and enjoy attending concerts and music festivals. I'm a foodie and enjoy trying new recipes and experimenting with different cuisines. I'm a fitness enthusiast and enjoy staying active through regular exercise and sports."  # Example input text
    
    # Predict personality probabilities
    probabilities = predict_personality_proba(input_text, model, vectorizer)
    

    # Get class labels
    classes = model.classes_
    predicted_personality = classes[np.argmax(probabilities)]
    print(f"Predicted Personality Type: {trait_mapping[predicted_personality]}")
    print(f"Description: {mbti_descriptions.get(predicted_personality, 'No description available')}")
    liopop = [trait_mapping[i] for i in classes]
    # Visualize personality traits
    visualize_personality_traits(probabilities, liopop)

    # Visualize MBTI scales
    visualize_mbti_scales(probabilities, liopop)
    # Predict top personality probabilities and classes
    top_probabilities, top_classes = predict_top_personalities(input_text, model, vectorizer)
    
    typepersonality=""
    typedescription=""
    # Print results
    for i in range(len(top_classes)):
        typepersonality+=trait_mapping.get(top_classes[i], 'Unknown')
        typedescription+=mbti_descriptions.get(top_classes[i], 'No description available')
    
