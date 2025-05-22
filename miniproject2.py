import sys
import json
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import traceback

# Suppress Hugging Face symlinks warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Improved Text Preprocessing
def preprocess_text(text):
    if not text or not isinstance(text, str):
        return ""
    
    text = text.lower()  # Convert to lowercase
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)  # Tokenize (remove punctuation)
    
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    
    return ' '.join(filtered_tokens)

# Calculate similarity using sentence transformers
def calculate_similarity(student_answer, teacher_answer):
    try:
        # Handle empty inputs
        if not student_answer or not teacher_answer:
            return 0.0
            
        # Preprocess the texts
        student_processed = preprocess_text(student_answer)
        teacher_processed = preprocess_text(teacher_answer)
        
        # Skip if either processed text is empty
        if not student_processed or not teacher_processed:
            return 0.0
            
        # Load the model (this will be cached after first load)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embeddings
        student_embedding = model.encode([student_processed])
        teacher_embedding = model.encode([teacher_processed])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(student_embedding, teacher_embedding)[0][0]
        
        # Convert to percentage
        similarity_percentage = float(similarity * 100)
        
        return similarity_percentage
    except Exception as e:
        print(f"Error calculating similarity: {str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 0.0

def main():
    try:
        # Read JSON data from stdin
        input_data = sys.stdin.read()
        data = json.loads(input_data)
        
        # Extract student and teacher answers
        student_answers = data.get('student', [])
        teacher_answers = data.get('teacher', [])
        
        results = []
        
        # Match questions and calculate similarity
        for student_item in student_answers:
            question_number = student_item.get('questionNumber')
            student_answer = student_item.get('answer', "")
            
            # Find the matching teacher answer
            teacher_answer = ""
            for teacher_item in teacher_answers:
                if teacher_item.get('questionNumber') == question_number:
                    teacher_answer = teacher_item.get('answer', "")
                    break
                    
            # Calculate similarity score
            similarity_score = calculate_similarity(student_answer, teacher_answer)
            
            # Add to results
            results.append({
                'questionNumber': question_number,
                'studentAnswer': student_answer,
                'teacherAnswer': teacher_answer,
                'similarityScore': similarity_score
            })
            
        # Return JSON results
        print(json.dumps(results))
        
    except Exception as e:
        error_message = {
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        print(json.dumps(error_message), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
