# Sentiment Analysis on Movie Reviews

## Project Overview
This project focuses on sentiment analysis of movie reviews using Natural Language Processing (NLP) and Machine Learning techniques. The dataset consists of 1,000 movie reviews, each labeled as either positive (`pos`) or negative (`neg`). 
The goal is to preprocess the text data and build classification models to predict the sentiment of reviews accurately.

## Technologies Used
- **Python**
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **NLTK (Natural Language Toolkit)**: Text preprocessing (stopword removal, lemmatization)
- **Scikit-learn**: Machine learning models (Logistic Regression, Random Forest, SVM)
- **WordCloud**: Word frequency visualization

## Dataset
The dataset consists of two columns:
- `text`: The movie review content
- `sentiment`: The label indicating whether the review is positive (`pos`) or negative (`neg`)

### Data Preprocessing
1. **Loading and Cleaning Data**:
   - Removed null values
   - Verified there are no duplicate entries
2. **Text Preprocessing**:
   - Converted text to lowercase
   - Removed punctuation
   - Removed stopwords using NLTK
   - Applied lemmatization
3. **Visualization**:
   - Generated word clouds for both positive and negative reviews

## Feature Engineering
- Applied **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical vectors with a max feature size of 5000.

## Model Training & Evaluation
Three machine learning models were implemented:
1. **Logistic Regression**
   - Accuracy: **83.5%**
2. **Random Forest Classifier**
   - Accuracy: **80%**
3. **Support Vector Machine (SVM) with Linear Kernel**
   - Accuracy: **85.5%**

### Performance Metrics
Each model was evaluated using:
- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-score)

### Hyperparameter Tuning
- **Random Forest Classifier** was fine-tuned using Grid Search Cross-Validation (GridSearchCV) to optimize hyperparameters like:
  - Number of estimators
  - Maximum depth
  - Minimum samples split/leaf
  - Bootstrap sampling

## Results & Conclusion
- SVM performed the best with an accuracy of **85.5%**, followed by Logistic Regression (83.5%) and Random Forest (80%).
- The project demonstrates the effectiveness of NLP preprocessing and machine learning models in sentiment analysis.

## Future Improvements
- Experiment with deep learning models such as LSTMs or Transformers
- Incorporate more advanced NLP techniques like word embeddings (Word2Vec, GloVe)
- Expand the dataset for better generalization

## Installation
```bash
git clone https://github.com/yakupzengin/datascience-ml-projects.git