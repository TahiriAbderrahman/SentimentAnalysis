Sentiment Analysis Project
This project performs sentiment analysis on text data from XML files, both on a sentence level and an aspect-based level. The project utilizes various NLP techniques such as tokenization, POS tagging, lemmatization, and sentiment analysis using SentiWordNet. Additionally, it includes feature extraction using TF-IDF and classification using a Naive Bayes classifier.

Requirements
Before running the script, make sure you have the following Python packages installed:

nltk
xml.etree.ElementTree
sklearn
numpy
matplotlib
beautifulsoup4
You can install the required packages using the following command:

bash
Copier le code
pip install nltk scikit-learn numpy matplotlib beautifulsoup4
Also, download the necessary NLTK data by running:

python
Copier le code
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('sentiwordnet')
Files
sentiment_analysis.py: Main script to run the sentiment analysis.
Restaurants_Train.xml: Training data for the restaurant dataset.
Laptops_Train.xml: Training data for the laptop dataset.
Restaurants_Test_NoLabels.xml: Test data for the restaurant dataset.
Laptops_Test_NoLabels.xml: Test data for the laptop dataset.
Usage
Running the Script
Save the script as sentiment_analysis.py.

Ensure the XML files (Restaurants_Train.xml, Laptops_Train.xml, Restaurants_Test_NoLabels.xml, Laptops_Test_NoLabels.xml) are in the same directory as the script.

Open a terminal or command prompt and navigate to the directory containing the script and XML files.

Run the script using the following command:

bash
Copier le code
python sentiment_analysis.py
Functions
load_data(file_path): Loads data from the specified XML file.
load_data_with_polarity(file_path): Loads data from the specified XML file, including aspect polarity.
preprocess(text): Tokenizes, lemmatizes, and POS tags the text.
sentiment_analysis(tokens): Performs sentiment analysis using SentiWordNet.
extract_features(sentences): Extracts features using TF-IDF vectorization.
train_and_predict(X_train, y_train, X_test): Trains a Naive Bayes classifier and predicts the sentiment.
Analysis Types
Sentence-based Sentiment Analysis: Analyzes the overall sentiment of sentences in the dataset.
Aspect-based Sentiment Analysis: Analyzes the sentiment of specific aspects within sentences in the dataset.
Visualization
The script includes a simple bar chart visualization to show the distribution of positive and negative sentiments.

Examples
Sentence-based Sentiment Analysis:

python
Copier le code
main('Restaurants_Train.xml')
main('Laptops_Train.xml')
Aspect-based Sentiment Analysis:

python
Copier le code
main_aspect_based_sentiment_analysis('Restaurants_Train.xml', 'Restaurants_Test_NoLabels.xml')
main_aspect_based_sentiment_analysis('Laptops_Train.xml', 'Laptops_Test_NoLabels.xml')
Output
The script prints the sentiment distribution (positive and negative) and aspect-based sentiment predictions to the console.
A bar chart showing the sentiment distribution is displayed.
Notes
Make sure the XML files are properly formatted and contain the necessary data.
Adjust the preprocessing and feature extraction steps if needed to suit your specific data and requirements.
License
This project is licensed under the MIT License.

Acknowledgments
The nltk library for NLP tasks.
The sklearn library for machine learning tasks.
The creators of the XML datasets used in this project.
