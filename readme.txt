https://app-app-kcwfqx3lbyzgkxkfuv9ujh.streamlit.app/


Resume Screening App
This repository hosts a Streamlit application designed to screen resumes using machine learning models. The project involves data preprocessing, model training, and evaluation to predict job categories based on resume content.

Dataset
The dataset used in this project is sourced from Kaggle, specifically the Resume Category Dataset. It consists of resumes labeled with various job categories.

Reading Dataset
The dataset is read into the application using pandas for further processing and analysis.

Understanding Categories
Initial exploration includes using value counts and plotting to understand the distribution of job categories in the dataset, providing insights into the data's structure.

Data Cleaning
The dataset undergoes rigorous cleaning processes to prepare it for machine learning tasks:

Removal of URLs: Any URLs present in the resume text are removed.
Handling Hashtags and Mentions: Hashtags and mentions are eliminated as they do not contribute to job category prediction.
Special Characters and Punctuations: Special characters and punctuations are removed to ensure clean text data.
Word Removal: Certain words like 'RT' and 'cc' are removed as they are not relevant to job classification.
Text Cleaning Function
A lambda function is employed to clean resume text efficiently, applying the aforementioned cleaning steps uniformly across the dataset.

Words to Categories Using Label Encoder
To facilitate machine learning model training, text data is encoded into numerical labels using sklearn's LabelEncoder. This step assigns a unique numeric identifier to each job category, making the data compatible for classification tasks.

Vectorization Using TF-IDF
Text vectorization is performed using TF-IDF (Term Frequency-Inverse Document Frequency). This technique converts resume text into numerical features while removing stopwords (common words that do not carry significant meaning in the context of job classification). TF-IDF ensures that words contributing more to the meaning of a document within the dataset are given higher weights.

Train-Test Split
The dataset is split into training and testing sets using sklearn's train_test_split function. This division allows for the evaluation of model performance on unseen data, ensuring the model's ability to generalize beyond the training dataset.

Machine Learning Models
Two classification algorithms are trained and evaluated on the processed data:

k-Nearest Neighbors (KNN) Classifier: A non-parametric method used for classification based on similarity measures between data points.
One-vs-Rest (OvR) Classifier: A strategy involving training multiple binary classifiers, one for each class, and selecting the class with the highest confidence score as the predicted class.
Accuracy
The models achieve an impressive accuracy of 99.48% in predicting job categories from resumes, demonstrating robust performance in classifying job applicants based on their resumes.
