import streamlit as st
import pickle
import re
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load models
clf = pickle.load(open('classifier.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# CSS for custom styling
st.markdown("""
    <style>
    .main {
        background-color: #ADD8E6;
        color: #ADD8E6;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .stTitle {
        font-family: Arial, Helvetica, sans-serif;
        color: #fff;
        text-align: center;
    }
    </style>"""
, unsafe_allow_html=True)

def clean_resume(resume_text):
    """Clean the resume text by removing URLs, mentions, special characters, and extra spaces."""
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

def main():
    """Main function to run the Streamlit app."""
    st.markdown("<div class='heading'><h1 class='stTitle'>Resume Screening App</h1></div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        
        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and Fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.success(f"Predicted Category: {category_name}")

if __name__ == "__main__":
    main()
