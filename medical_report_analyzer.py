import streamlit as st
import os
import re
import spacy
import nltk
from spacy.matcher import Matcher
import pandas as pd
from collections import defaultdict
from datetime import datetime
import PyPDF2
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Load English NLP model
nlp = spacy.load("en_core_web_sm", disable=["parser"])
nlp.add_pipe("sentencizer")

# Set page config
st.set_page_config(page_title="Medical Report Analyzer", layout="wide")

# Custom CSS for better appearance
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 4px;
            border: none;
            padding: 8px 16px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stFileUploader>div>div>div>button {
            background-color: #2196F3;
            color: white;
        }
        .stFileUploader>div>div>div>button:hover {
            background-color: #0b7dda;
        }
        .result-card {
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
            background-color: #f9f9f9;
        }
        .stat-card {
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
            background-color: #f0f8ff;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("Medical Report Analyzer")
st.markdown("""
    Upload medical reports (PDF or TXT) to analyze drug response statistics by age group.
    The system will extract key information and provide acceptance/rejection decisions.
""")

# File uploader
uploaded_files = st.file_uploader(
    "Choose medical report files (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# All the functions from your original code (unchanged)
def extract_text_from_file(file_path):
    """Extract text from PDF or TXT files using NLP-aware methods"""
    text = ""
    try:
        if file_path.lower().endswith('.pdf'):
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        elif file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
    except Exception as e:
        st.error(f"Error reading {file_path}: {str(e)}")
    return text

def analyze_document_nlp(text):
    """Analyze document using multiple NLP techniques"""
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    
    doc = nlp(text)
    
    return {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'filtered_words': filtered_words,
        'entities': [(ent.text, ent.label_) for ent in doc.ents]
    }

def extract_age_nlp(text):
    """Pure NLP age extraction with medical context awareness"""
    doc = nlp(text)
    
    age_patterns = [
        [{"LOWER": "age"}, {"IS_PUNCT": True, "OP": "?"}, {"LIKE_NUM": True}],
        [{"LOWER": "patient"}, {"LOWER": "age"}, {"IS_PUNCT": True, "OP": "?"}, {"LIKE_NUM": True}],
        [{"LIKE_NUM": True}, {"LOWER": {"IN": ["years", "yrs"]}}, {"LOWER": "old", "OP": "?"}],
        [{"LOWER": "dob"}, {"IS_PUNCT": True}, {"TEXT": {"REGEX": r"\d{1,2}/\d{1,2}/(\d{4})"}}]
    ]
    
    matcher = Matcher(nlp.vocab)
    matcher.add("AGE_PATTERNS", age_patterns)
    
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        if nlp.vocab.strings[match_id] == "AGE_PATTERNS":
            for token in span:
                if token.like_num:
                    if "dob" in span.text.lower():
                        birth_year = int(re.search(r"\d{4}", span.text).group())
                        return datetime.now().year - birth_year
                    return int(token.text)
    
    for ent in doc.ents:
        if ent.label_ == "AGE" or (ent.label_ == "CARDINAL" and 1 <= int(ent.text) <= 120):
            try:
                return int(ent.text)
            except:
                continue
    
    return None

def extract_drug_response_nlp(text):
    """NLP-based response percentage extraction"""
    doc = nlp(text)
    
    response_patterns = [
        [{"LOWER": {"IN": ["response", "success", "efficacy"]}}, 
         {"LOWER": "rate", "OP": "?"}, 
         {"IS_PUNCT": True, "OP": "?"}, 
         {"LIKE_NUM": True}, 
         {"LOWER": "%", "OP": "?"}],
        [{"LIKE_NUM": True}, {"LOWER": "%"}, {"LOWER": {"IN": ["response", "success", "improvement"]}}]
    ]
    
    matcher = Matcher(nlp.vocab)
    matcher.add("RESPONSE_PATTERNS", response_patterns)
    
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        for token in span:
            if token.like_num:
                return float(token.text)
    
    positive_terms = ["improved", "effective", "success", "positive", "reduced"]
    negative_terms = ["worsened", "ineffective", "failure", "negative", "increased"]
    
    words = word_tokenize(text.lower())
    pos_score = sum(1 for term in positive_terms if term in words)
    neg_score = sum(1 for term in negative_terms if term in words)
    
    if pos_score > neg_score:
        return 85.0
    elif neg_score > pos_score:
        return 35.0
    return 50.0

def classify_age_group(age):
    """Age group classification"""
    if age is None:
        return "Unknown"
    if age < 20:
        return "Below 20"
    elif 20 <= age <= 60:
        return "20 to 60"
    else:
        return "Above 60"

def process_reports_nlp(uploaded_files, temp_dir):
    """Pure NLP processing pipeline"""
    data = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        text = extract_text_from_file(file_path)
        
        # Document analysis using NLP
        analysis = analyze_document_nlp(text)
        
        age = extract_age_nlp(text)
        response = extract_drug_response_nlp(text)
        age_group = classify_age_group(age)
        
        decision = "Accepted" if response > 80 else "Rejected"
        
        data.append({
            'file': uploaded_file.name,
            'age': age,
            'age_group': age_group,
            'response_percentage': response,
            'decision': decision,
            'word_count': analysis['word_count'],
            'entities_found': len(analysis['entities'])
        })
    
    return pd.DataFrame(data)

def calculate_statistics(df):
    """Calculate acceptance statistics by age group and overall"""
    if df is None or df.empty:
        return None
    
    results = {
        'age_groups': {
            'Below 20': {'total': 0, 'accepted': 0, 'rejected': 0, 'avg_response': 0},
            '20 to 60': {'total': 0, 'accepted': 0, 'rejected': 0, 'avg_response': 0},
            'Above 60': {'total': 0, 'accepted': 0, 'rejected': 0, 'avg_response': 0},
            'Unknown': {'total': 0, 'accepted': 0, 'rejected': 0, 'avg_response': 0}
        },
        'overall': {'total': 0, 'accepted': 0, 'rejected': 0, 'avg_response': 0}
    }
    
    response_sums = defaultdict(float)
    response_counts = defaultdict(int)
    
    for _, row in df.iterrows():
        age_group = row['age_group']
        decision = row['decision']
        response = row['response_percentage'] if pd.notna(row['response_percentage']) else 0
        
        results['age_groups'][age_group]['total'] += 1
        results['overall']['total'] += 1
        
        if decision == 'Accepted':
            results['age_groups'][age_group]['accepted'] += 1
            results['overall']['accepted'] += 1
        else:
            results['age_groups'][age_group]['rejected'] += 1
            results['overall']['rejected'] += 1
        
        if pd.notna(row['response_percentage']):
            response_sums[age_group] += response
            response_counts[age_group] += 1
            response_sums['overall'] += response
            response_counts['overall'] += 1
    
    for age_group in results['age_groups']:
        total = results['age_groups'][age_group]['total']
        accepted = results['age_groups'][age_group]['accepted']
        
        if response_counts[age_group] > 0:
            avg_response = response_sums[age_group] / response_counts[age_group]
            results['age_groups'][age_group]['avg_response'] = round(avg_response, 2)
            
            if avg_response > 80:
                results['age_groups'][age_group]['final_decision'] = "Accepted"
            else:
                results['age_groups'][age_group]['final_decision'] = "Rejected"
        
        if total > 0:
            results['age_groups'][age_group]['acceptance_rate'] = round((accepted / total) * 100, 2)
    
    if response_counts['overall'] > 0:
        overall_avg = response_sums['overall'] / response_counts['overall']
        results['overall']['avg_response'] = round(overall_avg, 2)
        
        if overall_avg > 85:
            results['overall']['final_decision'] = "Accepted"
        else:
            results['overall']['final_decision'] = "Rejected"
    
    if results['overall']['total'] > 0:
        results['overall']['acceptance_rate'] = round(
            (results['overall']['accepted'] / results['overall']['total']) * 100, 2
        )
    
    return results

def display_statistics(results):
    """Display the statistics in Streamlit"""
    if not results:
        st.warning("No results to display.")
        return
    
    st.subheader("Drug Experiment Response Statistics")
    
    # Overall statistics
    with st.expander("Overall Statistics", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reports", results['overall']['total'])
        col2.metric("Accepted", results['overall']['accepted'])
        col3.metric("Rejected", results['overall']['rejected'])
        col4.metric("Acceptance Rate", f"{results['overall']['acceptance_rate']}%")
        
        col5, col6 = st.columns(2)
        col5.metric("Average Response", f"{results['overall']['avg_response']}%")
        col6.metric("Final Decision", results['overall']['final_decision'])
    
    # Age group statistics
    st.subheader("Statistics by Age Group")
    for age_group in ['Below 20', '20 to 60', 'Above 60', 'Unknown']:
        stats = results['age_groups'][age_group]
        if stats['total'] > 0:
            with st.expander(f"{age_group} Group", expanded=False):
                cols = st.columns(4)
                cols[0].metric("Total Reports", stats['total'])
                cols[1].metric("Accepted", stats['accepted'])
                cols[2].metric("Rejected", stats['rejected'])
                cols[3].metric("Acceptance Rate", f"{stats['acceptance_rate']}%")
                
                cols2 = st.columns(2)
                cols2[0].metric("Average Response", f"{stats['avg_response']}%")
                cols2[1].metric("Final Decision", stats['final_decision'])

# Main processing
if uploaded_files:
    # Create a temporary directory to store uploaded files
    temp_dir = "temp_uploaded_files"
    os.makedirs(temp_dir, exist_ok=True)
    
    with st.spinner("Processing reports..."):
        try:
            # Process the uploaded files
            df = process_reports_nlp(uploaded_files, temp_dir)
            
            if df is not None and not df.empty:
                st.success("Analysis completed successfully!")
                
                # Show individual report results
                st.subheader("Individual Report Results")
                st.dataframe(df)
                
                # Calculate and display statistics
                results = calculate_statistics(df)
                display_statistics(results)
                
                # Add download button for results
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="medical_report_analysis_results.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No valid data could be extracted from the uploaded files.")
        
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
        
        finally:
            # Clean up temporary files
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                if os.path.exists(file_path):
                    os.remove(file_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
else:
    st.info("Please upload medical report files to begin analysis.")