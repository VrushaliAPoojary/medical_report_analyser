import os
import re
import spacy
from spacy.matcher import Matcher
import pandas as pd
from collections import defaultdict
from datetime import datetime

# Load English NLP model
nlp = spacy.load("en_core_web_sm", disable=["parser"])
nlp.add_pipe("sentencizer")

def extract_text_from_file(file_path):
    """Extract text from PDF or TXT files using NLP-aware methods"""
    text = ""
    try:
        if file_path.lower().endswith('.pdf'):
            import PyPDF2
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        elif file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
    return text

def extract_age_nlp(text):
    """Pure NLP age extraction with medical context awareness"""
    doc = nlp(text)
    
    # Custom NER patterns for medical reports
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
    
    # Enhanced NER with custom rules
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
    
    # Custom response pattern matcher
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
    
    # Sentiment-based estimation
    positive_terms = ["improved", "effective", "success", "positive", "reduced"]
    negative_terms = ["worsened", "ineffective", "failure", "negative", "increased"]
    
    doc_lower = nlp(text.lower())
    pos_score = sum(1 for term in positive_terms if term in doc_lower.text)
    neg_score = sum(1 for term in negative_terms if term in doc_lower.text)
    
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

def process_reports_nlp(folder_path):
    """Pure NLP processing pipeline"""
    data = []
    
    for file in [f for f in os.listdir(folder_path) if f.lower().endswith(('.pdf', '.txt'))]:
        file_path = os.path.join(folder_path, file)
        text = extract_text_from_file(file_path)
        doc = nlp(text)
        
        age = extract_age_nlp(text)
        response = extract_drug_response_nlp(text)
        age_group = classify_age_group(age)
        
        decision = "Accepted" if response > 80 else "Rejected"
        
        data.append({
            'file': file,
            'age': age,
            'age_group': age_group,
            'response_percentage': response,
            'decision': decision
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
    
    # Calculate averages and final decisions
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
    
    # Overall calculations
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

def print_statistics(results):
    """Print the statistics in a readable format"""
    if not results:
        print("No results to display.")
        return
    
    print("\nDrug Experiment Response Statistics by Age Group")
    print("--------------------------------------------")
    for age_group in ['Below 20', '20 to 60', 'Above 60', 'Unknown']:
        stats = results['age_groups'][age_group]
        if stats['total'] > 0:
            print(f"{age_group}:")
            print(f"  Total reports: {stats['total']}")
            print(f"  Accepted: {stats['accepted']}")
            print(f"  Rejected: {stats['rejected']}")
            print(f"  Average response percentage: {stats['avg_response']}%")
            print(f"  Acceptance rate: {stats['acceptance_rate']}%")
            print(f"  Final decision: {stats['final_decision']}")
            print()
    
    print("\nOverall Statistics")
    print("-----------------")
    print(f"Total reports processed: {results['overall']['total']}")
    print(f"Overall accepted: {results['overall']['accepted']}")
    print(f"Overall rejected: {results['overall']['rejected']}")
    print(f"Overall average response: {results['overall']['avg_response']}%")
    print(f"Overall acceptance rate: {results['overall']['acceptance_rate']}%")
    print(f"Final overall decision: {results['overall']['final_decision']}")

def main():
    """Main function to run the analysis"""
    print("Medical Report Analysis")
    print("--------------------------------------")
    
    folder_path = input("Enter the path to the folder containing medical reports: ")
    
    if not os.path.isdir(folder_path):
        print("Error: The specified folder does not exist.")
        return
    
    print("\nProcessing reports ...")
    df = process_reports_nlp(folder_path)
    
    if df is not None:
        results = calculate_statistics(df)
        print_statistics(results)
        
        save_csv = input("\nWould you like to save the detailed results to CSV? (y/n): ").strip().lower()
        if save_csv == 'y':
            csv_path = os.path.join(folder_path, "nlp_analysis_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()