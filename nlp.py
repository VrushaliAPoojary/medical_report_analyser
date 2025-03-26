import os
import re
import spacy
from spacy.matcher import Matcher
import pandas as pd
from collections import defaultdict
from datetime import datetime

# Load English NLP model with optimized pipeline
nlp = spacy.load("en_core_web_sm", disable=["parser"])
nlp.add_pipe("sentencizer")

def extract_text_from_file(file_path):
    """Extract text from files while preserving structure"""
    text = ""
    try:
        if file_path.lower().endswith('.pdf'):
            import PyPDF2
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = "\n".join(page.extract_text() for page in reader.pages)
        elif file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
    return text

def extract_age_nlp(text):
    """Enhanced age extraction using NLP patterns"""
    doc = nlp(text)
    
    # Pattern matching for different age formats
    matcher = Matcher(nlp.vocab)
    patterns = [
        [{"LOWER": "age"}, {"IS_PUNCT": True, "OP": "?"}, {"LIKE_NUM": True}],
        [{"LOWER": "aged"}, {"LIKE_NUM": True}],
        [{"LIKE_NUM": True}, {"LOWER": {"IN": ["years", "yrs"]}}],
        [{"LOWER": "dob"}, {"TEXT": {"REGEX": r"\d{1,2}/\d{1,2}/\d{4}"}}]
    ]
    matcher.add("AGE_PATTERNS", patterns)
    
    for match_id, start, end in matcher(doc):
        span = doc[start:end]
        for token in span:
            if token.like_num:
                if "dob" in span.text.lower():
                    birth_year = int(re.search(r"\d{4}", span.text).group())
                    return datetime.now().year - birth_year
                return int(token.text)
    
    # Entity recognition fallback
    for ent in doc.ents:
        if ent.label_ in ("AGE", "CARDINAL") and ent.text.isdigit():
            age = int(ent.text)
            if 1 <= age <= 120:
                return age
    
    return None

def extract_drug_response_nlp(text):
    """Response percentage extraction using NLP"""
    doc = nlp(text)
    
    # Percentage pattern matching
    matcher = Matcher(nlp.vocab)
    percent_patterns = [
        [{"LIKE_NUM": True}, {"LOWER": "%"}],
        [{"LOWER": {"IN": ["response", "efficacy"]}}, {"LOWER": "rate", "OP": "?"}, 
         {"IS_PUNCT": True, "OP": "?"}, {"LIKE_NUM": True}, {"LOWER": "%", "OP": "?"}]
    ]
    matcher.add("PERCENT_PATTERNS", percent_patterns)
    
    for match_id, start, end in matcher(doc):
        span = doc[start:end]
        for token in span:
            if token.like_num:
                # Check if percentage is near relevant terms
                window = doc[max(0, start-3):min(len(doc), end+3)]
                if any(term in window.text.lower() for term in ["response", "efficacy", "success"]):
                    return float(token.text)
    
    # Sentiment analysis fallback using POS tags
    pos_score = len([t for t in doc if t.lemma_ in ["improve", "effective", "success"]])
    neg_score = len([t for t in doc if t.lemma_ in ["worsen", "fail", "adverse"]])
    
    return 85.0 if pos_score > neg_score else 35.0 if neg_score > pos_score else 50.0

def classify_age_group(age):
    """Age group classification"""
    if age is None:
        return "Unknown"
    return next(
        (group for group, (min_age, max_age) in 
         {"Below 20": (0, 19), "20 to 60": (20, 60), "Above 60": (61, 120)}.items()
         if min_age <= age <= max_age),
        "Unknown"
    )

def process_reports_nlp(folder_path):
    """Process all reports using NLP pipeline"""
    data = []
    
    for file in [f for f in os.listdir(folder_path) if f.lower().endswith(('.pdf', '.txt'))]:
        file_path = os.path.join(folder_path, file)
        text = extract_text_from_file(file_path)
        
        age = extract_age_nlp(text)
        response = extract_drug_response_nlp(text)
        
        data.append({
            'file': file,
            'age': age,
            'age_group': classify_age_group(age),
            'response_percentage': response,
            'decision': "Accepted" if response > 80 else "Rejected"
        })
    
    return pd.DataFrame(data)

def calculate_statistics(df):
    """Calculate statistics from processed reports"""
    if df.empty:
        return None
    
    results = {
        'age_groups': {
            'Below 20': {'total': 0, 'accepted': 0, 'rejected': 0, 'response_sum': 0},
            '20 to 60': {'total': 0, 'accepted': 0, 'rejected': 0, 'response_sum': 0},
            'Above 60': {'total': 0, 'accepted': 0, 'rejected': 0, 'response_sum': 0},
            'Unknown': {'total': 0, 'accepted': 0, 'rejected': 0, 'response_sum': 0}
        },
        'overall': {'total': 0, 'accepted': 0, 'rejected': 0, 'response_sum': 0}
    }
    
    for _, row in df.iterrows():
        age_group = row['age_group']
        decision = row['decision']
        response = row['response_percentage'] or 0
        
        results['age_groups'][age_group]['total'] += 1
        results['age_groups'][age_group]['accepted' if decision == "Accepted" else 'rejected'] += 1
        results['age_groups'][age_group]['response_sum'] += response
        
        results['overall']['total'] += 1
        results['overall']['accepted' if decision == "Accepted" else 'rejected'] += 1
        results['overall']['response_sum'] += response
    
    # Calculate metrics
    for group in results['age_groups']:
        stats = results['age_groups'][group]
        if stats['total'] > 0:
            stats['acceptance_rate'] = round(stats['accepted'] / stats['total'] * 100, 2)
            stats['avg_response'] = round(stats['response_sum'] / stats['total'], 2)
            stats['final_decision'] = "Accepted" if stats['avg_response'] > 80 else "Rejected"
    
    # Overall calculations
    if results['overall']['total'] > 0:
        results['overall']['acceptance_rate'] = round(
            results['overall']['accepted'] / results['overall']['total'] * 100, 2
        )
        results['overall']['avg_response'] = round(
            results['overall']['response_sum'] / results['overall']['total'], 2
        )
        results['overall']['final_decision'] = "Accepted" if results['overall']['avg_response'] > 85 else "Rejected"
    
    return results

def print_statistics(results):
    """Display analysis results"""
    if not results:
        print("No results to display")
        return
    
    print("\nMedical Report Analysis Results")
    print("="*50)
    
    for group in ['Below 20', '20 to 60', 'Above 60', 'Unknown']:
        stats = results['age_groups'][group]
        if stats['total'] > 0:
            print(f"\n{group}:")
            print(f"  Total reports: {stats['total']}")
            print(f"  Accepted: {stats['accepted']} ({stats['acceptance_rate']:.1f}%)")
            print(f"  Average response: {stats['avg_response']:.1f}%")
            print(f"  Final decision: {stats['final_decision']}")
    
    print("\nOverall Statistics:")
    print(f"  Total reports: {results['overall']['total']}")
    print(f"  Acceptance rate: {results['overall']['acceptance_rate']:.1f}%")
    print(f"  Final decision: {results['overall']['final_decision']}")

def main():
    """Main application workflow"""
    print("Medical Report Analyzer")
    print("="*50)
    
    folder_path = input("Enter path to medical reports: ").strip()
    if not os.path.isdir(folder_path):
        print("Error: Invalid directory path")
        return
    
    print("\nProcessing reports...")
    df = process_reports_nlp(folder_path)
    
    if df is not None:
        results = calculate_statistics(df)
        print_statistics(results)
        
        if input("\nSave results to CSV? (y/n): ").lower() == 'y':
            csv_path = os.path.join(folder_path, "report_analysis.csv")
            df.to_csv(csv_path, index=False)
            print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()