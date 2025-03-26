import os
import re
import PyPDF2
import pandas as pd
from collections import defaultdict
import spacy
from spacy.matcher import PhraseMatcher, Matcher
from datetime import datetime

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_file(file_path):
    """Extract text from PDF or TXT files."""
    text = ""
    try:
        if file_path.lower().endswith('.pdf'):
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + " "
        elif file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
    return text

def extract_age_nlp(text):
    """Extract age using NLP techniques with improved medical report patterns."""
    # First try direct pattern matching for medical reports
    medical_patterns = [
        r'Age:\s*(\d+)',
        r'Age\s*[\|:]\s*(\d+)',
        r'Patient\s*Age:\s*(\d+)',
        r'Age\s*[\|]\s*(\d+)',
        r'Age\s*(\d+)',
        r'DOB:\s*\d+/\d+/(\d{4})'  # Extract from birth year
    ]
    
    for pattern in medical_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if 'DOB' in pattern:  # Handle birth year
                birth_year = int(match.group(1))
                current_year = datetime.now().year
                return current_year - birth_year
            return int(match.group(1))
    
    # Then try spaCy NER
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "AGE":
            try:
                return int(ent.text)
            except ValueError:
                continue
    
    # Try more complex patterns if simple ones fail
    complex_patterns = [
        [{"LOWER": "age"}, {"IS_PUNCT": True, "OP": "?"}, {"LIKE_NUM": True}],
        [{"LIKE_NUM": True}, {"LOWER": {"IN": ["years", "yrs"]}}, {"LOWER": "old"}]
    ]
    
    matcher = Matcher(nlp.vocab)
    matcher.add("AGE_PATTERNS", complex_patterns)
    
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        for token in span:
            if token.like_num:
                return int(token.text)
    
    return None

def extract_drug_response_percentage(text):
    """Extract response percentage from text."""
    # First look for explicit success rates in medical context
    medical_response_patterns = [
        r'Success\s*Rate:\s*(\d+)%',
        r'Response\s*Rate:\s*(\d+)%',
        r'Efficacy:\s*(\d+)%',
        r'Improvement:\s*(\d+)%'
    ]
    
    for pattern in medical_response_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    
    # Fall back to general percentage extraction
    doc = nlp(text.lower())
    percentage_patterns = [
        [{"LIKE_NUM": True}, {"LOWER": {"IN": ["%", "percent", "percentage"]}}],
        [{"LOWER": "response"}, {"LOWER": "rate"}, {"LIKE_NUM": True}],
        [{"LOWER": "success"}, {"LOWER": "rate"}, {"LIKE_NUM": True}]
    ]
    
    matcher = Matcher(nlp.vocab)
    matcher.add("PERCENTAGE_PATTERNS", percentage_patterns)
    
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        for token in span:
            if token.like_num:
                return float(token.text)
    
    # If no percentage found, infer from positive language
    positive_terms = ["improved", "effective", "success", "positive", "reduced"]
    negative_terms = ["worsened", "ineffective", "failure", "negative", "increased"]
    
    pos_count = sum(1 for term in positive_terms if term in text.lower())
    neg_count = sum(1 for term in negative_terms if term in text.lower())
    
    if pos_count > neg_count:
        return 85.0  # Assume positive outcome
    elif neg_count > pos_count:
        return 35.0  # Assume negative outcome
    
    return 50.0  # Default neutral value

def process_reports(folder_path):
    """Process all reports in the given folder."""
    data = []
    
    valid_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.pdf', '.txt'))]
    
    if not valid_files:
        print("No PDF or TXT files found in the specified folder.")
        return None
    
    for file in valid_files:
        file_path = os.path.join(folder_path, file)
        text = extract_text_from_file(file_path)
        
        # Improved age extraction with medical report focus
        age = None
        age_match = re.search(
            r'(?:Age|Patient[\s-]*Age)\s*[:|]\s*(\d+)', 
            text, 
            re.IGNORECASE
        )
        if age_match:
            age = int(age_match.group(1))
        else:
            age = extract_age_nlp(text)
        
        response_percentage = extract_drug_response_percentage(text)
        
        # Robust age group classification including Below 20
        if isinstance(age, (int, float)):
            if age < 20:
                age_group = "Below 20"
            elif 20 <= age <= 60:
                age_group = "20 to 60"
            else:
                age_group = "Above 60"
        else:
            age_group = "Unknown"
            print(f"Note: Age not found in {file}")
        
        # Decision logic with medical context
        if response_percentage > 80:
            decision = "Accepted"
        else:
            decision = "Rejected"
        
        data.append({
            'file': file,
            'age': age,
            'age_group': age_group,
            'response_percentage': response_percentage,
            'decision': decision
        })
    
    return pd.DataFrame(data)

def calculate_statistics(df):
    """Calculate acceptance statistics by age group and overall."""
    if df is None or df.empty:
        return None
    
    # Initialize all age groups including Below 20
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
    
    # Calculate averages and final decisions for all age groups
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
        else:
            results['age_groups'][age_group]['avg_response'] = 0.0
            results['age_groups'][age_group]['final_decision'] = "Undetermined"
        
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
    
    total_overall = results['overall']['total']
    if total_overall > 0:
        results['overall']['acceptance_rate'] = round(
            (results['overall']['accepted'] / total_overall) * 100, 2
        )
    
    return results

def print_statistics(results):
    """Print the statistics in a readable format."""
    if not results:
        print("No results to display.")
        return
    
    print("\nDrug Experiment Response Statistics by Age Group")
    print("--------------------------------------------")
    # Print all age groups including Below 20
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
    """Main function to run the analysis."""
    print("Medical Report Analysis Tool")
    print("--------------------------")
    
    folder_path = input("Enter the path to the folder containing medical reports: ")
    
    if not os.path.isdir(folder_path):
        print("Error: The specified folder does not exist.")
        return
    
    print("\nProcessing reports...")
    df = process_reports(folder_path)
    
    if df is not None:
        results = calculate_statistics(df)
        print_statistics(results)
        
        save_csv = input("\nWould you like to save the detailed results to CSV? (y/n): ").strip().lower()
        if save_csv == 'y':
            csv_path = os.path.join(folder_path, "report_analysis_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()