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
    """Extract age using NLP techniques."""
    doc = nlp(text)
    
    # Pattern 1: Direct age mentions
    for ent in doc.ents:
        if ent.label_ == "AGE":
            return int(ent.text)
    
    # Pattern 2: Age-related phrases
    age_patterns = [
        [{"LOWER": "age"}, {"IS_PUNCT": True, "OP": "?"}, {"LIKE_NUM": True}],
        [{"LIKE_NUM": True}, {"LOWER": {"IN": ["years", "yrs"]}}, {"LOWER": "old"}]
    ]
    
    matcher = Matcher(nlp.vocab)
    matcher.add("AGE_PATTERNS", age_patterns)
    
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        for token in span:
            if token.like_num:
                return int(token.text)
    
    # Pattern 3: Extract from DOB if present
    dob_pattern = [{"LOWER": "dob"}, {"IS_PUNCT": True}, {"TEXT": {"REGEX": r"\d{1,2}/\d{1,2}/\d{4}"}}]
    matcher.add("DOB_PATTERN", [dob_pattern])
    matches = matcher(doc)
    
    for match_id, start, end in matches:
        dob_text = doc[end-1].text
        birth_year = int(dob_text.split("/")[-1])
        current_year = datetime.now().year
        return current_year - birth_year
    
    return None

def extract_drug_response_percentage(text):
    """Extract response percentage from text."""
    doc = nlp(text.lower())
    
    # Look for percentage patterns
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
    
    # If no percentage found, return None
    return None

def process_reports(folder_path):
    """Process all reports in the given folder."""
    data = []
    
    # Get all PDF and TXT files in the folder
    valid_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.pdf', '.txt'))]
    
    if not valid_files:
        print("No PDF or TXT files found in the specified folder.")
        return None
    
    for file in valid_files:
        file_path = os.path.join(folder_path, file)
        text = extract_text_from_file(file_path)
        
        age = extract_age_nlp(text)
        response_percentage = extract_drug_response_percentage(text)
        
        # Determine age group
        if age is not None:
            if age < 20:
                age_group = "Below 20"
            elif 20 <= age <= 60:
                age_group = "20 to 60"
            else:
                age_group = "Above 60"
        else:
            age_group = "Unknown"
        
        # Determine decision based on response percentage
        if response_percentage is not None:
            if response_percentage > 80:
                decision = "Accepted"
            else:
                decision = "Rejected"
        else:
            decision = "Undetermined"
        
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
    
    # Initialize results dictionary
    results = {
        'age_groups': {
            'Below 20': {'total': 0, 'accepted': 0, 'rejected': 0, 'avg_response': 0},
            '20 to 60': {'total': 0, 'accepted': 0, 'rejected': 0, 'avg_response': 0},
            'Above 60': {'total': 0, 'accepted': 0, 'rejected': 0, 'avg_response': 0},
            'Unknown': {'total': 0, 'accepted': 0, 'rejected': 0, 'avg_response': 0}
        },
        'overall': {'total': 0, 'accepted': 0, 'rejected': 0, 'avg_response': 0}
    }
    
    # Calculate sums for averages
    response_sums = defaultdict(float)
    response_counts = defaultdict(int)
    
    # Count decisions by age group
    for _, row in df.iterrows():
        age_group = row['age_group']
        decision = row['decision']
        response = row['response_percentage'] if pd.notna(row['response_percentage']) else 0
        
        results['age_groups'][age_group]['total'] += 1
        results['overall']['total'] += 1
        
        if decision == 'Accepted':
            results['age_groups'][age_group]['accepted'] += 1
            results['overall']['accepted'] += 1
        elif decision == 'Rejected':
            results['age_groups'][age_group]['rejected'] += 1
            results['overall']['rejected'] += 1
        
        if pd.notna(row['response_percentage']):
            response_sums[age_group] += response
            response_counts[age_group] += 1
            response_sums['overall'] += response
            response_counts['overall'] += 1
    
    # Calculate averages and acceptance rates
    for age_group in results['age_groups']:
        total = results['age_groups'][age_group]['total']
        accepted = results['age_groups'][age_group]['accepted']
        
        if response_counts[age_group] > 0:
            avg_response = response_sums[age_group] / response_counts[age_group]
        else:
            avg_response = 0.0
        
        results['age_groups'][age_group]['avg_response'] = round(avg_response, 2)
        
        if total > 0:
            acceptance_rate = (accepted / total) * 100
        else:
            acceptance_rate = 0.0
        
        results['age_groups'][age_group]['acceptance_rate'] = round(acceptance_rate, 2)
    
    # Calculate overall averages
    if response_counts['overall'] > 0:
        overall_avg = response_sums['overall'] / response_counts['overall']
    else:
        overall_avg = 0.0
    
    results['overall']['avg_response'] = round(overall_avg, 2)
    
    total_overall = results['overall']['total']
    accepted_overall = results['overall']['accepted']
    
    if total_overall > 0:
        overall_acceptance = (accepted_overall / total_overall) * 100
    else:
        overall_acceptance = 0.0
    
    results['overall']['acceptance_rate'] = round(overall_acceptance, 2)
    
    # Determine final decisions based on thresholds
    for age_group in results['age_groups']:
        if results['age_groups'][age_group]['avg_response'] > 80:
            results['age_groups'][age_group]['final_decision'] = "Accepted"
        else:
            results['age_groups'][age_group]['final_decision'] = "Rejected"
    
    if results['overall']['avg_response'] > 85:
        results['overall']['final_decision'] = "Accepted"
    else:
        results['overall']['final_decision'] = "Rejected"
    
    return results

def print_statistics(results):
    """Print the statistics in a readable format."""
    if not results:
        print("No results to display.")
        return
    
    print("\nDrug Experiment Response Statistics by Age Group")
    print("--------------------------------------------")
    for age_group, stats in results['age_groups'].items():
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
    
    # Get folder path from user
    folder_path = input("Enter the path to the folder containing medical reports: ")
    
    # Verify folder exists
    if not os.path.isdir(folder_path):
        print("Error: The specified folder does not exist.")
        return
    
    # Process reports
    print("\nProcessing reports...")
    df = process_reports(folder_path)
    
    if df is not None:
        # Calculate statistics
        results = calculate_statistics(df)
        
        # Print results
        print_statistics(results)
        
        # Optional: Save results to CSV
        save_csv = input("\nWould you like to save the detailed results to CSV? (y/n): ").strip().lower()
        if save_csv == 'y':
            csv_path = os.path.join(folder_path, "report_analysis_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()