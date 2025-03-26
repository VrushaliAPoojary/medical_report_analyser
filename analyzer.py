import os
import re
import PyPDF2
import pandas as pd
import spacy
from collections import defaultdict

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_file(file_path):
    """Extract text from a PDF or TXT file."""
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

def extract_age(text):
    """Extract age from text using NLP and regex patterns."""
    doc = nlp(text)
    
    # Look for age patterns using NLP
    for ent in doc.ents:
        if ent.label_ == "AGE":
            return int(ent.text)
    
    # Fallback to regex if NLP doesn't find age
    patterns = [
        r'Age:\s*(\d+)',
        r'Age\s*(\d+)',
        r'Patient\s*age:\s*(\d+)',
        r'(\d+)\s*years? old',
        r'DOB:\s*\d+/\d+/(\d{4})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if 'DOB' in pattern:
                birth_year = int(match.group(1))
                current_year = pd.Timestamp.now().year
                return current_year - birth_year
            return int(match.group(1))
    
    return None

def analyze_drug_response(text):
    """Analyze drug response using NLP and calculate acceptance percentage."""
    doc = nlp(text.lower())
    
    # Define keywords with weights
    acceptance_terms = {
        'accepted': 1.0, 'approved': 1.0, 'effective': 0.9, 
        'successful': 0.8, 'positive': 0.7, 'improved': 0.6
    }
    
    rejection_terms = {
        'rejected': 1.0, 'adverse': 0.9, 'side effect': 0.8,
        'contraindicated': 0.7, 'discontinued': 0.6, 'negative': 0.5
    }
    
    # Calculate scores
    accept_score = sum(acceptance_terms.get(token.text, 0) for token in doc)
    reject_score = sum(rejection_terms.get(token.text, 0) for token in doc)
    
    # Calculate percentage
    total_score = accept_score + reject_score
    if total_score > 0:
        acceptance_percentage = (accept_score / total_score) * 100
    else:
        acceptance_percentage = 0
    
    return acceptance_percentage

def process_reports(folder_path):
    """Process all reports in the given folder."""
    data = []
    
    # Get all PDF and TXT files in the folder
    valid_extensions = ('.pdf', '.txt')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    
    if not files:
        print("No PDF or TXT files found in the specified folder.")
        return None
    
    for file in files:
        file_path = os.path.join(folder_path, file)
        text = extract_text_from_file(file_path)
        
        age = extract_age(text)
        acceptance_percentage = analyze_drug_response(text)
        
        # Determine age group based on new criteria
        if age is not None:
            if age < 20:
                age_group = "Below 20"
            elif 20 <= age <= 60:
                age_group = "20 to 60"
            else:
                age_group = "Above 60"
        else:
            age_group = "Unknown"
        
        # Determine decision based on percentage
        if age_group != "Unknown":
            decision = "Accepted" if acceptance_percentage >= 80 else "Rejected"
        else:
            decision = "Undetermined"
        
        data.append({
            'file': file,
            'age': age,
            'age_group': age_group,
            'acceptance_percentage': round(acceptance_percentage, 2),
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
            'Below 20': {'total': 0, 'accepted': 0, 'rejected': 0, 'avg_percentage': 0},
            '20 to 60': {'total': 0, 'accepted': 0, 'rejected': 0, 'avg_percentage': 0},
            'Above 60': {'total': 0, 'accepted': 0, 'rejected': 0, 'avg_percentage': 0},
            'Unknown': {'total': 0, 'accepted': 0, 'rejected': 0, 'avg_percentage': 0}
        },
        'overall': {'total': 0, 'accepted': 0, 'rejected': 0, 'avg_percentage': 0}
    }
    
    # Calculate sums for percentages
    percentage_sums = defaultdict(float)
    
    # Count decisions by age group
    for _, row in df.iterrows():
        age_group = row['age_group']
        decision = row['decision']
        percentage = row['acceptance_percentage']
        
        results['age_groups'][age_group]['total'] += 1
        results['overall']['total'] += 1
        
        percentage_sums[age_group] += percentage
        percentage_sums['overall'] += percentage
        
        if decision == 'Accepted':
            results['age_groups'][age_group]['accepted'] += 1
            results['overall']['accepted'] += 1
        elif decision == 'Rejected':
            results['age_groups'][age_group]['rejected'] += 1
            results['overall']['rejected'] += 1
    
    # Calculate average percentages
    for age_group in results['age_groups']:
        total = results['age_groups'][age_group]['total']
        if total > 0:
            results['age_groups'][age_group]['avg_percentage'] = round(percentage_sums[age_group] / total, 2)
    
    # Calculate overall average percentage
    if results['overall']['total'] > 0:
        results['overall']['avg_percentage'] = round(percentage_sums['overall'] / results['overall']['total'], 2)
    
    # Determine overall decision based on 85% threshold
    overall_decision = "Accepted" if results['overall']['avg_percentage'] >= 85 else "Rejected"
    results['overall']['decision'] = overall_decision
    
    return results

def print_statistics(results):
    """Print the statistics in a readable format."""
    if not results:
        print("No results to display.")
        return
    
    print("\nDrug Experiment Acceptance Statistics by Age Group")
    print("------------------------------------------------")
    for age_group, stats in results['age_groups'].items():
        if stats['total'] > 0:
            print(f"{age_group}:")
            print(f"  Total reports: {stats['total']}")
            print(f"  Accepted: {stats['accepted']}")
            print(f"  Rejected: {stats['rejected']}")
            print(f"  Average acceptance percentage: {stats['avg_percentage']}%")
            print()
    
    print("\nOverall Statistics")
    print("-----------------")
    print(f"Total reports processed: {results['overall']['total']}")
    print(f"Overall accepted: {results['overall']['accepted']}")
    print(f"Overall rejected: {results['overall']['rejected']}")
    print(f"Overall average acceptance percentage: {results['overall']['avg_percentage']}%")
    print(f"Final decision: {results['overall']['decision']}")

def main():
    """Main function to run the analysis."""
    print("Medical Report Analysis Tool")
    print("---------------------------")
    
    # Get folder path from user
    folder_path = input("Enter the path to the folder containing medical reports: ").strip()
    
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
        
        # Save results to CSV
        csv_path = os.path.join(folder_path, "report_analysis_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nDetailed results saved to {csv_path}")

if __name__ == "__main__":
    # Install spaCy model if not already installed
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy English model...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    main()