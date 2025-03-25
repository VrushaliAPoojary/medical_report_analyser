import os
import re
import PyPDF2
import pandas as pd
from collections import defaultdict

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + " "
    except Exception as e:
        print(f"Error reading {pdf_path}: {str(e)}")
    return text

def extract_age(text):
    """Extract age from text using regex patterns."""
    # Common patterns for age extraction
    patterns = [
        r'Age:\s*(\d+)',
        r'Age\s*(\d+)',
        r'Patient\s*age:\s*(\d+)',
        r'(\d+)\s*years? old',
        r'DOB:\s*\d+/\d+/(\d{4})',  # Extract from birth year
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if 'DOB' in pattern:  # Handle birth year
                birth_year = int(match.group(1))
                current_year = pd.Timestamp.now().year
                return current_year - birth_year
            return int(match.group(1))
    
    # If no age found, return None
    return None

def extract_drug_decision(text):
    """Determine if drug experiment was accepted or rejected."""
    text_lower = text.lower()
    
    # Keywords for acceptance
    accept_keywords = [
        'accepted', 'approved', 'recommended', 'prescribed', 
        'administered', 'positive outcome', 'successful'
    ]
    
    # Keywords for rejection
    reject_keywords = [
        'rejected', 'not approved', 'contraindicated', 
        'adverse reaction', 'side effects', 'discontinued'
    ]
    
    accept_count = sum(1 for word in accept_keywords if word in text_lower)
    reject_count = sum(1 for word in reject_keywords if word in text_lower)
    
    if accept_count > reject_count:
        return "Accepted"
    elif reject_count > accept_count:
        return "Rejected"
    else:
        return "Undetermined"

def process_reports(folder_path):
    """Process all PDF reports in the given folder."""
    data = []
    
    # Get all PDF files in the folder
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in the specified folder.")
        return None
    
    for pdf_file in pdf_files:
        file_path = os.path.join(folder_path, pdf_file)
        text = extract_text_from_pdf(file_path)
        
        age = extract_age(text)
        decision = extract_drug_decision(text)
        
        # Determine age group
        if age is not None:
            if 0 <= age <= 20:
                age_group = "0-20"
            elif 21 <= age <= 50:
                age_group = "21-50"
            elif 51 <= age <= 80:
                age_group = "51-80"
            else:
                age_group = "Other"
        else:
            age_group = "Unknown"
        
        data.append({
            'file': pdf_file,
            'age': age,
            'age_group': age_group,
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
            '0-20': {'total': 0, 'accepted': 0, 'rejected': 0},
            '21-50': {'total': 0, 'accepted': 0, 'rejected': 0},
            '51-80': {'total': 0, 'accepted': 0, 'rejected': 0},
            'Other': {'total': 0, 'accepted': 0, 'rejected': 0},
            'Unknown': {'total': 0, 'accepted': 0, 'rejected': 0}
        },
        'overall': {'total': 0, 'accepted': 0, 'rejected': 0}
    }
    
    # Count decisions by age group
    for _, row in df.iterrows():
        age_group = row['age_group']
        decision = row['decision']
        
        results['age_groups'][age_group]['total'] += 1
        results['overall']['total'] += 1
        
        if decision == 'Accepted':
            results['age_groups'][age_group]['accepted'] += 1
            results['overall']['accepted'] += 1
        elif decision == 'Rejected':
            results['age_groups'][age_group]['rejected'] += 1
            results['overall']['rejected'] += 1
    
    # Calculate percentages
    for age_group in results['age_groups']:
        total = results['age_groups'][age_group]['total']
        accepted = results['age_groups'][age_group]['accepted']
        
        if total > 0:
            acceptance_rate = (accepted / total) * 100
        else:
            acceptance_rate = 0.0
        
        results['age_groups'][age_group]['acceptance_rate'] = round(acceptance_rate, 2)
    
    # Calculate overall acceptance rate
    total_overall = results['overall']['total']
    accepted_overall = results['overall']['accepted']
    
    if total_overall > 0:
        overall_acceptance = (accepted_overall / total_overall) * 100
    else:
        overall_acceptance = 0.0
    
    results['overall']['acceptance_rate'] = round(overall_acceptance, 2)
    
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
            print(f"{age_group} years:")
            print(f"  Total reports: {stats['total']}")
            print(f"  Accepted: {stats['accepted']}")
            print(f"  Rejected: {stats['rejected']}")
            print(f"  Acceptance rate: {stats['acceptance_rate']}%")
            print("")
    
    print("\nOverall Statistics")
    print("-----------------")
    print(f"Total reports processed: {results['overall']['total']}")
    print(f"Overall accepted: {results['overall']['accepted']}")
    print(f"Overall rejected: {results['overall']['rejected']}")
    print(f"Overall acceptance rate: {results['overall']['acceptance_rate']}%")

def main():
    """Main function to run the analysis."""
    print("Medical Report Analysis Tool")
    print("---------------------------")
    
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
        save_csv = input("\nWould you like to save the detailed results to CSV? (y/n): ")
        if save_csv.lower() == 'y':
            csv_path = os.path.join(folder_path, "report_analysis_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()