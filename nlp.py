import os
import re
import spacy
from spacy.matcher import Matcher
import pandas as pd
from collections import defaultdict
from datetime import datetime
import PyPDF2

# Load English NLP model
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer")

class MedicalReportProcessor:
    def __init__(self):
        self.matcher = Matcher(nlp.vocab)
        self._initialize_patterns()
        
    def _initialize_patterns(self):
        # Age patterns
        age_patterns = [
            [{"LOWER": "age"}, {"IS_PUNCT": True, "OP": "?"}, {"LIKE_NUM": True}],
            [{"LOWER": "patient"}, {"LOWER": "age"}, {"IS_PUNCT": True, "OP": "?"}, {"LIKE_NUM": True}],
            [{"LIKE_NUM": True}, {"LOWER": {"IN": ["years", "yrs"]}}, {"LOWER": "old", "OP": "?"}],
            [{"LOWER": "dob"}, {"IS_PUNCT": True}, {"TEXT": {"REGEX": r"\d{1,2}/\d{1,2}/(\d{4})"}}]
        ]
        self.matcher.add("AGE_PATTERNS", age_patterns)
        
        # Response patterns
        response_patterns = [
            [{"LOWER": {"IN": ["response", "success", "efficacy"]}}, 
             {"LOWER": "rate", "OP": "?"}, 
             {"IS_PUNCT": True, "OP": "?"}, 
             {"LIKE_NUM": True}, 
             {"LOWER": "%", "OP": "?"}],
            [{"LIKE_NUM": True}, {"LOWER": "%"}, {"LOWER": {"IN": ["response", "success", "improvement"]}}]
        ]
        self.matcher.add("RESPONSE_PATTERNS", response_patterns)

    def analyze_report(self, text, file_path):
        """Analyze medical report using NLP techniques"""
        doc = nlp(text)
        
        # Extract age using three NLP techniques
        age = self._extract_age(doc)
        age_group = self._classify_age_group(age)
        
        # Extract response using three NLP techniques
        response = self._extract_response(doc)
        decision = "Accepted" if response > 80 else "Rejected"
        
        return {
            'file': os.path.basename(file_path),
            'age': age,
            'age_group': age_group,
            'response_percentage': response,
            'decision': decision
        }

    def _extract_age(self, doc):
        """Extract age using tokenization, NER and pattern matching"""
        # Technique 1: Pattern matching
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            if nlp.vocab.strings[match_id] == "AGE_PATTERNS":
                span = doc[start:end]
                if "dob" in span.text.lower():
                    birth_year = int(re.search(r"\d{4}", span.text).group())
                    return datetime.now().year - birth_year
                for token in span:
                    if token.like_num and 1 <= int(token.text) <= 120:
                        return int(token.text)
        
        # Technique 2: NER
        for ent in doc.ents:
            if ent.label_ == "AGE" or (ent.label_ == "CARDINAL" and 1 <= int(ent.text) <= 120):
                return int(ent.text)
        
        # Technique 3: Token analysis
        for token in doc:
            if token.like_num and 1 <= int(token.text) <= 120:
                # Check context window for age-related terms
                window_start = max(0, token.i - 3)
                window_end = min(token.i + 3, len(doc))
                context = doc[window_start:window_end].text.lower()
                if any(term in context for term in ["age", "year", "old", "yrs"]):
                    return int(token.text)
        
        return None

    def _extract_response(self, doc):
        """Extract response percentage using NLP"""
        # Technique 1: Pattern matching
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            if nlp.vocab.strings[match_id] == "RESPONSE_PATTERNS":
                span = doc[start:end]
                for token in span:
                    if token.like_num:
                        return float(token.text)
        
        # Technique 2: NER for percentages
        for ent in doc.ents:
            if "%" in ent.text:
                try:
                    return float(ent.text.replace("%", ""))
                except ValueError:
                    continue
        
        # Technique 3: Token-based sentiment analysis
        positive_terms = ["improved", "effective", "success", "positive", "reduced"]
        negative_terms = ["worsened", "ineffective", "failure", "negative", "increased"]
        
        pos_score = sum(1 for token in doc if token.text.lower() in positive_terms)
        neg_score = sum(1 for token in doc if token.text.lower() in negative_terms)
        
        if pos_score > neg_score:
            return 85.0
        elif neg_score > pos_score:
            return 35.0
        return 50.0

    def _classify_age_group(self, age):
        """Classify age into groups"""
        if age is None:
            return "Unknown"
        if age < 20:
            return "Below 20"
        elif 20 <= age <= 60:
            return "20 to 60"
        else:
            return "Above 60"

def extract_text_from_file(file_path):
    """Extract text from file"""
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
        print(f"Error reading {file_path}: {str(e)}")
    return text

def process_reports_nlp(folder_path):
    """Process all reports using NLP"""
    processor = MedicalReportProcessor()
    data = []
    
    for file in [f for f in os.listdir(folder_path) if f.lower().endswith(('.pdf', '.txt'))]:
        file_path = os.path.join(folder_path, file)
        text = extract_text_from_file(file_path)
        if text:
            analysis = processor.analyze_report(text, file_path)
            data.append(analysis)
    
    return pd.DataFrame(data)

def calculate_statistics(df):
    """Calculate statistics (same as original)"""
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
            results['age_groups'][age_group]['final_decision'] = "Accepted" if avg_response > 80 else "Rejected"
        
        if total > 0:
            results['age_groups'][age_group]['acceptance_rate'] = round((accepted / total) * 100, 2)
    
    if response_counts['overall'] > 0:
        overall_avg = response_sums['overall'] / response_counts['overall']
        results['overall']['avg_response'] = round(overall_avg, 2)
        results['overall']['final_decision'] = "Accepted" if overall_avg > 85 else "Rejected"
    
    if results['overall']['total'] > 0:
        results['overall']['acceptance_rate'] = round(
            (results['overall']['accepted'] / results['overall']['total']) * 100, 2
        )
    
    return results

def print_statistics(results):
    """Print statistics (same as original)"""
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
            print(f"  Final decision: {stats.get('final_decision', 'N/A')}")
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
    """Main function (same as original)"""
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