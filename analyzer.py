import spacy
import re
from collections import defaultdict
import os
from datetime import datetime

# Load the English language model for spaCy
nlp = spacy.load("en_core_web_sm")

class MedicalReportAnalyzer:
    def __init__(self):
        self.age_groups = {
            '0-20': {'accepted': 0, 'rejected': 0, 'reports': []},
            '21-50': {'accepted': 0, 'rejected': 0, 'reports': []},
            '51-80': {'accepted': 0, 'rejected': 0, 'reports': []}
        }
        self.total_reports = 0
    
    def extract_age(self, text):
        """Extract age from medical report text"""
        # Look for age patterns like "age: 25", "25 years old", etc.
        age_patterns = [
            r'age[\s:]*(\d+)',
            r'(\d+)\s*years old',
            r'patient is (\d+)\s*years',
            r'aged (\d+)'
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        # If no explicit age found, try to find numbers that might represent age
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "CARDINAL" and ent.text.isdigit():
                age = int(ent.text)
                if 0 <= age <= 120:  # Reasonable age range
                    return age
        
        return None
    
    def determine_experiment_status(self, text):
        """Determine if drug experiment was accepted or rejected"""
        text_lower = text.lower()
        accept_keywords = ['accepted', 'approved', 'successful', 'positive outcome', 'recommended']
        reject_keywords = ['rejected', 'failed', 'not approved', 'adverse effects', 'discontinued']
        
        accept_count = sum(text_lower.count(keyword) for keyword in accept_keywords)
        reject_count = sum(text_lower.count(keyword) for keyword in reject_keywords)
        
        if accept_count > reject_count:
            return 'accepted'
        elif reject_count > accept_count:
            return 'rejected'
        else:
            return 'undetermined'
    
    def categorize_report(self, text, filename):
        """Categorize the report by age group and experiment status"""
        age = self.extract_age(text)
        if age is None:
            print(f"Warning: Could not determine age for {filename}")
            return
        
        status = self.determine_experiment_status(text)
        if status == 'undetermined':
            print(f"Warning: Could not determine experiment status for {filename}")
            return
        
        # Determine age group
        if 0 <= age <= 20:
            age_group = '0-20'
        elif 21 <= age <= 50:
            age_group = '21-50'
        elif 51 <= age <= 80:
            age_group = '51-80'
        else:
            print(f"Warning: Age {age} out of range for {filename}")
            return
        
        # Update statistics
        self.age_groups[age_group][status] += 1
        self.age_groups[age_group]['reports'].append({
            'filename': filename,
            'age': age,
            'status': status,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        self.total_reports += 1
    
    def calculate_acceptance_percentages(self):
        """Calculate acceptance percentages for each age group"""
        percentages = {}
        for group, data in self.age_groups.items():
            total = data['accepted'] + data['rejected']
            if total > 0:
                acceptance_pct = (data['accepted'] / total) * 100
                rejection_pct = 100 - acceptance_pct
            else:
                acceptance_pct = 0
                rejection_pct = 0
            percentages[group] = {
                'acceptance_pct': round(acceptance_pct, 2),
                'rejection_pct': round(rejection_pct, 2),
                'total_reports': total
            }
        return percentages
    
    def process_uploaded_file(self, file_path):
        """Process an uploaded medical report file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            filename = os.path.basename(file_path)
            self.categorize_report(text, filename)
            return True
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return False
    
    def generate_report(self):
        """Generate a summary report of the analysis"""
        percentages = self.calculate_acceptance_percentages()
        
        report = "Medical Report Analysis Summary\n"
        report += "=============================\n\n"
        report += f"Total Reports Processed: {self.total_reports}\n\n"
        
        for group, stats in percentages.items():
            report += f"Age Group: {group}\n"
            report += f"- Total Reports: {stats['total_reports']}\n"
            report += f"- Acceptance Percentage: {stats['acceptance_pct']}%\n"
            report += f"- Rejection Percentage: {stats['rejection_pct']}%\n"
            report += "\n"
        
        return report

# Example usage
if __name__ == "__main__":
    analyzer = MedicalReportAnalyzer()
    
    # Simulate file uploads - in a real application, you'd have a file upload interface
    sample_files = [
        "sample_reports/report1.txt",  # Should contain age and experiment status
        "sample_reports/report2.txt",
        "sample_reports/report3.txt"
    ]
    
    # Process each file
    for file_path in sample_files:
        if os.path.exists(file_path):
            analyzer.process_uploaded_file(file_path)
        else:
            print(f"File not found: {file_path}")
    
    # Generate and display the report
    print(analyzer.generate_report())