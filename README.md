unction to analyze resume content
def analyze_resume(text, file_path):
    doc = nlp(text)
    found_skills = [token.text for token in doc if token.text in stored_skills]

    if found_skills:  # If any matching skill is found
        os.makedirs(selected_path, exist_ok=True)

        selected_file_path = os.path.join(selected_path, os.path.basename(file_path))

        # Move the original file, handle if file exists
        try:
            shutil.move(file_path, selected_file_path)
            print(f"Resume stored in: {selected_file_path}")
        except Exception as e:
            print(f"Error moving file: {e}")

    return {
        "word_count": len(text.split()),
        "found_skills": list(set(found_skills)),
        "suggestions": "Consider adding more technical skills or certifications."
    }