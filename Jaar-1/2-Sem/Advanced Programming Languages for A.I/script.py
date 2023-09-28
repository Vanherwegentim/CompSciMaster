def extract_text_between_asterisks(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    start_marker = '**'
    end_marker = '**'

    extracted_texts = []
    start_index = content.find(start_marker)
    while start_index != -1:
        end_index = content.find(end_marker, start_index + len(start_marker))
        if end_index != -1:
            extracted_text = content[start_index + len(start_marker):end_index].strip()
            extracted_texts.append(extracted_text)
            start_index = content.find(start_marker, end_index + len(end_marker))
        else:
            break

    extracted_content = start_marker + ''.join(extracted_texts) + end_marker

    with open(file_path, 'w') as file:
        file.write(extracted_content)

    print(f"Extracted text between '**' in the file '{file_path}'.")

# Usage example
file_path = 'Exam_theory_questions.md'  # Replace with your file path
extract_text_between_asterisks(file_path)
