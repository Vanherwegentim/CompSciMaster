def extract_text_between_markers(input_file, output_file):
    with open(input_file, 'r') as file:
        text = file.read()

    start_marker = '**'
    end_marker = '**'

    start_index = text.find(start_marker)
    end_index = text.find(end_marker, start_index + len(start_marker))

    extracted_text = text[start_index + len(start_marker):end_index]

    with open(output_file, 'w') as file:
        file.write(extracted_text)

    print(f"Text between '{start_marker}' and '{end_marker}' has been extracted to '{output_file}'.")


input_file_path = 'Exam_theory_questions.md'  # Replace with the path to your input file
output_file_path = 'Exam_theory_questions.md'  # Replace with the desired output file path

extract_text_between_markers(input_file_path, output_file_path)