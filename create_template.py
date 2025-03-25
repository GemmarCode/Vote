import pandas as pd
import os

# Define the columns for the template
columns = [
    'student_id',
    'student_name',
    'sex',
    'year_level',
    'course',
    'college'
]

# Create sample data
sample_data = {
    'student_id': ['2020-0001', '2020-0002'],
    'student_name': ['Juan Dela Cruz', 'Maria Santos'],
    'sex': ['M', 'F'],
    'year_level': [1, 2],
    'course': ['BS Computer Science', 'BS Information Technology'],
    'college': ['CIT', 'CIT']
}

# Create DataFrame
df = pd.DataFrame(sample_data)

# Create Excel file with two sheets
with pd.ExcelWriter('user_import_template.xlsx', engine='openpyxl') as writer:
    # Write sample data to first sheet
    df.to_excel(writer, sheet_name='Users', index=False)
    
    # Create instructions sheet
    instructions_df = pd.DataFrame({
        'Instructions': [
            'Instructions for filling out the template:',
            '',
            '1. student_id: The unique identifier for each student (e.g., 2020-0001)',
            '2. student_name: The full name of the student',
            '3. sex: Must be either "M" for Male or "F" for Female',
            '4. year_level: Must be a number between 1 and 5',
            '5. course: The full name of the student\'s course',
            '6. college: Must be one of the following codes:',
            '   - CAS: College of Arts and Sciences',
            '   - CAF: College of Agriculture and Forestry',
            '   - CCJE: College of Criminal Justice Education',
            '   - CBA: College of Business Administration',
            '   - CTED: College of Teacher Education',
            '   - CIT: College of Industrial Technology',
            '',
            'Notes:',
            '- All fields are required',
            '- Student photos should be named as student_id.jpg or student_id.png',
            '- Photos should be clear, front-facing images suitable for face recognition'
        ]
    })
    instructions_df.to_excel(writer, sheet_name='Instructions', index=False)

print("Template file 'user_import_template.xlsx' has been created successfully.") 