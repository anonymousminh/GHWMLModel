import pandas as pd

def load_standard_data(path):
    df = pd.read_csv('../data/Student_Performance_Dataset.csv')
    allowed_columns = [
        'Attendance (%)',
        'Midterm_Score',
        'Final_Score',
        'Projects_Score',
        'Study_Hours_per_Week'
    ]
    return df[allowed_columns]

def load_biased_data(path):
    df = pd.read_csv('../data/Student_Performance_Dataset.csv')
    biased_allowed_columns = [
        'Attendance (%)',
        'Midterm_Score',
        'Final_Score',
        'Project_Score',
        'Study_Hours_per_Week',
        'Gender',
        'Parent_Education',
        'Internet_Access_at_Home'
    ]
