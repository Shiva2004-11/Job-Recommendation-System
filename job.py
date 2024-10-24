import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from transformers import pipeline
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import PyPDF2  # For parsing PDFs
import webbrowser  # For LinkedIn integration
import plotly.express as px  # For interactive charts

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Semantic Analysis with BERT
nlp_semantic = pipeline('feature-extraction')

# Download necessary NLTK data
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# Load sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis')

# Connect to SQLite database
conn = sqlite3.connect('user_profiles.db')
c = conn.cursor()

# Create table for user profiles and job applications
c.execute('''
    CREATE TABLE IF NOT EXISTS profiles (
        username TEXT PRIMARY KEY,
        password TEXT
    )
''')
c.execute('''
    CREATE TABLE IF NOT EXISTS job_applications (
        username TEXT,
        job_id INTEGER,
        job_title TEXT,
        company_name TEXT,
        application_status TEXT
    )
''')
conn.commit()

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("C:/NLP PROJ CAT-2/linkedin_job_data.csv")

data = load_data()

# Fill missing values in 'job_details' column with an empty string
data['job_details'] = data['job_details'].fillna('')

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Job Classification
def classify_jobs(user_input, data):
    # TF-IDF to classify jobs
    tfidf_matrix = vectorizer.fit_transform(data['job_details'])
    user_input_tfidf = vectorizer.transform([user_input])
    
    # Apply cosine similarity to find similar jobs
    similarities = cosine_similarity(user_input_tfidf, tfidf_matrix)
    
    data['similarity'] = similarities[0]
    recommended_jobs = data.sort_values(by='similarity', ascending=False).head(5)
    
    return recommended_jobs[['job', 'location', 'company_name']]

# Syntax Analysis
def perform_syntax_analysis(query):
    doc = nlp(query)
    analysis = [f'{token.text}: {token.pos_}' for token in doc]
    return analysis

# Semantic Analysis
def perform_semantic_analysis(query):
    result = nlp_semantic(query)
    return result[0]

# Tagging Concepts
def perform_tagging(query):
    doc = nlp(query)
    pos_tags = [(token.text, token.pos_) for token in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return pos_tags, entities

# Chunking Information
def chunk_information(query):
    try:
        tokens = word_tokenize(query)
        pos_tags = pos_tag(tokens)
        chunks = ne_chunk(pos_tags)
        return chunks
    except LookupError as e:
        return str(e)

# Sentiment Analysis function
def analyze_sentiment(user_input):
    result = sentiment_analyzer(user_input)[0]
    sentiment_label = result['label']
    sentiment_score = result['score']
    
    # Return the sentiment label (POSITIVE/NEGATIVE) and score
    return sentiment_label, sentiment_score

# Filter Jobs based on Sentiment
def filter_jobs_by_sentiment(sentiment_label, user_input, recommended_jobs):
    if sentiment_label == 'NEGATIVE':
        doc = nlp(user_input.lower())
        # Exclude jobs related to negative sentiment words
        negative_keywords = [token.text for token in doc if token.dep_ == 'neg' or token.text == 'no' or token.text == 'not']
        
        for keyword in negative_keywords:
            recommended_jobs = recommended_jobs[~recommended_jobs['job'].str.contains(keyword, case=False)]
    
    return recommended_jobs

# User Authentication
def sign_up(username, password):
    try:
        c.execute('INSERT INTO profiles (username, password) VALUES (?, ?)', 
                  (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login(username, password):
    c.execute('SELECT * FROM profiles WHERE username = ? AND password = ?', (username, password))
    return c.fetchone() is not None

# Resume Parsing Feature
def parse_resume(file):
    # If the resume is in PDF format
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)  # Updated PdfFileReader to PdfReader
        resume_text = ""
        for page in range(len(reader.pages)):
            resume_text += reader.pages[page].extract_text()
    else:
        resume_text = file.read().decode('utf-8')
    
    doc = nlp(resume_text)
    skills = [ent.text for ent in doc.ents if ent.label_ == 'SKILL']
    experiences = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
    
    return skills, experiences

# Application Tracking Feature
def track_applications(username):
    c.execute('SELECT job_title, company_name, application_status FROM job_applications WHERE username = ?', (username,))
    applications = c.fetchall()
    return applications

# Submit a job application
def submit_application(username, job_id, job_title, company_name):
    c.execute('INSERT INTO job_applications (username, job_id, job_title, company_name, application_status) VALUES (?, ?, ?, ?, ?)',
              (username, job_id, job_title, company_name, 'Submitted'))
    conn.commit()

# LinkedIn Integration (Redirecting to LinkedIn Job Application)
def apply_via_linkedin(job_title):
    search_url = f"https://www.linkedin.com/jobs/search/?keywords={job_title.replace(' ', '%20')}"
    webbrowser.open(search_url)
    st.write(f"Opening LinkedIn for job: {job_title}")

# Streamlit UI
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Create an Account", "Login", "Home", "Job Recommendation System", "Track Applications", "Job Trends", "Feedback"])

if page == "Create an Account":
    st.title("Create an Account")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Sign Up"):
        if sign_up(username, password):
            st.success("Account created successfully!")
        else:
            st.error("Username already exists.")
    
elif page == "Login":
    st.title("Login")
    
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    
    if st.button("Login"):
        if login(username, password):
            st.success("Logged in successfully!")
            st.session_state['authenticated'] = True
            st.session_state['username'] = username
        else:
            st.error("Invalid username or password.")

elif page == "Home":
    if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
        st.error("Please log in to access the home page.")
    else:
        st.title("Welcome to the Job Recommendation System")
        
        # Provide details about the system
        st.write("""
        This Job Recommendation System helps you find the most relevant jobs based on your skills and experience.
        By analyzing job descriptions and comparing them with your input or resume, it recommends the best matching jobs.
        You can also track your job applications and apply for jobs directly via LinkedIn.
        """)
        
        # Add an image to represent the job recommendation system (you can use any suitable image path)
        st.image("C:/NLP PROJ CAT-2/jobimages.jpg", caption="Job Recommendation System", use_column_width=True)
    
elif page == "Job Recommendation System":
    if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
        st.error("Please log in to access the Job Recommendation System.")
    else:
        st.title("Job Recommendation System Based on Skills")

        # Resume Parsing Section
        st.subheader("Upload Your Resume (Optional)")
        resume_file = st.file_uploader("Upload a resume (PDF or Text)", type=['pdf', 'txt'])
        
        if resume_file:
            skills, experiences = parse_resume(resume_file)
            st.write("Extracted Skills:", skills)
            st.write("Job Experiences:", experiences)

        # User Input
        user_input = st.text_input("Enter your job-related query:")

        if user_input:
            # Analyze sentiment of the user's query
            sentiment_label, sentiment_score = analyze_sentiment(user_input)
            
            # Display sentiment analysis result
            st.subheader("Sentiment Analysis")
            st.write(f"Sentiment: {sentiment_label} (Confidence: {sentiment_score:.2f})")
            
            # Job Classification
            st.subheader("Recommended Jobs")
            recommended_jobs = classify_jobs(user_input, data)
            
            # Filter jobs based on sentiment analysis
            filtered_jobs = filter_jobs_by_sentiment(sentiment_label, user_input, recommended_jobs)
            st.write(filtered_jobs)
            
            # Syntax Analysis
            st.subheader("Syntax Analysis")
            syntax_analysis = perform_syntax_analysis(user_input)
            st.write(syntax_analysis)
            
            # Semantic Analysis
            st.subheader("Semantic Analysis")
            semantic_analysis = perform_semantic_analysis(user_input)
            st.write(semantic_analysis)
            
            # Tagging
            st.subheader("POS and Entity Tagging")
            pos_tags, entities = perform_tagging(user_input)
            st.write("POS Tags:", pos_tags)
            st.write("Named Entities:", entities)
            
            # Chunking
            st.subheader("Information Chunking")
            chunks = chunk_information(user_input)
            st.write(chunks)

            # Apply for a job
            st.subheader("Apply for a Job")
            selected_job = st.selectbox("Select a job to apply for", filtered_jobs['job'])
            if st.button("Apply"):
                job_info = filtered_jobs[filtered_jobs['job'] == selected_job].iloc[0]
                submit_application(st.session_state['username'], job_info.name, job_info['job'], job_info['company_name'])
                st.success("Application submitted!")
                
        # Apply for a job via LinkedIn
        st.subheader("Apply via LinkedIn")
        if st.button("Apply on LinkedIn"):
            apply_via_linkedin(user_input)
    
elif page == "Track Applications":
    if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
        st.error("Please log in to access your job applications.")
    else:
        st.title("Track Your Job Applications")
        applications = track_applications(st.session_state['username'])
        if applications:
            st.write(pd.DataFrame(applications, columns=['Job Title', 'Company Name', 'Application Status']))
        else:
            st.write("No applications found.")
    
elif page == "Job Trends":
    st.title("Job Trends")

    # Plot job trend visualization
    st.subheader("Job Openings Over Time")
    job_trend_data = data['posted_day_ago'].value_counts().reset_index()
    job_trend_data.columns = ['Days Ago', 'Job Count']

    fig = px.bar(job_trend_data, x='Days Ago', y='Job Count', title="Job Openings Over Time")
    st.plotly_chart(fig)

elif page == "Feedback":
    st.title("Feedback")

    # Collect user feedback
    st.write("We would love to hear your feedback on the Job Recommendation System.")
    feedback = st.text_area("Provide your feedback here")

    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")
