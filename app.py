import streamlit as st

# IMPORTANT: st.set_page_config() must be the FIRST Streamlit command
st.set_page_config(
    page_title="AI Resume Evaluator", 
    page_icon="💼", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import other libraries
import google.generativeai as genai
import os
import re
import PyPDF2 as pdf
from docx import Document
from dotenv import load_dotenv
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure API silently
if not api_key:
    st.error("⚠️ Configuration error. Please check your setup.")
    st.stop()
else:
    genai.configure(api_key=api_key)

# Rate limiting setup - CRITICAL FOR AVOIDING QUOTA ERRORS
if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = datetime.min
if "request_count" not in st.session_state:
    st.session_state.request_count = 0
if "daily_reset" not in st.session_state:
    st.session_state.daily_reset = datetime.now().date()

# Reset daily counter
if st.session_state.daily_reset != datetime.now().date():
    st.session_state.request_count = 0
    st.session_state.daily_reset = datetime.now().date()

# Rate limiting constants
MIN_REQUEST_INTERVAL = 12  # seconds between requests
MAX_REQUESTS_PER_DAY = 300  # conservative limit for free tier
MAX_REQUESTS_PER_MINUTE = 5

def check_rate_limits():
    """Check if request is within rate limits"""
    now = datetime.now()
    
    # Check daily limit
    if st.session_state.request_count >= MAX_REQUESTS_PER_DAY:
        return False, f"Daily limit reached ({MAX_REQUESTS_PER_DAY} requests). Try again tomorrow."
    
    # Check time between requests
    time_since_last = (now - st.session_state.last_request_time).total_seconds()
    if time_since_last < MIN_REQUEST_INTERVAL:
        wait_time = MIN_REQUEST_INTERVAL - time_since_last
        return False, f"Please wait {int(wait_time)} more seconds before next analysis."
    
    return True, "OK"

# Custom CSS Styling
st.markdown("""
<style>
/* Global styling */
body, .stApp {
    font-family: 'Arial', sans-serif;
    color: #FFFFFF;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    margin: 0;
}

.main-header {
    text-align: center;
    background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3em;
    font-weight: bold;
    margin-bottom: 20px;
}

.rate-limit-info {
    background: rgba(255, 193, 7, 0.2);
    border: 1px solid rgba(255, 193, 7, 0.5);
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    color: #856404;
}

.metric-container {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.strength-item {
    background: linear-gradient(135deg, #4CAF50, #45a049);
    color: white;
    padding: 10px 15px;
    margin: 5px 0;
    border-radius: 8px;
    border-left: 4px solid #2E7D32;
}

.weakness-item {
    background: linear-gradient(135deg, #f44336, #d32f2f);
    color: white;
    padding: 10px 15px;
    margin: 5px 0;
    border-radius: 8px;
    border-left: 4px solid #c62828;
}

.improvement-item {
    background: linear-gradient(135deg, #ff9800, #f57c00);
    color: white;
    padding: 10px 15px;
    margin: 5px 0;
    border-radius: 8px;
    border-left: 4px solid #ef6c00;
}

.stButton > button {
    width: 100%;
    height: 60px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}

.stButton > button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
}

.section-header {
    font-size: 1.5em;
    font-weight: bold;
    margin: 20px 0 10px 0;
    padding: 10px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 8px;
    color: white;
}

.countdown-timer {
    background: rgba(255, 193, 7, 0.3);
    border: 2px solid #ffc107;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    margin: 10px 0;
    font-weight: bold;
    color: #856404;
}
</style>
""", unsafe_allow_html=True)

def get_current_date():
    return datetime.now().strftime('%B %d, %Y')

def input_pdf_text(uploaded_file):
    try:
        reader = pdf.PdfReader(uploaded_file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error reading PDF file: {str(e)}")

def input_docx_text(uploaded_file):
    try:
        doc = Document(uploaded_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error reading DOCX file: {str(e)}")
    
def input_txt_text(uploaded_file):
    try:
        return uploaded_file.read().decode("utf-8").strip()
    except Exception as e:
        raise ValueError(f"Error reading TXT file: {str(e)}")

def get_gemini_response(input_text, max_retries=3):
    """Enhanced Gemini API call with proper error handling and rate limiting"""
    
    # Check rate limits first
    can_proceed, message = check_rate_limits()
    if not can_proceed:
        st.error(f"⚠️ Rate limit: {message}")
        return None
    
    for attempt in range(max_retries):
        try:
            # Use gemini-1.5-flash instead of gemini-1.5-pro for better rate limits
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Add small delay before request
            time.sleep(1)
            
            response = model.generate_content(input_text)
            
            # Update rate limiting counters on successful request
            st.session_state.last_request_time = datetime.now()
            st.session_state.request_count += 1
            
            return response.text
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if "quota" in error_msg or "exceeded" in error_msg:
                st.error("⚠️ API quota exceeded. Please try again later.")
                st.info("""
                💡 **Solutions:**
                - Wait a few minutes and try again
                - Check your Google AI Studio usage dashboard
                - Free tier has daily limits - try again tomorrow if needed
                - Consider upgrading for higher limits
                """)
                return None
                
            elif "rate limit" in error_msg or "429" in error_msg:
                wait_time = min(30 * (2 ** attempt), 300)  # Exponential backoff, max 5 minutes
                st.warning(f"⏳ Rate limit hit. Waiting {wait_time} seconds before retry...")
                
                # Show countdown
                countdown_placeholder = st.empty()
                for i in range(wait_time, 0, -1):
                    countdown_placeholder.markdown(f"""
                    <div class="countdown-timer">
                        ⏳ Waiting for rate limit... {i} seconds remaining
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(1)
                countdown_placeholder.empty()
                
                if attempt < max_retries - 1:
                    continue
                else:
                    st.error("❌ Maximum retries reached. Please try again later.")
                    return None
                    
            elif attempt < max_retries - 1:
                st.warning(f"⚠️ Request failed (attempt {attempt + 1}/{max_retries}). Retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                st.error(f"❌ API request failed: {str(e)}")
                return None
    
    return None

# Enhanced prompts for comprehensive analysis (optimized for fewer tokens)
comprehensive_analysis_prompt = """
You are an expert ATS and HR professional. Analyze this resume against the job description.

Resume: {resume_text}
Job Description: {job_description}

Provide analysis in JSON format with realistic scores (0-100):
{{
    "overall_match_percentage": 75,
    "category_scores": {{
        "skills_match": 70,
        "experience_relevance": 80,
        "education_alignment": 75,
        "keyword_optimization": 65,
        "format_quality": 85
    }},
    "strengths": [
        "Strong relevant experience",
        "Good technical skills"
    ],
    "weaknesses": [
        "Missing key certifications",
        "Limited tool experience"
    ],
    "areas_for_improvement": [
        "Add quantifiable achievements",
        "Include more keywords"
    ],
    "missing_keywords": ["Python", "AWS", "Docker"],
    "recommendations": [
        "Add metrics to achievements",
        "Optimize for ATS"
    ],
    "ats_compatibility_score": 72,
    "profile_summary": "Strong candidate with relevant background. Needs keyword optimization."
}}

Be concise and actionable.
"""

def clean_and_parse_json(response):
    """Clean and parse JSON response from Gemini."""
    if not response:
        return None
    
    try:
        # Remove markdown formatting
        cleaned = response.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        
        # Remove control characters
        cleaned = re.sub(r'[\x00-\x1F\x7F]', '', cleaned)
        
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None

def create_score_visualization(scores):
    """Create enhanced score visualization."""
    categories = list(scores.keys())
    values = list(scores.values())
    
    # Radar Chart
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Your Score',
        line_color='rgb(102, 126, 234)',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Skills Assessment Radar Chart",
        font=dict(size=14),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig_radar

def create_overall_score_gauge(score):
    """Create a gauge chart for overall score."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Overall ATS Score"},
        delta = {'reference': 70},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    
    return fig

def create_comparison_chart(category_scores):
    """Create bar chart for category comparison."""
    df = pd.DataFrame(list(category_scores.items()), columns=['Category', 'Score'])
    
    fig = px.bar(df, x='Category', y='Score', 
                 title="Category-wise Performance",
                 color='Score',
                 color_continuous_scale=['red', 'yellow', 'green'])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

# Sidebar with enhanced information
st.sidebar.markdown("""
### 📊 Usage Status
""")

# Display current usage stats
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Requests Today", st.session_state.request_count)
with col2:
    remaining = MAX_REQUESTS_PER_DAY - st.session_state.request_count
    st.metric("Remaining", remaining)

# Time since last request
if st.session_state.last_request_time != datetime.min:
    time_since_last = (datetime.now() - st.session_state.last_request_time).total_seconds()
    st.sidebar.info(f"⏰ Last request: {int(time_since_last)}s ago")

st.sidebar.markdown("""
### ⚠️ Rate Limit Info
- **12 seconds** between requests
- **300 requests** per day max
- **5 requests** per minute max

### 💡 Tips for Success
- Wait for green "Ready" status
- Use clear, readable resumes
- Provide complete job descriptions
- Check Google AI Studio dashboard for quota
""")

# Rate limit status indicator
time_since_last = (datetime.now() - st.session_state.last_request_time).total_seconds()
if time_since_last >= MIN_REQUEST_INTERVAL:
    st.sidebar.success("✅ Ready for next request")
else:
    wait_time = MIN_REQUEST_INTERVAL - time_since_last
    st.sidebar.warning(f"⏳ Wait {int(wait_time)}s more")

# Main App Layout
st.markdown('<h1 class="main-header">🎯 Advanced Resume ATS Evaluator</h1>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; font-size: 1.2em; margin-bottom: 30px; color: #ecf0f1;">
    Get comprehensive insights into your resume's ATS compatibility with smart rate limiting
</div>
""", unsafe_allow_html=True)

# Rate limiting warning if needed
can_proceed, rate_message = check_rate_limits()
if not can_proceed:
    st.markdown(f"""
    <div class="rate-limit-info">
        ⚠️ <strong>Rate Limit:</strong> {rate_message}
    </div>
    """, unsafe_allow_html=True)

# Input Section
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="section-header">📋 Job Description</div>', unsafe_allow_html=True)
    job_description = st.text_area(
        "", 
        placeholder="Paste the complete job description here...", 
        height=200,
        key="job_desc"
    )

with col2:
    st.markdown('<div class="section-header">📄 Upload Resume</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "", 
        type=["pdf", "docx", "txt"],
        help="Upload your resume in PDF, DOCX or TXT format"
    )

# Analysis Button with rate limiting
analyze_disabled = not can_proceed or not job_description.strip() or uploaded_file is None

if st.button("🚀 Analyze Resume", key="analyze_btn", disabled=analyze_disabled):
    if not job_description.strip():
        st.error("❌ Please provide a job description")
    elif uploaded_file is None:
        st.error("❌ Please upload your resume")
    else:
        try:
            with st.spinner("🔍 Analyzing your resume... (This may take 10-15 seconds)"):
                # Extract resume text silently
                if uploaded_file.type == "application/pdf":
                    resume_text = input_pdf_text(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    resume_text = input_docx_text(uploaded_file)
                else:
                    resume_text = input_txt_text(uploaded_file)
                
                if not resume_text.strip():
                    st.error("❌ Could not extract text from the resume. Please check your file.")
                else:
                    # Truncate text if too long to avoid token limits
                    if len(resume_text) > 8000:
                        resume_text = resume_text[:8000] + "... [truncated for API limits]"
                    
                    # Perform comprehensive analysis
                    analysis_prompt = comprehensive_analysis_prompt.format(
                        resume_text=resume_text, 
                        job_description=job_description[:4000]  # Limit job description length too
                    )
                    
                    analysis_response = get_gemini_response(analysis_prompt)
                    
                    if analysis_response:
                        analysis_data = clean_and_parse_json(analysis_response)
                        
                        if analysis_data:
                            st.success("✅ Analysis Complete!")
                            
                            # Overall Score Section
                            st.markdown('<div class="section-header">📊 Overall Performance</div>', unsafe_allow_html=True)
                            
                            col1, col2, col3 = st.columns([1, 2, 1])
                            
                            with col1:
                                overall_score = analysis_data.get('overall_match_percentage', 0)
                                st.metric(
                                    label="ATS Match Score", 
                                    value=f"{overall_score}%",
                                    delta=f"{overall_score-70}% vs benchmark"
                                )
                                
                                ats_score = analysis_data.get('ats_compatibility_score', 0)
                                st.metric(
                                    label="ATS Compatibility", 
                                    value=f"{ats_score}%"
                                )
                            
                            with col2:
                                # Gauge Chart
                                gauge_fig = create_overall_score_gauge(overall_score)
                                st.plotly_chart(gauge_fig, use_container_width=True)
                            
                            with col3:
                                # Score interpretation
                                if overall_score >= 80:
                                    st.success("🎉 Excellent Match!")
                                    st.write("Your resume is well-aligned with the job requirements.")
                                elif overall_score >= 60:
                                    st.warning("⚠️ Good Match")
                                    st.write("Some improvements needed for better alignment.")
                                else:
                                    st.error("❌ Needs Improvement")
                                    st.write("Significant gaps need to be addressed.")
                            
                            # Category Scores
                            st.markdown('<div class="section-header">📈 Detailed Category Analysis</div>', unsafe_allow_html=True)
                            
                            category_scores = analysis_data.get('category_scores', {})
                            
                            if category_scores:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    radar_fig = create_score_visualization(category_scores)
                                    st.plotly_chart(radar_fig, use_container_width=True)
                                
                                with col2:
                                    bar_fig = create_comparison_chart(category_scores)
                                    st.plotly_chart(bar_fig, use_container_width=True)
                            
                            # Strengths, Weaknesses, Improvements
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown('<div class="section-header">💪 Strengths</div>', unsafe_allow_html=True)
                                strengths = analysis_data.get('strengths', [])
                                for strength in strengths:
                                    st.markdown(f'<div class="strength-item">✅ {strength}</div>', unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown('<div class="section-header">⚠️ Areas of Concern</div>', unsafe_allow_html=True)
                                weaknesses = analysis_data.get('weaknesses', [])
                                for weakness in weaknesses:
                                    st.markdown(f'<div class="weakness-item">❌ {weakness}</div>', unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown('<div class="section-header">🎯 Improvements Needed</div>', unsafe_allow_html=True)
                                improvements = analysis_data.get('areas_for_improvement', [])
                                for improvement in improvements:
                                    st.markdown(f'<div class="improvement-item">🔧 {improvement}</div>', unsafe_allow_html=True)
                            
                            # Missing Keywords
                            st.markdown('<div class="section-header">🔑 Missing Keywords</div>', unsafe_allow_html=True)
                            missing_keywords = analysis_data.get('missing_keywords', [])
                            if missing_keywords:
                                keywords_str = ", ".join([f"`{kw}`" for kw in missing_keywords])
                                st.markdown(f"**Add these keywords to improve ATS score:** {keywords_str}")
                            else:
                                st.success("✅ All important keywords are present!")
                            
                            # Recommendations
                            st.markdown('<div class="section-header">💡 AI Recommendations</div>', unsafe_allow_html=True)
                            recommendations = analysis_data.get('recommendations', [])
                            for i, rec in enumerate(recommendations, 1):
                                st.write(f"**{i}.** {rec}")
                            
                            # Profile Summary
                            st.markdown('<div class="section-header">📝 Overall Assessment</div>', unsafe_allow_html=True)
                            profile_summary = analysis_data.get('profile_summary', '')
                            st.write(profile_summary)
                            
                        else:
                            st.error("❌ Unable to process the analysis. Please try again.")
                    else:
                        st.error("❌ Analysis failed due to rate limits or API issues.")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("""
---
<div style="text-align: center; color: #7f8c8d; padding: 20px;">
    <p>Built with ❤️ using Streamlit & Google Gemini AI</p>
    <p><strong>Rate Limited Version</strong> - Respects API quotas for reliable performance</p>
</div>
""", unsafe_allow_html=True)