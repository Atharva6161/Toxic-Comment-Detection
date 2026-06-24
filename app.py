import streamlit as st
import pickle
import re
import pandas as pd
import plotly.express as px
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Toxic Comment Detection",
    layout="centered"
)

# --- GLOBAL STYLING (Sidebar and Main Content) ---
st.markdown(
    """
    <style>
    /* --- GENERAL FONT SIZE --- */
    html, body, [class*="st-"], .main {
        font-size: 18px; /* Change this value to make the font bigger or smaller */
    }

    /* --- SIDEBAR STYLING --- */
    [data-testid="stSidebar"] {
        background-color: #B22222; /* A nice firebrick red */
    }
    .sidebar-title {
        font-size: 35px;
        font-weight: bold;
        color: white;
    }
    .sidebar-subtitle {
        font-size: 22px;
        color: white;
        font-style: italic;
    }
    /* --- SEVERITY METER STYLE (for emphasis) --- */
    .severity-high {
        font-size: 28px;
        color: #D32F2F; /* Severe Toxic Red */
        font-weight: bold;
    }
    .severity-moderate {
        font-size: 28px;
        color: orange;
        font-weight: bold;
    }
    .severity-very-low { /* Adjusted class name for consistency */
        font-size: 28px;
        color: #3498DB; /* Blue for less concern */
        font-weight: bold;
    }
    .severity-none {
        font-size: 28px;
        color: #28A745; /* Green for success/none */
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Helper Functions ---
@st.cache_resource
def load_pickle(filepath):
    """Load a pickled object from a file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def clean_text(text):
    """Text cleaning function (must be the same as used in training)."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.strip()
    return text

# --- NEW FUNCTION 1: Toxic Word Contribution Analysis ---
def analyze_toxic_contributing_words(comment, model, tfidf_vectorizer, label_cols, top_n=5):
    """
    Analyzes a comment to identify words contributing most ONLY to the 'toxic' prediction.
    """
    cleaned_comment = clean_text(comment)
    if 'toxic' not in label_cols:
        return None

    comment_tfidf = tfidf_vectorizer.transform([cleaned_comment])
    feature_names = tfidf_vectorizer.get_feature_names_out()

    try:
        toxic_label_index = label_cols.index('toxic')
    except ValueError:
        return None

    toxic_coef = model.estimators_[toxic_label_index].coef_[0]
    comment_scores = comment_tfidf.toarray()[0]
    word_contributions = comment_scores * toxic_coef

    word_contributions_series = pd.Series(word_contributions, index=feature_names)

    # Filter for words present in the comment (non-zero TF-IDF score) and positive contribution
    # We only look for words that increase the toxic score
    relevant_contributions = word_contributions_series[(comment_scores > 0) & (word_contributions > 0)]
    
    top_words = relevant_contributions.nlargest(top_n)
    
    top_words = top_words[~top_words.index.str.fullmatch(r'\d+')]
    
    return top_words

# --- NEW FUNCTION 2: Toxic Severity Classification ---
def classify_toxic_severity_only(predictions_df):
    """
    Classifies the toxicity severity of comments based ONLY on the 'toxic' prediction probability.
    """
    if 'toxic' not in predictions_df.columns:
        predictions_df['toxic_severity'] = 'Error: Missing Toxic Score'
        return predictions_df
        
    predictions_df['toxic_score'] = predictions_df['toxic']

    # Define severity thresholds for the 'toxic' score 
    low_threshold = 0.1 # 10%
    moderate_threshold = 0.5 # 50%

    def assign_toxic_severity(score):
        if score >= moderate_threshold:
            return 'High'
        elif score >= low_threshold:
            return 'Moderate'
        elif score > 0.005: # Use a very low threshold (e.g., 0.5%) to distinguish from None
             return 'Very Low'
        else:
            return 'None'

    predictions_df['toxic_severity'] = predictions_df['toxic_score'].apply(assign_toxic_severity)
    predictions_df = predictions_df.drop(columns=['toxic_score'])

    return predictions_df

# --- Load Model and Vectorizer ---
# Define labels here as they are needed for functions
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
model_loaded = False
try:
    # Load the uploaded files
    model = load_pickle('model.pkl')
    vectorizer = load_pickle('vectorizer.pkl')
    model_loaded = True
except FileNotFoundError:
    st.error("Model or vectorizer files not found. Ensure 'model.pkl' and 'vectorizer.pkl' are in the same directory.")


# --- UI Sidebar ---
st.sidebar.markdown('<p class="sidebar-title">TOXIC COMMENT DETECTION</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="sidebar-subtitle">"Here we understand and detect toxic comments and their severity."</p>', unsafe_allow_html=True)


# --- Main Page UI ---
st.header("Analyze a Comment for Toxicity")

# Initialize session state for the text area if it doesn't exist
if 'comment_input' not in st.session_state:
    st.session_state.comment_input = ""

# User input text area
user_input = st.text_area(
    "Enter a comment to analyze:",
    value=st.session_state.comment_input,
    height=150,
    key="comment_input_area"
)

# --- Buttons ---
col1, col2 = st.columns([1, 5])

with col1:
    analyze_button = st.button("Analyze", type="primary")

with col2:
    if st.button("Clear"):
        st.session_state.comment_input = ""
        st.rerun()

# --- Analysis and Display Logic ---
if analyze_button and model_loaded:
    if user_input:
        # 1. Clean and vectorize the input
        cleaned_input = clean_text(user_input)
        input_vector = vectorizer.transform([cleaned_input])

        # 2. Predict probabilities
        pred_probs = model.predict_proba(input_vector)[0]

        # 3. Define colors
        colors = {
            'threat': 'red',
            'severe_toxic': '#D32F2F',
            'toxic': 'orange',
            'obscene': '#8E44AD',
            'insult': '#3498DB',
            'identity_hate': '#A0522D'
        }

        # Create a DataFrame for results
        results_df = pd.DataFrame({
            'Toxicity Type': labels,
            'Probability': pred_probs
        })
        
        # Prepare DataFrame for severity classification
        results_df_severity = results_df.rename(columns={'Toxicity Type': 'label', 'Probability': 'toxic'})
        
        # --- NEW: Classify Toxic Severity ---
        results_df_severity = classify_toxic_severity_only(results_df_severity)
        toxic_severity = results_df_severity[results_df_severity['label'] == 'toxic']['toxic_severity'].iloc[0]

        # Determine the HTML class for styling the severity result
        severity_class = f"severity-{toxic_severity.lower().replace(' ', '-')}"

        st.markdown("---")
        st.subheader("Analysis Results")
        
        # --- Display Part 1a: Toxic Severity Meter (Explicit Non-Toxic Declaration) ---
        st.markdown(f"""
            **Overall Toxic Severity:** <span class="{severity_class}">{toxic_severity}</span> 
        """, unsafe_allow_html=True)
        st.write("") # Add a small spacer


        # --- Display Part 1b: Summary (Clearer Declaration) ---
        threshold = 0.5
        toxic_categories = results_df[results_df['Probability'] > threshold]
        
        if not toxic_categories.empty:
            summary = "The comment is classified as toxic because it crosses the 50% probability threshold for: **"
            summary += "**, **".join(toxic_categories['Toxicity Type'].tolist()) + "**."
            st.error(summary)
        else:
            # THIS IS THE KEY CHANGE for a non-toxic comment
            if toxic_severity in ['None', 'Very Low']:
                st.success(f"This comment is generally considered **Non-Toxic** (Toxic Severity: {toxic_severity}). No category exceeds the 50% probability threshold.")
            else:
                st.info("The comment does not cross the 50% probability threshold for any single category, but is classified as Moderately Toxic.")
        
        # --- Display Part 2: Table ---
        st.subheader("Probability Table")
        results_df_display = results_df.copy()
        results_df_display['Probability'] = results_df_display['Probability'].apply(lambda x: f"{x:.2%}")
        st.table(results_df_display)

        # --- Display Part 3: Visualization (Bar Chart) ---
        st.subheader("Probability Visualization")
        
        fig = px.bar(
            results_df,
            x='Toxicity Type',
            y='Probability',
            color='Toxicity Type',
            color_discrete_map=colors,
            title='Toxicity Probability Scores',
            labels={'Probability': 'Probability Score', 'Toxicity Type': 'Category'}
        )
        fig.update_layout(yaxis_title='Probability', xaxis_title='Toxicity Category', showlegend=False)
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        # --- NEW BLOCK: Word Contribution Analysis ---
        st.subheader("Word Contributions to 'Toxic' Classification")
        
        if toxic_severity in ['None', 'Very Low']:
             st.info(f"The comment is classified as having '{toxic_severity}' toxic severity. Word analysis is not necessary.")
        else:
            top_words = analyze_toxic_contributing_words(
                user_input,
                model,
                vectorizer,
                labels,
                top_n=5
            )

            if top_words is not None and not top_words.empty:
                st.info("The following words contribute most positively to the comment being classified as **'toxic'**: (Features listed have the highest positive weights.)")
                
                # Display words
                word_list = [f"**{word.capitalize()}** (Score: {score:.4f})" for word, score in top_words.items()]
                st.markdown(" - " + "\n - ".join(word_list))
            else:
                st.info("No significant words found to positively contribute to the 'toxic' classification based on the model's weights.")


    else:
        st.warning("Please enter a comment to analyze.")