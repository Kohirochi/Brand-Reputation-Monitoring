import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import pandas as pd
import emoji
import re
import string
import plotly.graph_objects as go
from numerize.numerize import numerize
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards
import os
from plotly.subplots import make_subplots
from lingua import Language, LanguageDetectorBuilder
import translators as ts

def show_file_upload_details():
    # Title for the File Upload Tab
    st.header("File Upload Tab | File-Based Prediction")

    # Purpose Section
    st.subheader("Purpose")
    st.write("Upload a CSV or Excel file for batch sentiment prediction.")

    # Instructions Section
    st.subheader("Instructions")
    st.write("1. **Upload File**: Choose a CSV or Excel file.")
    st.write("2. **Set Weightage (Optional)**: Customize weightage for 'E-commerce' and 'Social Media' if needed.")
    st.write("3. **Predict**: Click 'Predict' to analyze sentiments of all comments.")
    st.write("4. **Download Results**: Download the results as a CSV file after predictions.")

    # Functionality Section
    st.subheader("Functionality")
    st.write("1. **File Upload**: Accepts files with a 'Comment' column.")
    st.write("2. **Weightage**: Set weightage for categories if a 'Type' column is present.")
    st.write("3. **Batch Prediction**: Analyzes each comment and displays results.")
    st.write("4. **Dynamic Dashboard**: The dashboard adjusts based on the columns in the file.")
    st.write("5. **Progress & Stop**: Monitor progress and stop processing if needed.")

def show_text_tab_details():
    # Header for the Text Tab
    st.header("Text Tab | Single Comment Prediction")

    # Purpose Section
    st.subheader("Purpose")
    st.write("Predicts sentiment of a single comment.")

    # Instructions Section
    st.subheader("Instructions")
    st.write("1. **Enter Comment**: Type a comment in the text area.")
    st.write("2. **Predict**: Click 'Predict' to get the sentiment.")
    st.write("3. **Clear**: Click 'Clear Text' to reset the text area.")

    # Functionality Section
    st.subheader("Functionality")
    st.write("Displays the predicted sentiment of the comment.")

def detect_language(text, detector):
    """
    Detect the language of the input text using the provided language detector.
    """
    try:
        return detector.detect_language_of(text).iso_code_639_1.name.lower()
    except Exception as e:
        return 'unknown'

def translate_to_english(detected_language, text, max_retries=2):
    """
    Translate the input text to English if it's not already in English.
    """
    retries = 0
    while retries < max_retries:
        try:
            translated_text = ts.translate_text(text, from_language=detected_language, to_language="en", translator="bing")
            return translated_text  # Return the translated text if successful
        except Exception as e:
            retries += 1
    return text  # Return original text if all retries fail

def remove_noise(text):
    # Convert Emoji to String Representation
    text = emoji.demojize(text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove emails
    text = re.sub(r'\S+@\S+', '', text, flags=re.MULTILINE)

    # Remove usernames (assuming usernames start with @)
    text = re.sub(r'@[\w]+', '', text, flags=re.MULTILINE)

    # Remove words containing numbers
    text = re.sub(r'\w*\d\w*', '', text, flags=re.MULTILINE)

    # Remove words containing special characters
    text = re.sub(r'\w*[@#$%^&*()+={}[\]|\\<>/~`]\w*', '', text, flags=re.MULTILINE) # cannot remove word with _ because emoji has _

    # Remove punctuation
    text = re.sub(r'[_,.?!;:\'\"]', ' ', text, flags=re.MULTILINE)

    return text

def preprocess_text(text, detector):
    text = remove_noise(text)

    # Detect language
    detected_language = detect_language(text, detector)

    # Translate to English if necessary
    if detected_language != 'en':
        text = translate_to_english(detected_language, text, max_retries=2)
    return text

def preprocess_file(df):
    # Initialize the language detector
    languages = [Language.ENGLISH, Language.MALAY, Language.CHINESE, Language.KOREAN, Language.JAPANESE]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    # Preprocess the comments and create a new column for processed text
    df['Translated_Comment'] = df['Comment'].apply(lambda x: preprocess_text(x, detector))
    return df

def validate_user_input(input):
    # Check if input is None or empty
    if not input:
        return False, "Input is empty. Please enter a valid comment."
    
    # Check if input contains only punctuation or special characters
    if all(char in string.punctuation or char.isspace() for char in input):
        return False, "Input contains only punctuation or special characters. Please enter a valid comment."
    
    return True, ""

def predict_sentiment(text):
    # Preprocess text and convert to features
    text_features = vectorizer.transform([text])
    prediction = model.predict(text_features)
    sentiment_mapping = {0: "negative", 1: "neutral", 2: "positive"}
    return sentiment_mapping[int(prediction[0])]

def clear_text():
    st.session_state["input_area"] = ""

def get_state(key):
    return st.session_state[key]

def set_state(key, value):
    st.session_state[key] = value

# Initialize session state
def initialize_session_state():
    """Initialize session state with default values if not already set."""
    for key, value in default_values.items():
        if key not in st.session_state:
            set_state(key, value)

def reset_session():
    """Reset session state to default values."""
    for key, value in default_values.items():
        set_state(key, value)

def metric_card_plot():
    df = st.session_state.result
    metrics_container = st.container()
    with metrics_container:
        if not st.session_state.brand_column_present:
            col_total_comments, col_positive_comments, col_negative_comments, col_neutral_comments = st.columns(4, gap="large")
        else:
            col_total_comments, col_positive_comments, col_negative_comments, col_neutral_comments, col_unique_brands = st.columns(5, gap="large")
        
        with col_total_comments:
            total_comments = len(df) - 1 # headers
            st.metric(label="Total Comment", value=numerize(total_comments), help=f"Total Comment: {total_comments}")
        with col_positive_comments:
            total_positive = df[df['Sentiment'] == 'positive'].shape[0]
            st.metric(label="Total Positive", value=numerize(total_positive), help=f"Total Positive Comments: {total_positive}")
        with col_negative_comments:
            total_negative = df[df['Sentiment'] == 'negative'].shape[0]
            st.metric(label="Total Negative ", value=numerize(total_negative), help=f"Total Negative Comments: {total_negative}")
        with col_neutral_comments:
            total_neutral = df[df['Sentiment'] == 'neutral'].shape[0]
            st.metric(label="Total Neutral", value=numerize(total_neutral), help=f"Total Neutral Comments: {total_neutral}")
        if st.session_state.brand_column_present:
            with col_unique_brands:
                total_unique_brands = df['Brand'].nunique()
                st.metric(label="Total Brand", value=total_unique_brands)
        style_metric_cards(background_color="#000000", border_left_color="#686664", border_color="#FFFFFF", box_shadow="#F71938")

def compute_score(predicted_classes):
    df = get_state('result')
    if get_state('type_column_present') and get_state('enable_score_weightage'):
        types = df['Type']
        # Get weightages
        social_weightage = get_state('social_pct') / 100
        ecom_weightage = get_state('ecom_pct') / 100
        # Count positive and negative comments by type
        counts = {'social media': {'positive': 0, 'negative': 0, 'neutral': 0}, 'e-commerce': {'positive': 0, 'negative': 0, 'neutral': 0}}
        totals = {'social media': 0, 'e-commerce': 0}
        
        for c, t in zip(predicted_classes, types):
            if t in counts:
                counts[t][c] += 1
                totals[t] += 1
        # Calculate proportions
        def proportion(count, total):
            return count / total if total > 0 else 0
        
        positive_proportion = (
            social_weightage * proportion(counts['social media']['positive'], totals['social media']) +
            ecom_weightage * proportion(counts['e-commerce']['positive'], totals['e-commerce'])
        )
        negative_proportion = (
            social_weightage * proportion(counts['social media']['negative'], totals['social media']) +
            ecom_weightage * proportion(counts['e-commerce']['negative'], totals['e-commerce'])
        )
    else:
        total_comments = len(predicted_classes)
        positive_proportion = predicted_classes.value_counts().get("positive", 0) / total_comments if total_comments > 0 else 0
        negative_proportion = predicted_classes.value_counts().get("negative", 0) / total_comments if total_comments > 0 else 0

    # Calculate the reputation score range between -100 to 100
    reputation_score = round(100 * (positive_proportion - negative_proportion))
    
    return reputation_score

def plot_gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        gauge={
            'axis': {'range': [-100, 100], 'tickcolor': "white", 'tickfont': {'color': "white", 'size': 15}},
            'bar': {'color': "#1E90FF",
                    'thickness': 0.3},
            'steps': [
                {'range': [-100, -50], 'color': "red"},
                {'range': [-50, 0], 'color': "orange"},
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "green"}],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': score},
            'bgcolor': "black"
            },
            number={'font': {'color': "white"}},
        ))
    
    fig.update_layout(
        height=400,
        margin=dict(t=10, b=10, l=30, r=30),
        title={
            'text': "Overall Reputation Score",
            'y': 0.95,  # Adjust title position vertically
            'x': 0.5,  # Center title horizontally
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}
        }
    )
    st.plotly_chart(fig)

# Plot reputation scores as a horizontal bar chart using Plotly
def plot_brand_score_by_brand(df):
    df = df.sort_values(by='Reputation Score', ascending=True)
    fig = px.bar(df, x='Reputation Score', y='Brand', orientation='h', labels={'Reputation Score': 'Reputation Score'})
    fig.update_layout(
        height=300,
        margin=go.layout.Margin(
                l=20, #left margin
                r=20, #right margin
                b=20, #bottom margin
                t=20  #top margin
            )
    )
    fig.update_layout()
    st.plotly_chart(fig)

def create_bar_and_pie_subplots(df, chart_type, title, legend_title):
    """Create subplots with bar and pie charts."""

    # Create pie chart
    pie_chart = px.pie(df, values='Count', names=chart_type, color=chart_type)
    pie_chart.update_traces(textinfo='percent+label', hole=0.3)

    # Create bar chart
    bar_chart = px.bar(df, x=chart_type, y='Count', color=chart_type)

    # Create subplots
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "pie"}]])

    # Add bar chart to subplot
    for trace in bar_chart['data']:
        trace['offset'] = -0.4
        trace['showlegend'] = False
        fig.add_trace(trace, row=1, col=1)

    # Add pie chart to subplot
    for trace in pie_chart['data']:
        fig.add_trace(trace, row=1, col=2)

    # Update layout
    fig.update_layout(
        height=400,
        title={
            'text': title,
            'y': 0.95,
            'x': 0.45,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}
        },
        showlegend=True,
        margin=dict(t=65, b=50, l=30, r=30),
        legend=dict(title=dict(text=legend_title), x=1.1, y=1.0)
    )

    return fig

def get_reputation_scores_by_brand(df):
    """Generate brand reputation scores."""
    brand_groups = df.groupby('Brand')
    brand_reputation_scores = {
        brand: compute_score(comments_df['Sentiment'])
        for brand, comments_df in brand_groups if not comments_df.empty
    }

    brand_scores_df = pd.DataFrame(list(brand_reputation_scores.items()), columns=['Brand', 'Reputation Score'])
    # Sort by score in descending
    brand_scores_df = brand_scores_df.sort_values(by='Reputation Score', ascending=False)

    return brand_scores_df

def get_counts_by_column(df, column_name, count_column_name):
    """Count occurrences of each sentiment/type."""
    counts = df[column_name].value_counts().reset_index()
    counts.columns = [column_name, count_column_name]
    return counts

def brand_reputation_score_plot(df, brand_column_present):
    if not brand_column_present:
        # Calculate the brand reputation score
        reputation_score = compute_score(df['Sentiment'])
        
        reputation_score_gauge, distribution_chart_by_column = st.columns([1, 4], gap="small")
        
        with reputation_score_gauge:
            with st.container(border=True):
                plot_gauge_chart(reputation_score)
        
        with distribution_chart_by_column:
            with st.container(border=True):
                column_name, title = ('Type', 'Comment Type Distribution') if get_state('type_column_present') else ('Sentiment', 'Sentiment Distribution')
                sentiment_counts = get_counts_by_column(df, column_name, 'Count')
                fig = create_bar_and_pie_subplots(sentiment_counts, column_name, title, column_name)
                st.plotly_chart(fig)

    # Competitor Analysis
    if brand_column_present:
        brand_scores_df = get_reputation_scores_by_brand(df)

        with st.container(border=True):
            st.subheader("Brand Reputation Scores")
            brand_score_table, brand_score_chart  = st.columns([1, 2], gap="small")
            with brand_score_table:
                st.dataframe(brand_scores_df, hide_index=True, use_container_width=True)
            with brand_score_chart:
                plot_brand_score_by_brand(brand_scores_df)

def generate_title(type_column_present, brand_column_present, selected_comment_type, selected_brand):
    if type_column_present and brand_column_present:
        return f'Sentiment Distribution of {selected_brand} ({selected_comment_type})' if selected_brand != 'All' else f'Sentiment Distribution by Brand ({selected_comment_type})'
    elif type_column_present:
        return f'Sentiment Distribution by Brand ({selected_comment_type})'
    elif brand_column_present:
        return f'Sentiment Distribution of {selected_brand}' if selected_brand != 'All' else f'Sentiment Distribution by Brand'
    else:
        return 'Sentiment Distribution'

def get_sentiment_counts(df, type_column_present, brand_column_present, selected_comment_type, selected_brand, sentiments):
    if not brand_column_present:
        if selected_comment_type == 'All':
            sentiment_counts = df.groupby(['Type', 'Sentiment']).size().reset_index(name='Count')
            cartesian_product = pd.DataFrame([(type, sentiment) for type in df['Type'].unique() for sentiment in sentiments], columns=['Type', 'Sentiment'])
        else:
            df_filtered = df[df['Type'] == selected_comment_type]
            sentiment_counts = df_filtered.groupby(['Sentiment']).size().reset_index(name='Count')
            cartesian_product = pd.DataFrame([(sentiment,) for sentiment in sentiments], columns=['Sentiment'])
    elif not type_column_present:
        if selected_brand == 'All':
            sentiment_counts = df.groupby(['Brand', 'Sentiment']).size().reset_index(name='Count')
            cartesian_product = pd.DataFrame([(brand, sentiment) for brand in df['Brand'].unique() for sentiment in sentiments], columns=['Brand', 'Sentiment'])
        else:
            df_filtered = df[df['Brand'] == selected_brand]
            sentiment_counts = df_filtered.groupby(['Sentiment']).size().reset_index(name='Count')
            cartesian_product = pd.DataFrame([(sentiment,) for sentiment in sentiments], columns=['Sentiment'])
    else:
        if selected_comment_type == 'All' and selected_brand == 'All':
            sentiment_counts = df.groupby(['Type', 'Brand', 'Sentiment']).size().reset_index(name='Count')
            # cartesian_product = pd.DataFrame([(brand, sentiment) for brand in df['Brand'].unique() for sentiment in sentiments], columns=['Brand', 'Sentiment'])
            cartesian_product = pd.DataFrame(pd.MultiIndex.from_product([df['Type'].unique(), df['Brand'].unique(), sentiments], names=['Type', 'Brand', 'Sentiment']).to_frame(index=False))
        elif selected_comment_type == 'All':
            df_filtered = df[df['Brand'] == selected_brand]
            sentiment_counts = df_filtered.groupby(['Type', 'Sentiment']).size().reset_index(name='Count')
            cartesian_product = pd.DataFrame([(type, sentiment) for type in df['Type'].unique() for sentiment in sentiments], columns=['Type', 'Sentiment'])
        elif selected_brand == 'All':
            df_filtered = df[df['Type'] == selected_comment_type]
            sentiment_counts = df_filtered.groupby(['Brand', 'Sentiment']).size().reset_index(name='Count')
            cartesian_product = pd.DataFrame([(brand, sentiment) for brand in df['Brand'].unique() for sentiment in sentiments], columns=['Brand', 'Sentiment'])
        else:
            df_filtered = df[(df['Type'] == selected_comment_type) & (df['Brand'] == selected_brand)]
            sentiment_counts = df_filtered.groupby(['Sentiment']).size().reset_index(name='Count')
            cartesian_product = pd.DataFrame([(sentiment,) for sentiment in sentiments], columns=['Sentiment'])
    return pd.merge(cartesian_product, sentiment_counts, how='left').fillna(0)

def plot_pie_chart(df, values, names):
    fig = px.pie(df, values=values, names=names, hole=0.3)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(margin=dict(t=50, b=30, l=10, r=10), legend_title_text=names)
    st.plotly_chart(fig)

# Sentiment Distribution (SD)
def sentiment_distribution_plot():
    df = get_state('result')
    type_column_present = get_state('type_column_present')
    brand_column_present = get_state('brand_column_present')

    if not brand_column_present and not type_column_present:
        pass
    else:
        sd_container = st.container(border=True)
        with sd_container:
            st.subheader("Distribution")

            # Initialize selections
            selected_comment_type = 'All'
            selected_brand = 'All'

            selection_cols = st.columns(2)

            # Create brand selection if brand column is present
            if brand_column_present:
                with selection_cols[0]:
                    brand_options = df['Brand'].unique()
                    brand_selection = ['All'] + brand_options.tolist()
                    selected_brand = st.selectbox('Brand', brand_selection)

            # Create comment type selection if type column is present
            if type_column_present:
                with selection_cols[1 if brand_column_present else 0]:
                    type_options = df['Type'].unique()
                    comment_types_selection = ['All'] + type_options.tolist()
                    selected_comment_type = st.selectbox('Comment Type', comment_types_selection)

            # Set the title based on selections
            title = generate_title(type_column_present, brand_column_present, selected_comment_type, selected_brand)
            st.subheader("")
            st.markdown(f"<h5 style='text-align: center; color: white; margin: 0 0 0px 0;'>{title}</h5>", unsafe_allow_html=True)
            
            sentiments = ['positive', 'negative', 'neutral']      
            
            sentiment_counts = get_sentiment_counts(df, type_column_present, brand_column_present, selected_comment_type, selected_brand, sentiments)
            
            cols = st.columns(2, gap="small")
            with cols[0]:
                if brand_column_present and type_column_present and selected_brand == 'All':
                    sentiment_counts_grouped = sentiment_counts.groupby(['Brand', 'Sentiment']).agg({'Count': 'sum'}).reset_index()
                    sentiment_counts_grouped = sentiment_counts_grouped.sort_values(by=['Brand', 'Sentiment'], ascending=[True, False])
                    fig = px.bar(sentiment_counts_grouped, x='Sentiment', y='Count', color='Brand', barmode='group')
                elif brand_column_present and selected_brand == 'All':                    
                    fig = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Brand', barmode='group')
                elif type_column_present and selected_comment_type == 'All':
                    fig = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Type', barmode='group')
                else:
                    fig = px.bar(sentiment_counts, x='Sentiment', y='Count')
                
                fig.update_layout(
                    xaxis_title='Sentiment', 
                    yaxis_title='Count', 
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                st.plotly_chart(fig)
            with cols[1]:
                plot_pie_chart(sentiment_counts, 'Count', 'Sentiment')

            if brand_column_present and type_column_present:
                cols = st.columns(2, gap="large")
                with cols[0]:
                    if selected_brand == 'All':
                        plot_pie_chart(sentiment_counts, 'Count', 'Brand')
                    elif selected_comment_type == 'All':
                        plot_pie_chart(sentiment_counts, 'Count', 'Type')
                with cols[1]:
                    if selected_brand == 'All' and selected_comment_type == 'All':
                        plot_pie_chart(sentiment_counts, 'Count', 'Type')
# Time Series (TS)
def time_series_plot(df):
    time_series_container = st.container(border=True)
    with time_series_container:
        st.subheader("Brand Reputation Score Over Time")
        selectbox_container = st.container()
        with selectbox_container:
            selectbox1, selectbox2 = st.columns(2, gap="small")
            with selectbox1:
                st.selectbox("Group by", ["Year", "Month"], key='time_series_group_by')
            if get_state('brand_column_present') and get_state('time_series_group_by') == "Month":
                with selectbox2:
                    brand = st.selectbox('Brand', df['Brand'].unique())
                    st.session_state.time_series_brand = brand
            # print_state(default_values)
            if get_state('time_series_group_by') == "Month":
                plot_brand_reputation_over_month(df)
            else:
                plot_brand_reputation_over_year(df)

def plot_brand_reputation_over_month(df):
    if not get_state('brand_column_present'):
        time_series_df = df.groupby(['Year', 'Month'])['Sentiment'].apply(compute_score).reset_index()
    else:
        brand = st.session_state.time_series_brand
        df_by_brand = df[df['Brand'] == brand]
        time_series_df = df_by_brand.groupby(['Year', 'Month'])['Sentiment'].apply(compute_score).reset_index()

    time_series_df.rename(columns={'Sentiment': 'Score'}, inplace=True)

    # Create a complete DataFrame with all year-month combinations
    all_years = df['Year'].unique()
    all_months = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M').strftime('%B')
    complete_df = pd.DataFrame([(year, month) for year in all_years for month in all_months], columns=['Year', 'Month'])

    # Merge with the actual time series data
    complete_time_series_df = pd.merge(complete_df, time_series_df, on=['Year', 'Month'], how='left')

    # Sort months chronologically
    month_order = pd.CategoricalDtype(categories=all_months, ordered=True)
    complete_time_series_df['Month'] = complete_time_series_df['Month'].astype(month_order)

    # Sort by Year (descending) and Month (chronologically)
    complete_time_series_df = complete_time_series_df.sort_values(by=['Year', 'Month'], ascending=[False, True])
    
    # Plot the line chart using Plotly
    fig = px.line(complete_time_series_df, x='Month', y='Score', color='Year', title='Brand Reputation Over Time by Month and Year',
              hover_name='Year', hover_data={'Score': True})
    fig.update_traces(mode='lines+markers') # Add markers to each point
    fig.update_xaxes(categoryorder='array', categoryarray=all_months)  # Explicitly set the month order
    fig.update_yaxes(title_text='Score')

    st.plotly_chart(fig)
    
def plot_brand_reputation_over_year(df):
    if not get_state('brand_column_present'):
        time_series_df = df.groupby('Year')['Sentiment'].apply(compute_score).reset_index()
        time_series_df.rename(columns={'Sentiment': 'Score'}, inplace=True)

        fig = px.line(time_series_df, x='Year', y='Score', title=f'Brand Reputation Over Time by Year')
        fig.update_yaxes(title_text='Score')
        st.plotly_chart(fig) 
    else:
        time_series_df = df.groupby(['Year', 'Brand'])['Sentiment'].apply(compute_score).reset_index()
        time_series_df.rename(columns={'Sentiment': 'Score'}, inplace=True)
        # Create a complete DataFrame with all year and brand combinations
        all_years = df['Year'].unique()
        all_brands = df['Brand'].unique()
        complete_df = pd.DataFrame([(year, brand) for year in all_years for brand in all_brands], columns=['Year', 'Brand'])
        
        # Merge with the actual time series data
        complete_time_series_df = pd.merge(complete_df, time_series_df, on=['Year', 'Brand'], how='left')
        
        # Sort by Year (ascending) and Brand (ascending) for plotting clarity
        complete_time_series_df = complete_time_series_df.sort_values(by=['Year', 'Brand'], ascending=[True, True])
        
        # Plot the line chart using Plotly
        fig = px.line(complete_time_series_df, x='Year', y='Score', color='Brand', title='Brand Reputation Score Over Years')
        fig.update_traces(mode='lines+markers')  # Add markers to the lines
        # Update layout
        fig.update_layout(xaxis_title='Year', yaxis_title='Brand Reputation Score')
        
        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig)

# Callback functions to update sliders
def update_social_slider():
    set_state('social_pct', 100 - get_state('ecom_pct'))

def update_ecom_slider():
    set_state('ecom_pct', 100 - get_state('social_pct'))
    
@st.cache_data
def load_file(uploaded_file, file_extension):
    if file_extension.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

@st.cache_resource
def load_model():
    return joblib.load('svm_base_model.pkl')

@st.cache_resource
def load_vectorizer():
    return joblib.load('vectorizer.pkl')

# Set page configuration with larger font size for the title
st.set_page_config(
    page_title="BReMo", 
    page_icon=":material/track_changes:", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

try:
    model = load_model()
    vectorizer = load_vectorizer()
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")

st.markdown("<h1 style='text-align: center; color: white; margin: 0 0 20px 0;'>Brand Reputation Monitoring</h1>", unsafe_allow_html=True)

navigation = option_menu(
                menu_title=None, 
                options=["Home", "Text", "File Upload"], 
                icons=['house', "file-text", "cloud-arrow-up"], 
                menu_icon="cast", 
                default_index=0, 
                orientation="horizontal"
            )

# Session State
# Default values for session state
default_values = {
    'show_result': False,
    'result': None,
    'time_series_group_by': "Year",
    'time_series_brand': "",
    'brand_column_present': False, # Competitor Analysis
    'date_column_present': False, # Time Series
    'type_column_present': False, # Weightage
    'ecom_pct': 50,
    'social_pct': 50,
    'enable_score_weightage': False,
    'progress_bar': None
}

# Initialize session state on app startup
initialize_session_state()

if navigation == "Home":
    reset_session()
    cols  = st.columns(2, gap="small")
    with cols[0]:
        show_text_tab_details()
    with cols[1]:
        show_file_upload_details()

if navigation == "Text":
    reset_session()
    st.subheader("Single Comment Prediction")

    # Text Input
    user_input = st.text_area("Enter Comment", key='input_area', height=150)
    # Predict button
    predict_button = st.button("Predict")
        # Show clear button when the text area is being typed
    if user_input:
        # Clear button
        st.button("Clear Text", key="clear_button", on_click=clear_text)
    if predict_button: # Predict button clicked
        is_valid, error_message = validate_user_input(user_input)
        if is_valid:
            with st.spinner('Processing...'):
                # Initialize the language detector
                languages = [Language.ENGLISH, Language.MALAY, Language.CHINESE, Language.KOREAN, Language.JAPANESE]
                detector = LanguageDetectorBuilder.from_languages(*languages).build()
                sentiment = predict_sentiment(preprocess_text(user_input, detector))
                st.markdown(f"<div style='font-size:20px;'>Predicted Sentiment: {sentiment}</div>", unsafe_allow_html=True)
        else:
            st.warning(error_message)

if navigation == "File Upload":
    st.subheader("File Upload Prediction")
    st.write("**Dynamic Dashboard Feature**: The dashboard adjusts based on the columns in the file.")
    st.write("   - **'Comment' Column**: Enables sentiment analysis features.")
    st.write("   - **'Type' Column**: Allows customization of weightage for 'E-commerce' and 'Social Media'.")
    st.write("   - **'Brand' Column**: Enables competitor analysis.")
    st.write("   - **'Date' Column**: Provides time series chart visualization.")
    st.write(" ")
    with st.expander("Brand Reputation Score Weightage Setting"):
        # Checkbox to enable weightage setting
        st.checkbox("Enable Brand Reputation Score Weightage", key='enable_score_weightage')

        if get_state('enable_score_weightage'): 
            st.caption("Customize the weightage for 'E-commerce' and 'Social Media' categories if your data includes a 'Type' column.")
            # Create sliders with callbacks
            st.slider(
                'E-commerce', 
                min_value=0, 
                max_value=100,
                value=50,
                key='ecom_pct',
                on_change=update_social_slider
            )
            st.slider(
                'Social Media', 
                min_value=0, 
                max_value=100,
                value=50,
                key='social_pct',
                on_change=update_ecom_slider
            )

    # File Upload
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["xlsx", "xls", "csv"] , on_change=reset_session)

    # Predict Button
    if st.button("Predict"):
        set_state('show_result', True)

    if get_state('show_result'):
        if uploaded_file is None:
            st.warning("Please upload a CSV or Excel file.")
        else:
            # Record file name
            try:
                file_name, file_extension = os.path.splitext(uploaded_file.name)
                df = load_file(uploaded_file, file_extension)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
            
            if not 'Comment' in df.columns:
                st.error("Excel file does not contain a column named 'Comment'.")
            else:
                stop_button = st.empty() # Placeholder for the Stop button

                result_tab, dashboard_tab = st.tabs(["Result", "Dashboard"])
                with st.spinner('Processing...'):
                    # Preprocess
                    df = df.dropna(subset=['Comment'])

                    # Dashboard Layout
                    if 'Date' in df.columns:
                        set_state('date_column_present', True)

                        # Create Year and Month column
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                        df = df.dropna(subset=['Date'])
                        df['Year'] = df['Date'].dt.year
                        df['Month'] = df['Date'].dt.strftime('%B')

                    if 'Brand' in df.columns:
                        set_state('brand_column_present', True)
                    if 'Type' in df.columns:
                        set_state('type_column_present', True)
                        df['Type'] = df['Type'].str.lower().replace({'ecommerce': 'e-commerce'})

                    if st.session_state.result is None:
                        set_state('progress_bar', st.progress(0))

                        # Show Stop button during processing
                        with stop_button.container():
                            if st.button("Stop"):
                                # st.warning("Prediction stopped.")
                                set_state('show_result', False)
                                get_state('progress_bar').empty()
                                stop_button.empty()
                                st.stop()
                        df = preprocess_file(df)
                        num_comments = len(df['Comment'])
                        # Prediction loop with stop functionality
                        sentiments = []
                        for i, comment in enumerate(df['Translated_Comment']):
                            sentiments.append(predict_sentiment(comment))

                            # Update the progress bar
                            progress = (i + 1) / num_comments
                            get_state('progress_bar').progress(progress)
                        df['Sentiment'] = sentiments
                        df = df.drop(columns=['Translated_Comment'])
                        st.session_state.result = df
                        # Clear progress bar and hide stop button
                        get_state('progress_bar').empty()
                        stop_button.empty()
                    else:
                        df = get_state('result')
                    
                    # Result Tab
                    with result_tab:
                        st.dataframe(df.iloc[:1000], use_container_width=True) # Display only the first 1000 rows
                        # Convert dataframe to CSV
                        csv = df.to_csv(index=False).encode('utf-8')

                        # Create a download button with a custom filename
                        st.download_button(
                            label="Download",
                            data=csv,
                            file_name=f'{file_name}_result.csv',
                            mime='text/csv'
                        )
                    
                    # Dashboard Tab
                    with dashboard_tab:
                        brand_column_present = get_state('brand_column_present')
                        date_column_present = get_state('date_column_present')

                        metric_card_plot()
                        brand_reputation_score_plot(df, brand_column_present)
                        # If has Date column
                        if date_column_present:
                            time_series_plot(df)
                        sentiment_distribution_plot()
