import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter

# Local imports
from src.data_preprocessing import load_chat,clean_chat
from src.predict import predict_chat
from src.Labelling import auto_label
from src.analysis import chat_stats, generate_wordcloud, messages_over_time, avg_message_length, top_words, emoji_usage


# -------------------------------
# Load Model + Vectorizer
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("models/spam_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="WhatsApp Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Dark Theme CSS Only
# -------------------------------
def get_theme_css():
    # Dark Theme Colors (fixed)
    bg_primary = "#0f172a"
    bg_secondary = "#1e293b"
    bg_tertiary = "#334155"
    text_primary = "#f8fafc"
    text_secondary = "#cbd5e1"
    text_muted = "#94a3b8"
    border_color = "#475569"
    accent_color = "#6366f1"
    success_color = "#10b981"
    danger_color = "#ef4444"
    card_bg = "#1e293b"
    sidebar_bg = "linear-gradient(180deg, #1e293b 0%, #334155 100%)"

    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        :root {{
            --bg-primary: {bg_primary};
            --bg-secondary: {bg_secondary};
            --bg-tertiary: {bg_tertiary};
            --text-primary: {text_primary};
            --text-secondary: {text_secondary};
            --text-muted: {text_muted};
            --border-color: {border_color};
            --accent-color: {accent_color};
            --success-color: {success_color};
            --danger-color: {danger_color};
            --card-bg: {card_bg};
        }}
        .main > div {{
            padding-top: 1rem;
            background-color: var(--bg-primary);
            color: var(--text-primary);
        }}
        .stApp {{
            background-color: var(--bg-primary);
            color: var(--text-primary);
        }}
        .css-1d391kg, .css-1cypcdb {{
            background: {sidebar_bg};
        }}
        .sidebar-content {{
            color: var(--text-primary);
        }}
        .main-title {{
            font-family: 'Inter', sans-serif;
            font-size: 2.5rem;
            font-weight: 600;
            color: var(--text-primary);
            text-align: center;
            margin-bottom: 0.5rem;
            line-height: 1.3;
        }}
        .subtitle {{
            font-family: 'Inter', sans-serif;
            font-size: 1.1rem;
            color: var(--text-muted);
            text-align: center;
            margin-bottom: 3rem;
            font-weight: 400;
        }}
        .sidebar-title {{
            font-family: 'Inter', sans-serif;
            font-size: 1.4rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .section-header {{
            font-family: 'Inter', sans-serif;
            font-size: 1.4rem;
            font-weight: 600;
            margin-top: 2rem;
            margin-bottom: 1rem;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .section-subheader {{
            font-family: 'Inter', sans-serif;
            font-size: 1rem;
            font-weight: 500;
            color: var(--text-secondary);
            margin-bottom: 1rem;
        }}
        .metric-card {{
            background: var(--card-bg);
            padding: 2rem 1.5rem;
            border-radius: 16px;
            border: 1px solid var(--border-color);
            text-align: center;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .metric-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            border-color: var(--accent-color);
        }}
        .metric-value {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            font-family: 'Inter', sans-serif;
        }}
        .metric-label {{
            font-size: 0.85rem;
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}
        .welcome-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 3rem 2rem;
            text-align: center;
            margin: 2rem auto;
            max-width: 600px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }}
        .welcome-icon {{
            font-size: 4rem;
            margin-bottom: 1rem;
            display: block;
        }}
        .welcome-text {{
            color: var(--text-secondary);
            font-size: 1.1rem;
            line-height: 1.6;
            margin-bottom: 1.5rem;
        }}
        .upload-hint {{
            color: var(--text-muted);
            font-size: 0.9rem;
            font-weight: 500;
        }}
        .progress-step {{
            display: flex;
            align-items: center;
            margin: 0.8rem 0;
            font-weight: 500;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}
        .progress-step.completed {{ color: var(--success-color); }}
        .progress-step.active {{ color: var(--accent-color); }}
        .custom-alert {{
            padding: 1rem 1.5rem;
            border-radius: 12px;
            margin: 1.5rem 0;
            font-weight: 500;
        }}
        .alert-info {{
            background: rgba(99, 102, 241, 0.1);
            color: var(--accent-color);
            border-left: 4px solid var(--accent-color);
        }}
        .alert-success {{
            background: rgba(16, 185, 129, 0.1);
            color: var(--success-color);
            border-left: 4px solid var(--success-color);
        }}
        .stDownloadButton button {{
            background: linear-gradient(135deg, var(--success-color), #059669) !important;
            color: white !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            border: none !important;
            padding: 0.75rem 2rem !important;
            transition: all 0.3s ease !important;
            font-family: 'Inter', sans-serif !important;
        }}
        .stDownloadButton button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3) !important;
        }}
        .stFileUploader > div > div {{
            background: var(--card-bg) !important;
            border: 2px dashed var(--border-color) !important;
            border-radius: 16px !important;
            padding: 2rem !important;
            text-align: center !important;
            transition: all 0.3s ease !important;
            color: var(--text-primary) !important;
        }}
        .stFileUploader > div > div:hover {{
            border-color: var(--accent-color) !important;
            background: var(--bg-secondary) !important;
        }}
        .stDataFrame {{
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }}
        .stSelectbox > div > div {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
        }}
        .chart-container {{
            background: var(--card-bg);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid var(--border-color);
            margin: 1rem 0;
        }}
        .footer {{
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-top: 3rem;
            padding: 2rem 0;
            border-top: 1px solid var(--border-color);
        }}

        /* Section headers for analysis */
        .analysis-section-header {{
            font-family: 'Inter', sans-serif;
            font-size: 1.3rem;
            font-weight: 600;
            margin-top: 2rem;
            margin-bottom: 1rem;
            color: var(--text-primary);
}}

/* Metrics inside analysis (total messages, participants) */
        .analysis-metric {{
            background: var(--card-bg);
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            border: 1px solid var(--border-color);
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }}
        .analysis-metric:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
}}

/* Bar charts / line charts / time series */
        .stLineChart, .stBarChart, .chart-container {{
            background: var(--card-bg);
            border-radius: 16px;
            padding: 1rem;
            border: 1px solid var(--border-color);
            margin: 1rem 0;
}}

/* Word Cloud */
        .stImage img {{
            border-radius: 16px;
            border: 1px solid var(--border-color);
            margin: 1rem 0;
}}

/* Top Words and Emoji lists */
        .analysis-list {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            margin-bottom: 1rem;
}}
        .analysis-list span {{
            font-weight: 600;
            color: var(--accent-color);
}}

/* Export buttons (already have download button style, but we can refine) */
        .stDownloadButton button {{
            font-size: 0.95rem !important;
            padding: 0.6rem 1.5rem !important;
}}

/* Alert for analysis completion */
        .custom-alert {{
            padding: 1rem 1.5rem;
            border-radius: 12px;
            margin: 1.5rem 0;
            font-weight: 500;
}}
        .alert-success {{
            background: rgba(16, 185, 129, 0.1);
            color: var(--success-color);
            border-left: 4px solid var(--success-color);
}}

/* Remove sidebar progress steps */
        .sidebar .progress-step {{
            display: none !important;
}}
    </style>
    """

# Apply theme CSS (always dark)
st.markdown(get_theme_css(), unsafe_allow_html=True)

# -------------------------------
# Sidebar Content
# -------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>🛡️ Setup</div>", unsafe_allow_html=True)
    with st.expander("📖 Export Instructions", expanded=False):
        st.markdown("""
        **Android:**
        1. Open WhatsApp → Chat
        2. Tap ⋮ → More → Export Chat
        3. Choose "Without Media"
        
        **iPhone:**
        1. Open WhatsApp → Chat
        2. Tap contact name → Export Chat
        3. Choose "Without Media"
        """)
    st.markdown("---")
    st.markdown("### 📂 Upload Chat")
    uploaded_file = st.file_uploader(
        "Choose WhatsApp .txt file",
        type=["txt"],
        help="Upload exported WhatsApp chat"
    )
    if uploaded_file:
        uploaded_file.seek(0)
        file_size_mb = len(uploaded_file.read()) / (1024 * 1024)
        uploaded_file.seek(0)
        st.success(f"✅ {uploaded_file.name}")
        st.caption(f"Size: {file_size_mb:.2f} MB")
    

# -------------------------------
# Main Content Area
# -------------------------------
st.markdown("<div class='main-title'>🛡️ WhatsApp Spam Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered spam detection for WhatsApp chats</div>", unsafe_allow_html=True)




if uploaded_file is None:
    # Minimal, subtle welcome screen
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-icon">📱</div>
        <div class="welcome-text">
            Upload your WhatsApp chat export to analyze messages and detect spam using advanced AI algorithms.
        </div>
        <div class="upload-hint">
            👈 Use the sidebar to get started
        </div>
    </div>
    """, unsafe_allow_html=True)

else:

    # Process the uploaded file
    file_path = "temp_chat.txt"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # Load and process chat
        with st.spinner("🔄 Processing chat..."):
            df = load_chat(file_path)
            df = df[df["message"].apply(lambda x: isinstance(x, str) and x.strip() != "")]

        # Chat Overview
        st.markdown("<div class='section-header'>📋 Chat Overview</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(df.head(8), use_container_width=True, hide_index=True)
        
        with col2:
            st.metric("💬 Messages", len(df))
            if 'sender' in df.columns:
                st.metric("👥 Participants", df['sender'].nunique())

        # Run spam detection
        st.markdown("<div class='section-header'>🔍 Spam Analysis</div>", unsafe_allow_html=True)
        
        with st.spinner("🤖 Detecting spam..."):
            results = predict_chat(file_path)
            results = auto_label(results)
            results["final_prediction"] = results.apply(
                lambda row: "Spam" if row["prediction"] == "Spam" or row["auto_spam"] else "Ham",
                axis=1,
            )

        # Results metrics
        total_msgs = len(results)
        spam_msgs = (results["final_prediction"] == "Spam").sum()
        ham_msgs = total_msgs - spam_msgs
        spam_rate = (spam_msgs / total_msgs * 100) if total_msgs > 0 else 0

        # Metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: var(--accent-color);">{total_msgs}</div>
                <div class="metric-label">Total Messages</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: var(--danger-color);">{spam_msgs}</div>
                <div class="metric-label">Spam Detected</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: var(--success-color);">{ham_msgs}</div>
                <div class="metric-label">Clean Messages</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #f59e0b;">{spam_rate:.1f}%</div>
                <div class="metric-label">Spam Rate</div>
            </div>
            """, unsafe_allow_html=True)

        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Theme-aware pie chart
            colors = ['#ef4444', '#10b981']  
            fig_pie = px.pie(
                values=[spam_msgs, ham_msgs],
                names=["🚫 Spam", "✅ Ham"],
                color_discrete_sequence=colors,
                title="<b>Message Distribution</b>",
                hole=0.4
            )
            fig_pie.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc' ,
                height=350,
                title_x=0.5
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            if 'sender' in results.columns and results['sender'].nunique() > 1:
                sender_stats = results.groupby('sender')['final_prediction'].apply(
                    lambda x: (x == 'Spam').sum()
                ).reset_index()
                sender_stats.columns = ['Sender', 'Spam Count']
                sender_stats = sender_stats.sort_values('Spam Count', ascending=True).tail(8)
                
                fig_bar = px.bar(
                    sender_stats,
                    x='Spam Count',
                    y='Sender',
                    orientation='h',
                    title="<b>Spam by Sender</b>",
                    color_discrete_sequence=['#6366f1']
                )
                fig_bar.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f8fafc',
                    height=350,
                    title_x=0.5
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        # Results table with filters
        st.markdown("<div class='section-header'>📄 Detailed Results</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            filter_type = st.selectbox("Filter", ["All Messages", "Spam Only", "Ham Only"])
        with col2:
            show_count = st.selectbox("Show", [20, 50, 100])

        # Apply filters
        filtered_results = results.copy()
        if filter_type == "Spam Only":
            filtered_results = filtered_results[filtered_results["final_prediction"] == "Spam"]
        elif filter_type == "Ham Only":
            filtered_results = filtered_results[filtered_results["final_prediction"] == "Ham"]

        filtered_results = filtered_results.tail(show_count)

        # Color-coded results
        def color_predictions(val):
            if val == 'Spam':
                return 'background-color: rgba(239, 68, 68, 0.1); color: #ef4444; font-weight: 600;'
            else:
                return 'background-color: rgba(16, 185, 129, 0.1); color: #10b981; font-weight: 600;'

        display_cols = ["sender", "message", "final_prediction"] if 'sender' in filtered_results.columns else ["message", "final_prediction"]
        styled_df = filtered_results[display_cols].style.applymap(
            color_predictions, 
            subset=['final_prediction']
        )
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        st.markdown("<div class='section-header'>📊 WhatsApp Analysis</div>", unsafe_allow_html=True)



        chat_df = clean_chat(df)

# -------------------------------
# Basic Stats
# -------------------------------
        total_msgs, participants, active_senders = chat_stats(chat_df)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📨 Total Messages", total_msgs)
            st.metric("👥 Participants", participants)
        with col2:
            st.markdown("<div class='section-header'>📈 Active Senders</div>", unsafe_allow_html=True)
            st.bar_chart(active_senders)

# -------------------------------
# Word Cloud
# -------------------------------
        wc = generate_wordcloud(chat_df)
        if wc:
            st.markdown("<div class='section-header'>☁️ Word Cloud</div>", unsafe_allow_html=True)
            st.image(wc.to_array(), caption="Word Cloud", use_container_width=True)

# -------------------------------
# Messages Over Time
# -------------------------------
        daily_msgs = messages_over_time(chat_df)
        if not daily_msgs.empty:
            st.markdown("<div class='section-header'>📅 Messages Over Time</div>", unsafe_allow_html=True)
            st.line_chart(daily_msgs)
        with st.expander("Show Table"):
            st.dataframe(daily_msgs, use_container_width=True)

# Average Message Length
        avg_len = avg_message_length(chat_df)
        if not avg_len.empty:
            st.markdown("<div class='section-header'>✏️ Average Message Length</div>", unsafe_allow_html=True)
            st.bar_chart(avg_len)
        with st.expander("Show Table"):
            st.dataframe(avg_len, use_container_width=True)

# -------------------------------
# Top Words
# -------------------------------
        st.markdown("<div class='section-header'>📝 Top Words</div>", unsafe_allow_html=True)

# Let user select how many top words to show
        show_count_words = st.selectbox("Show Top Words", [10, 15, 20, 30], index=1)

# Get top words
        top_words_list = top_words(chat_df, n=show_count_words)
        top_words_df = pd.DataFrame(top_words_list, columns=['Word','Count'])

# Display as horizontal bar chart
        import plotly.express as px
        fig_words = px.bar(
        top_words_df[::-1],  # reverse for vertical ordering
        x='Count',
        y='Word',
        orientation='h',
        text='Count',
        color='Count',
        color_continuous_scale='blues',
        title="<b>Top Words</b>"
    )
        fig_words.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#f8fafc',
        height=400,
        title_x=0.5,
        yaxis=dict(automargin=True)
)
        st.plotly_chart(fig_words, use_container_width=True)

# Optional: Show table below chart
        with st.expander("Show Data Table"):
            st.dataframe(top_words_df, use_container_width=True)


# -------------------------------
# Emoji Usage
# -------------------------------
        emoji_counts = emoji_usage(chat_df)

        if emoji_counts:
            st.markdown("<div class='section-header'>😀 Top Emojis</div>", unsafe_allow_html=True)
        for emoji, count in emoji_counts:
            st.write(f"{emoji} : {count}")

# ---------------- Export Section ----------------
        st.markdown("<div class='section-header'>📥 Export Results</div>", unsafe_allow_html=True)
        full_csv_name = f"{uploaded_file.name.split('.')[0]}_full.csv"
        analysis_csv_name = f"{uploaded_file.name.split('.')[0]}_analysis.csv"

# Full CSV with predictions
        full_csv = results.to_csv(index=False).encode("utf-8")

# Analysis CSV (aggregated stats)
        summary_data = {
    "Metric": ["Total Messages", "Spam Messages", "Clean Messages", "Spam Rate (%)", "Participants"],
    "Value": [total_msgs, spam_msgs, ham_msgs, spam_rate, df['sender'].nunique() if 'sender' in df.columns else 1]
}
        summary_df = pd.DataFrame(summary_data)

# 2️⃣ Active Senders
        active_senders_df = active_senders.reset_index()
        active_senders_df.columns = ["Sender", "Message Count"]

# 3️⃣ Messages Over Time
        daily_msgs_df = messages_over_time(chat_df).reset_index()
        daily_msgs_df.columns = ["Date", "Messages"]

# 4️⃣ Average Message Length
        avg_len_df = avg_message_length(chat_df).reset_index()
        avg_len_df.columns = ["Sender", "Avg Message Length"]

# 5️⃣ Top Words
        top_words_df = pd.DataFrame(top_words(chat_df, n=show_count_words), columns=["Word", "Count"])

# 6️⃣ Emoji Usage
        emoji_counts = emoji_usage(chat_df)
        emoji_df = pd.DataFrame(emoji_counts, columns=["Emoji", "Count"])

# Combine all into one CSV with blank rows as separators
        from io import StringIO
        output = StringIO()

        summary_df.to_csv(output, index=False)
        output.write("\n\n")  # blank row

        output.write("Active Senders\n")
        active_senders_df.to_csv(output, index=False)
        output.write("\n\n")

        output.write("Messages Over Time\n")
        daily_msgs_df.to_csv(output, index=False)
        output.write("\n\n")

        output.write("Average Message Length\n")
        avg_len_df.to_csv(output, index=False)
        output.write("\n\n")

        output.write("Top Words\n")
        top_words_df.to_csv(output, index=False)
        output.write("\n\n")

        output.write("Emoji Usage\n")
        emoji_df.to_csv(output, index=False)

        analysis_csv = output.getvalue().encode("utf-8")


        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
            label="📊 Predictions CSV",
                data=full_csv,
                file_name=full_csv_name,
                mime="text/csv"
            )

        with col2:
            st.download_button(
            label="📈 Analysis CSV",
            data=analysis_csv,
            file_name=analysis_csv_name,
            mime="text/csv"
    )

        st.markdown('<div class="custom-alert alert-success">✅ Analysis completed successfully!</div>', unsafe_allow_html=True)
    
    except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")

# Footer
st.markdown("""
<div class="footer">
    Built with Streamlit • Powered by Machine Learning
</div>
""", unsafe_allow_html=True)