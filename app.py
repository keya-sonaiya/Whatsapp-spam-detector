import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Local imports
from src.data_preprocessing import load_chat, clean_chat
from src.predict import predict_chat
from src.Labelling import auto_label
from src.analysis import (
    chat_stats,
    generate_wordcloud,
    messages_over_time,
    avg_message_length,
    top_words,
    emoji_usage,
)

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
    initial_sidebar_state="expanded",
)

# -------------------------------
# THEME + CSS
# -------------------------------


def get_theme_css(mode: str = "dark", font_size: str = "medium") -> str:
    mode = mode.lower()

    size_map = {
        "small": 0.85,
        "medium": 1.0,
        "large": 1.15,
        "extra large": 1.3,
    }
    multiplier = size_map.get(font_size.lower(), 1.0)

    if mode == "light":
        bg_primary = "#f9fafb"
        bg_secondary = "#ffffff"
        bg_tertiary = "#e5e7eb"
        text_primary = "#111827"
        text_secondary = "#374151"
        text_muted = "#6b7280"
        border_color = "#d1d5db"
        accent_color = "#4f46e5"
        success_color = "#16a34a"
        danger_color = "#dc2626"
        card_bg = "#ffffff"
        sidebar_bg = "linear-gradient(180deg, #ffffff 0%, #f3f4f6 100%)"
        input_bg = "#f9fafb"
        input_text = "#111827"
    else:
        bg_primary = "#0f172a"
        bg_secondary = "#020617"
        bg_tertiary = "#111827"
        text_primary = "#f1f5f9"
        text_secondary = "#cbd5e1"
        text_muted = "#94a3b8"
        border_color = "#475569"
        accent_color = "#6366f1"
        success_color = "#10b981"
        danger_color = "#ef4444"
        card_bg = "#020617"
        sidebar_bg = "linear-gradient(180deg, #020617 0%, #020617 100%)"
        input_bg = "#020617"
        input_text = "#f1f5f9"

    base_font = 16 * multiplier
    small_font = 14 * multiplier
    medium_font = 16 * multiplier
    large_font = 18 * multiplier
    title_font = 40 * multiplier
    subtitle_font = 18 * multiplier
    section_font = 22 * multiplier
    metric_value_font = 40 * multiplier
    metric_label_font = 13 * multiplier

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
            --input-bg: {input_bg};
            --input-text: {input_text};
        }}

        body, [data-testid="stMarkdownContainer"], .stMarkdown, .stText, label {{
            color: var(--text-primary) !important;
            font-family: 'Inter', sans-serif !important;
            font-size: {base_font}px !important;
        }}

        [data-testid="stAppViewContainer"] {{
            background-color: var(--bg-primary) !important;
        }}

        [data-testid="stAppViewContainer"] .main {{
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
        }}

        .stApp {{
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
        }}

        p, span, div, label, h1, h2, h3, h4, h5, h6 {{
            color: var(--text-primary) !important;
        }}

        [data-testid="stSidebar"] {{
            background: {sidebar_bg} !important;
        }}

        [data-testid="stSidebar"] *, 
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {{
            color: var(--text-primary) !important;
            font-size: {small_font}px !important;
        }}

        [data-testid="stSidebar"] .stRadio label {{
            color: var(--text-primary) !important;
            font-size: {small_font}px !important;
        }}

        [data-testid="stSidebar"] hr {{
            border: none !important;
            border-top: 1px solid var(--border-color) !important;
            opacity: 1 !important;
            margin: 1.25rem 0 !important;
        }}

        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
            background: var(--card-bg) !important;
            border: 2px dashed var(--border-color) !important;
            border-radius: 16px !important;
            padding: 2rem !important;
            text-align: center !important;
        }}

        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {{
            color: var(--text-primary) !important;
            font-size: {small_font}px !important;
        }}

        .stSidebar button, [data-testid="stFileUploaderDropzone"] button {{
            background: var(--success-color) !important;
            color: #ffffff !important;
            border-radius: 10px !important;
            padding: 0.5rem 1.2rem !important;
            font-size: {small_font}px !important;
            border: none !important;
            font-weight: 600 !important;
        }}
        .stSidebar button:hover {{
            filter: brightness(1.12);
            transform: translateY(-1px);
        }}

        [data-testid="stSidebar"] .stSuccess {{
            background-color: rgba(16, 185, 129, 0.1) !important;
            color: var(--success-color) !important;
            font-size: {small_font}px !important;
        }}
        [data-testid="stSidebar"] .stSuccess * {{
            color: var(--success-color) !important;
            font-size: {small_font}px !important;
        }}

        [data-testid="stSidebar"] .stCaption {{
            color: var(--text-muted) !important;
            font-size: {small_font * 0.9}px !important;
        }}

        .main-title {{
            font-size: {title_font}px !important;
            font-weight: 600 !important;
            color: var(--text-primary) !important;
            text-align: center !important;
            margin-bottom: 0.5rem !important;
            line-height: 1.3 !important;
        }}
        .subtitle {{
            font-size: {subtitle_font}px !important;
            color: var(--text-muted) !important;
            text-align: center !important;
            margin-bottom: 3rem !important;
            font-weight: 400 !important;
        }}
        .sidebar-title {{
            font-size: {large_font}px !important;
            font-weight: 600 !important;
            color: var(--text-primary) !important;
            margin-bottom: 1.5rem !important;
            display: flex !important;
            align-items: center !important;
            gap: 0.5rem !important;
        }}
        .section-header {{
            font-size: {section_font}px !important;
            font-weight: 600 !important;
            margin-top: 2rem !important;
            margin-bottom: 1rem !important;
            color: var(--text-primary) !important;
            display: flex !important;
            align-items: center !important;
            gap: 0.5rem !important;
        }}

        .metric-card {{
            background: var(--card-bg) !important;
            padding: 2rem 1.5rem !important;
            border-radius: 16px !important;
            border: 1px solid var(--border-color) !important;
            text-align: center !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
        }}
        .metric-card:hover {{
            transform: translateY(-4px) !important;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1) !important;
            border-color: var(--accent-color) !important;
        }}
        .metric-value {{
            font-size: {metric_value_font}px !important;
            font-weight: 700 !important;
            margin-bottom: 0.5rem !important;
        }}
        .metric-label {{
            font-size: {metric_label_font}px !important;
            color: var(--text-muted) !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.1em !important;
        }}

        .sidebar-help-card {{
            background-color: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            padding: 1rem 1.25rem !important;
            margin-top: 0.5rem !important;
            margin-bottom: 1rem !important;
            font-size: {small_font}px !important;
        }}
        .sidebar-help-card strong {{
            color: var(--text-primary) !important;
        }}
        .sidebar-help-card p,
        .sidebar-help-card li {{
            color: var(--text-primary) !important;
            font-size: {small_font}px !important;
        }}

        .welcome-card {{
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 20px !important;
            padding: 3rem 2rem !important;
            text-align: center !important;
            margin: 2rem auto !important;
            max-width: 600px !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
        }}
        .welcome-icon {{
            font-size: {metric_value_font * 1.2}px !important;
            margin-bottom: 1rem !important;
            display: block !important;
        }}
        .welcome-text {{
            color: var(--text-secondary) !important;
            font-size: {small_font * 1.1}px !important;
            line-height: 1.5 !important;
            margin-bottom: 1rem !important;
        }}
        .upload-hint {{
            color: var(--text-muted) !important;
            font-size: {small_font * 1.1}px !important;
            font-weight: 500 !important;
        }}

        [data-testid="stDataFrame"] {{
            background-color: var(--card-bg) !important;
        }}
        [data-testid="stDataFrame"] .row_heading, 
        [data-testid="stDataFrame"] .blank,
        [data-testid="stDataFrame"] .col_heading {{
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }}
        [data-testid="stDataFrame"] .data {{
            background-color: transparent !important;
            color: var(--text-primary) !important;
            border-color: var(--border-color) !important;
        }}
        [data-testid="stDataFrame"] * {{
            font-family: 'Inter', sans-serif !important;
            font-size: {small_font}px !important;
            color: var(--text-primary) !important;
        }}
        [data-testid="stDataFrame"] tbody tr {{
            background-color: var(--card-bg) !important;
        }}
        [data-testid="stDataFrame"] tbody tr:hover {{
            background-color: var(--bg-secondary) !important;
        }}

        [data-testid="stMetric"] {{
            background-color: var(--card-bg) !important;
            padding: 1rem !important;
            border-radius: 12px !important;
            border: 1px solid var(--border-color) !important;
        }}
        [data-testid="stMetric"] label {{
            color: var(--text-primary) !important;
            font-size: {medium_font}px !important;
        }}
        [data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: var(--text-primary) !important;
            font-size: {section_font}px !important;
        }}

        .footer {{
            text-align: center !important;
            color: var(--text-muted) !important;
            font-size: {small_font}px !important;
            margin-top: 3rem !important;
            padding: 2rem 0 !important;
            border-top: 1px solid var(--border-color) !important;
        }}

        .custom-alert {{
            padding: 1rem 1.5rem !important;
            border-radius: 12px !important;
            margin: 1.5rem 0 !important;
            font-weight: 500 !important;
            font-size: {medium_font}px !important;
        }}
        .alert-success {{
            background: rgba(16, 185, 129, 0.1) !important;
            color: var(--success-color) !important;
            border-left: 4px solid var(--success-color) !important;
        }}

        .stSpinner > div {{
            color: var(--text-primary) !important;
            font-size: {medium_font}px !important;
        }}

        .stError {{
            background-color: rgba(239, 68, 68, 0.1) !important;
            color: var(--danger-color) !important;
            font-size: {medium_font}px !important;
        }}

        .stDownloadButton button, .stButton button {{
            background-color: #10b981 !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.6rem 1.2rem !important;
            font-size: 14px !important;
            font-weight: 600 !important;
            transition: 0.25s ease !important;
        }}

        .stDownloadButton button:hover, .stButton button:hover {{
            background-color: #15803d !important;
            transform: translateY(-2px) !important;
        }}

        .section-separator {{
            border: none;
            border-top: 1px solid var(--border-color) !important;
            opacity: 1 !important;
            margin: 1.5rem 0 !important;
        }}

    </style>
    """



def get_plotly_theme(mode: str = "dark"):
    """Return common layout colors for Plotly charts."""
    mode = mode.lower()
    if mode == "light":
        font_color = "#111827"
        grid_color = "rgba(107,114,128,0.3)"
        axis_color = "#9ca3af"
    else:
        font_color = "#f1f5f9"
        grid_color = "rgba(148,163,184,0.3)"
        axis_color = "#64748b"

    base_layout = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=font_color, family="Inter, sans-serif", size=14),
        title=dict(font=dict(color=font_color, size=18)),
        xaxis=dict(
            linecolor=axis_color,
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            title_font=dict(color=font_color),
            tickfont=dict(color=font_color),
        ),
        yaxis=dict(
            linecolor=axis_color,
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            title_font=dict(color=font_color),
            tickfont=dict(color=font_color),
        ),
        legend=dict(font=dict(color=font_color)),
    )
    return base_layout, font_color


# SIDEBAR (theme + upload)
with st.sidebar:
    theme_mode = st.radio("🎨 Theme", ("Dark", "Light"), index=0, horizontal=True)
    font_size = "Medium"  # Fixed font size

# apply CSS AFTER we know theme and font size
st.markdown(get_theme_css(theme_mode, font_size), unsafe_allow_html=True)

plot_layout, plot_font_color = get_plotly_theme(theme_mode)

with st.sidebar:
    st.markdown("<div class='sidebar-title'>🛡️ Setup</div>", unsafe_allow_html=True)

    # Checkbox to show/hide instructions (no expander = no dark header bug)
    show_help = st.checkbox("📖 Export Instructions", value=False, key="export_help")

    if show_help:
        st.markdown(
            """
            <div class="sidebar-help-card">
                <p><strong>Android:</strong></p>
                <ol>
                    <li>Open WhatsApp → Chat</li>
                    <li>Tap ⋮ → More → Export Chat</li>
                    <li>Choose "Without Media"</li>
                </ol>
                <p><strong>iPhone:</strong></p>
                <ol>
                    <li>Open WhatsApp → Chat</li>
                    <li>Tap contact name → Export Chat</li>
                    <li>Choose "Without Media"</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 📂 Upload Chat")
    uploaded_file = st.file_uploader(
        "Choose WhatsApp .txt file",
        type=["txt"],
        help="Upload exported WhatsApp chat",
    )

    if uploaded_file:
        uploaded_file.seek(0)
        file_size_mb = len(uploaded_file.read()) / (1024 * 1024)
        uploaded_file.seek(0)
        st.success(f"✅ {uploaded_file.name}")
        st.caption(f"Size: {file_size_mb:.2f} MB")


# -------------------------------
# MAIN TITLE
# -------------------------------
st.markdown(
    "<div class='main-title'>🛡️ WhatsApp Spam Detector</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='subtitle'>AI-powered spam detection for WhatsApp chats</div>",
    unsafe_allow_html=True,
)

# -------------------------------
# NO FILE → WELCOME CARD
# -------------------------------
if uploaded_file is None:
    st.markdown(
        """
    <div class="welcome-card">
        <div class="welcome-icon">📱</div>
        <div class="welcome-text">
            Upload your WhatsApp chat export to analyze messages and detect spam using advanced AI algorithms.
        </div>
        <div class="upload-hint">
            👈 Use the sidebar to get started
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

else:
    # Save temp file
    file_path = "temp_chat.txt"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # -------------------------------
        # LOAD + CLEAN CHAT
        # -------------------------------
        with st.spinner("🔄 Processing chat..."):
            df = load_chat(file_path)
            df = df[df["message"].apply(lambda x: isinstance(x, str) and x.strip() != "")]

        # -------------------------------
        # CHAT OVERVIEW
        # -------------------------------
        st.markdown(
            "<div class='section-header'>📋 Chat Overview</div>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns([2, 1])

        with col1:
            st.dataframe(df.head(8), use_container_width=True, hide_index=True)

        with col2:
            st.metric("💬 Messages", len(df))
            if "sender" in df.columns:
                st.metric("👥 Participants", df["sender"].nunique())
        st.markdown("<hr class='section-separator'>", unsafe_allow_html=True)

        # -------------------------------
        # SPAM ANALYSIS
        # -------------------------------
        st.markdown(
            "<div class='section-header'>🔍 Spam Analysis</div>",
            unsafe_allow_html=True,
        )

        with st.spinner("🤖 Detecting spam..."):
            results = predict_chat(file_path)
            results = auto_label(results)
            results["final_prediction"] = results.apply(
                lambda row: "Spam"
                if row["prediction"] == "Spam" or row["auto_spam"]
                else "Ham",
                axis=1,
            )

        total_msgs = len(results)
        spam_msgs = (results["final_prediction"] == "Spam").sum()
        ham_msgs = total_msgs - spam_msgs
        spam_rate = (spam_msgs / total_msgs * 100) if total_msgs > 0 else 0

        # METRIC CARDS
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value" style="color: var(--accent-color);">{total_msgs}</div>
                <div class="metric-label">Total Messages</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value" style="color: var(--danger-color);">{spam_msgs}</div>
                <div class="metric-label">Spam Detected</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value" style="color: var(--success-color);">{ham_msgs}</div>
                <div class="metric-label">Clean Messages</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #f59e0b;">{spam_rate:.1f}%</div>
                <div class="metric-label">Spam Rate</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # -------------------------------
        # SPAM vs HAM PIE + SPAM BY SENDER
        # -------------------------------
        col1, col2 = st.columns(2)

        with col1:
            colors = ["#ef4444", "#10b981"]
            fig_pie = px.pie(
                values=[spam_msgs, ham_msgs],
                names=["🚫 Spam", "✅ Ham"],
                color_discrete_sequence=colors,
                hole=0.4,
                title="<b>Message Distribution</b>",
            )
            fig_pie.update_layout(**plot_layout, height=350, title_x=0.5)
            fig_pie.update_traces(textfont=dict(color=plot_font_color))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            if "sender" in results.columns and results["sender"].nunique() > 1:
                sender_stats = (
                    results.groupby("sender")["final_prediction"]
                    .apply(lambda x: (x == "Spam").sum())
                    .reset_index()
                )
                sender_stats.columns = ["Sender", "Spam Count"]
                sender_stats = sender_stats.sort_values(
                    "Spam Count", ascending=True
                ).tail(8)

                fig_bar = px.bar(
                    sender_stats,
                    x="Spam Count",
                    y="Sender",
                    orientation="h",
                    title="<b>Spam by Sender</b>",
                    color_discrete_sequence=["#6366f1"],
                )
                fig_bar.update_layout(**plot_layout, height=350, title_x=0.5)
                st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("<hr class='section-separator'>", unsafe_allow_html=True)

        # -------------------------------
        # DETAILED RESULTS TABLE
        # -------------------------------
        st.markdown(
            "<div class='section-header'>📄 Detailed Results</div>",
            unsafe_allow_html=True,
        )
        

        col1, col2 = st.columns(2)
        with col1:
            filter_type = st.radio(
                "Filter",
                ["All Messages", "Spam Only", "Ham Only"],
                index=0,
                horizontal=True,
                key="filter_radio",
            )

        with col2:
            show_count = st.radio(
                "Show",
                [20, 50, 100],
                index=0,
                horizontal=True,
                key="detailed_show_radio",
            )

        filtered_results = results.copy()
        if filter_type == "Spam Only":
            filtered_results = filtered_results[
                filtered_results["final_prediction"] == "Spam"
            ]
        elif filter_type == "Ham Only":
            filtered_results = filtered_results[
                filtered_results["final_prediction"] == "Ham"
            ]

        filtered_results = filtered_results.tail(show_count)

        def color_predictions(val):
            if val == "Spam":
                return (
                    "background-color: rgba(239, 68, 68, 0.12); "
                    "color: #b91c1c; font-weight: 600;"
                )
            else:
                return (
                    "background-color: rgba(22, 163, 74, 0.12); "
                    "color: #166534; font-weight: 600;"
                )

        display_cols = (
            ["sender", "message", "final_prediction"]
            if "sender" in filtered_results.columns
            else ["message", "final_prediction"]
        )
        styled_df = filtered_results[display_cols].style.applymap(
            color_predictions, subset=["final_prediction"]
        )
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        st.markdown("<hr class='section-separator'>", unsafe_allow_html=True)

        # -------------------------------
        # WHATSAPP ANALYSIS
        # -------------------------------
        st.markdown(
            "<div class='section-header'>📊 WhatsApp Analysis</div>",
            unsafe_allow_html=True,
        )

        chat_df = clean_chat(df)

        # Basic stats
        total_msgs_ana, participants, active_senders = chat_stats(chat_df)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📨 Total Messages", total_msgs_ana)
            st.metric("👥 Participants", participants)
        with col2:
            st.markdown(
                "<div class='section-header'>📈 Active Senders</div>",
                unsafe_allow_html=True,
            )
            active_df = active_senders.reset_index()
            active_df.columns = ["Sender", "Message Count"]
            fig_active = px.bar(
                active_df,
                x="Sender",
                y="Message Count",
                title="<b>Active Senders</b>",
                color_discrete_sequence=["#6366f1"],
            )
            fig_active.update_layout(**plot_layout, height=350, title_x=0.5)
            st.plotly_chart(fig_active, use_container_width=True)
        st.markdown("<hr class='section-separator'>", unsafe_allow_html=True)

        # Word Cloud
        wc = generate_wordcloud(chat_df)
        if wc:
            st.markdown(
                "<div class='section-header'>☁️ Word Cloud</div>",
                unsafe_allow_html=True,
            )
            st.image(wc.to_array(), caption="Word Cloud", use_container_width=True)
            st.markdown("<hr class='section-separator'>", unsafe_allow_html=True)

        # Messages over time
        daily_msgs = messages_over_time(chat_df)
        if daily_msgs is not None and not daily_msgs.empty:
            st.markdown(
                "<div class='section-header'>📅 Messages Over Time</div>",
                unsafe_allow_html=True,
            )

            if isinstance(daily_msgs, pd.Series):
                df_time = daily_msgs.reset_index()
                df_time.columns = ["Date", "Messages"]
            else:
                df_time = daily_msgs.copy()
                df_time.columns = ["Date", "Messages"]

            fig_time = px.line(
                df_time,
                x="Date",
                y="Messages",
                title="<b>Messages Over Time</b>",
            )
            fig_time.update_traces(mode="lines+markers")
            fig_time.update_layout(**plot_layout, height=400, title_x=0.5)
            st.plotly_chart(fig_time, use_container_width=True)

            show_time_table = st.checkbox(
                "Show Messages Table", value=False, key="time_table"
            )

            if show_time_table:
                st.dataframe(df_time, use_container_width=True)
        st.markdown("<hr class='section-separator'>", unsafe_allow_html=True)

        # Average message length
        avg_len = avg_message_length(chat_df)
        if avg_len is not None and not avg_len.empty:
            st.markdown(
                "<div class='section-header'>✏️ Average Message Length</div>",
                unsafe_allow_html=True,
            )
            if isinstance(avg_len, pd.Series):
                avg_df = avg_len.reset_index()
                avg_df.columns = ["Sender", "Avg Length"]
            else:
                avg_df = avg_len.copy()
                avg_df.columns = ["Sender", "Avg Length"]

            fig_avg = px.bar(
                avg_df,
                x="Sender",
                y="Avg Length",
                title="<b>Average Message Length</b>",
                color_discrete_sequence=["#0ea5e9"],
            )
            fig_avg.update_layout(**plot_layout, height=400, title_x=0.5)
            st.plotly_chart(fig_avg, use_container_width=True)

            show_avg_table = st.checkbox(
                "Show Avg Length Table", value=False, key="avglen_table"
            )

            if show_avg_table:
                st.dataframe(avg_df, use_container_width=True)
        st.markdown("<hr class='section-separator'>", unsafe_allow_html=True)

        # Top words
        st.markdown(
            "<div class='section-header'>📝 Top Words</div>",
            unsafe_allow_html=True,
        )
        show_count_words = st.radio(
            "Show Top Words",
            [10, 15, 20, 30],
            index=1,
            horizontal=True,
        )
        top_words_list = top_words(chat_df, n=show_count_words)
        top_words_df = pd.DataFrame(top_words_list, columns=["Word", "Count"])

        fig_words = px.bar(
            top_words_df[::-1],
            x="Count",
            y="Word",
            orientation="h",
            text="Count",
            color="Count",
            color_continuous_scale="Blues",
            title="<b>Top Words</b>",
        )
        fig_words.update_layout(**plot_layout, height=400, title_x=0.5, showlegend=False)
        fig_words.update_traces(
            textfont=dict(color=plot_font_color),
            textposition="outside",
        )
        fig_words.update_xaxes(
            showticklabels=True, tickfont=dict(color=plot_font_color)
        )
        fig_words.update_yaxes(
            showticklabels=True, tickfont=dict(color=plot_font_color)
        )
        fig_words.update_layout(
            coloraxis_colorbar=dict(
                tickfont=dict(color=plot_font_color, size=12),
                title=dict(font=dict(color=plot_font_color, size=12)),
            )
        )

        st.plotly_chart(fig_words, use_container_width=True)

        show_words_table = st.checkbox(
            "Show Top Words Table", value=False, key="words_table"
        )

        if show_words_table:
            st.dataframe(top_words_df, use_container_width=True)
        st.markdown("<hr class='section-separator'>", unsafe_allow_html=True)

        # Emoji usage
        emoji_counts = emoji_usage(chat_df)
        if emoji_counts:
            st.markdown(
                "<div class='section-header'>😀 Top Emojis</div>",
                unsafe_allow_html=True,
            )
            for e, count in emoji_counts:
                st.write(f"{e} : {count}")
        st.markdown("<hr class='section-separator'>", unsafe_allow_html=True)

        # -------------------------------
        # EXPORT
        # -------------------------------
        st.markdown(
            "<div class='section-header'>📥 Export Results</div>",
            unsafe_allow_html=True,
        )

        full_csv_name = f"{uploaded_file.name.split('.')[0]}_full.csv"
        analysis_csv_name = f"{uploaded_file.name.split('.')[0]}_analysis.csv"

        full_csv = results.to_csv(index=False).encode("utf-8")

        summary_data = {
            "Metric": [
                "Total Messages",
                "Spam Messages",
                "Clean Messages",
                "Spam Rate (%)",
                "Participants",
            ],
            "Value": [
                total_msgs,
                spam_msgs,
                ham_msgs,
                spam_rate,
                df["sender"].nunique() if "sender" in df.columns else 1,
            ],
        }
        summary_df = pd.DataFrame(summary_data)

        active_senders_df = active_senders.reset_index()
        active_senders_df.columns = ["Sender", "Message Count"]

        daily_msgs_df = (
            messages_over_time(chat_df).reset_index()
            if daily_msgs is not None and not daily_msgs.empty
            else pd.DataFrame(columns=["Date", "Messages"])
        )
        if not daily_msgs_df.empty:
            daily_msgs_df.columns = ["Date", "Messages"]

        avg_len_df_export = (
            avg_message_length(chat_df).reset_index()
            if avg_len is not None and not avg_len.empty
            else pd.DataFrame(columns=["Sender", "Avg Message Length"])
        )
        if not avg_len_df_export.empty:
            avg_len_df_export.columns = ["Sender", "Avg Message Length"]

        top_words_df_export = pd.DataFrame(
            top_words(chat_df, n=show_count_words), columns=["Word", "Count"]
        )

        emoji_df = (
            pd.DataFrame(emoji_counts, columns=["Emoji", "Count"])
            if emoji_counts
            else pd.DataFrame(columns=["Emoji", "Count"])
        )

        from io import StringIO

        output = StringIO()
        summary_df.to_csv(output, index=False)
        output.write("\n\nActive Senders\n")
        active_senders_df.to_csv(output, index=False)
        output.write("\n\nMessages Over Time\n")
        daily_msgs_df.to_csv(output, index=False)
        output.write("\n\nAverage Message Length\n")
        avg_len_df_export.to_csv(output, index=False)
        output.write("\n\nTop Words\n")
        top_words_df_export.to_csv(output, index=False)
        output.write("\n\nEmoji Usage\n")
        emoji_df.to_csv(output, index=False)

        analysis_csv = output.getvalue().encode("utf-8")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📊 Predictions CSV",
                data=full_csv,
                file_name=full_csv_name,
                mime="text/csv",
            )
        with col2:
            st.download_button(
                label="📈 Analysis CSV",
                data=analysis_csv,
                file_name=analysis_csv_name,
                mime="text/csv",
            )

        st.markdown(
            '<div class="custom-alert alert-success">✅ Analysis completed successfully!</div>',
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"❌ Processing error: {str(e)}")

# Footer
st.markdown(
    """
<div class="footer">
    Built with Streamlit • Powered by Machine Learning
</div>
""",
    unsafe_allow_html=True,
)
