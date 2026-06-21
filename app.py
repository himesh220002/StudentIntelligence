import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import os
import sys
import openai

# Ensure imports work from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from pipelines.data_cleaning_pipeline import DataCleaningPipeline

st.set_page_config(page_title="Student Intelligence System", layout="wide")

col_title, col_button = st.columns([4, 1])

with col_title:
    st.title("🎓 Student Performance Intelligence System")
    st.markdown("An automated AI/ML architecture for education risk-forecasting and engagement analytics.")

# --- View Sample Data Modal ---
@st.dialog("Sample Data Preview (2 Random Rows)")
def show_sample_data(file_path):
    try:
        sample_df = pd.read_csv(file_path)
        if len(sample_df) >= 2:
            sample_df = sample_df.sample(n=2)
        st.dataframe(sample_df, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load sample data: {e}")

with col_button:
    st.write("") # Vertical spacing
    st.write("") 
    if st.button("👀 View Sample Data", use_container_width=True):
        sample_file_path = os.path.join("data/raw", "student_sample_2_students.csv")
        if os.path.exists(sample_file_path):
            show_sample_data(sample_file_path)
        else:
            st.warning(f"Sample file not found: {sample_file_path}")

# --- Data Upload & Execution Flow ---
st.sidebar.header("📁 Data Source")

raw_dir = "data/raw"
proc_dir = "data/processed"
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(proc_dir, exist_ok=True)

source_type = st.sidebar.radio("Select Source Type", ["Upload New File", "Select Existing Raw Data", "Select Existing Processed Data"])

target_filename = None
is_processed = False

if source_type == "Upload New File":
    uploaded_file = st.sidebar.file_uploader("Upload Student Dataset (CSV/Excel)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        file_path = os.path.join(raw_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        target_filename = uploaded_file.name
    else:
        st.sidebar.info("Please upload a file.")

elif source_type == "Select Existing Raw Data":
    raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv') or f.endswith('.xlsx')]
    if raw_files:
        target_filename = st.sidebar.selectbox("Select Raw File", raw_files)
    else:
        st.sidebar.warning("No raw files found.")

elif source_type == "Select Existing Processed Data":
    proc_files = [f for f in os.listdir(proc_dir) if f.endswith('.csv')]
    if proc_files:
        target_filename = st.sidebar.selectbox("Select Processed File", proc_files)
        is_processed = True
    else:
        st.sidebar.warning("No processed files found.")

if target_filename:
    button_label = "⚡ Load Processed Dashboard" if is_processed else "🚀 Run Intelligence Analysis"
    if st.sidebar.button(button_label):
        if is_processed:
            with st.spinner("Loading Processed Data directly..."):
                try:
                    df = pd.read_csv(os.path.join(proc_dir, target_filename))
                    st.session_state['student_results'] = {'smart_ready': df}
                    st.session_state['student_filename'] = target_filename
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading processed file: {e}")
        else:
            with st.spinner("Processing Data through Cleanup Pipeline..."):
                pipeline = DataCleaningPipeline(raw_data_dir=raw_dir, processed_data_dir=proc_dir)
                try:
                    clean_results = pipeline.run_pipeline(
                        filename=target_filename,
                        output_filename=f"clean_{target_filename}"
                    )
                    st.session_state['student_results'] = clean_results
                    st.session_state['student_filename'] = target_filename
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing pipeline: {e}")

if 'student_results' in st.session_state:
    df_smart = st.session_state['student_results']['smart_ready'].copy()
    filename = st.session_state['student_filename']
    
    # --- Feature Engineering ---
    # --- Schema Normalization for Resiliency ---
    col_map = {}
    for c in df_smart.columns:
        c_lower = c.lower().replace(' ', '').replace('_', '')
        if 'attendence' in c_lower or 'attendance' in c_lower: col_map[c] = 'Attendance (%)'
        elif 'unittest1' in c_lower or 'unit1' in c_lower: col_map[c] = 'Unit Test 1'
        elif 'unittest2' in c_lower or 'unit2' in c_lower: col_map[c] = 'Unit Test 2'
        elif 'unittest3' in c_lower or 'unit3' in c_lower: col_map[c] = 'Unit Test 3'
        elif 'halfyrly' in c_lower or 'halfyearly' in c_lower: col_map[c] = 'Half Yearly'
        elif c_lower == 'class': col_map[c] = 'Class'
        elif c_lower == 'section': col_map[c] = 'Section'
        elif c_lower == 'name': col_map[c] = 'Name'
        elif c_lower == 'id': col_map[c] = 'ID'
        elif c_lower == 'subject': col_map[c] = 'Subject'
        elif 'hometution' in c_lower or 'hometuition' in c_lower: col_map[c] = 'Home Tuition (Y/N)'
    df_smart = df_smart.rename(columns=col_map)
    
    # Standardize ID to be unique per Student, not per Row
    if all(c in df_smart.columns for c in ['Class', 'Section', 'Name', 'ID']):
        if df_smart.groupby(['Class', 'Section', 'Name'])['ID'].nunique().max() > 1:
            group_id = df_smart.groupby(['Class', 'Section', 'Name']).ngroup() + 1
            df_smart['ID'] = df_smart['Class'].astype(str) + "_" + df_smart['Section'].astype(str) + "_" + group_id.astype(str)

    # Engagement Index (Focus + Homework + Q&A)
    # Robustly find column names (handles non-breaking hyphens)
    focus_cols = [c for c in df_smart.columns if 'Focus' in c]
    hw_cols = [c for c in df_smart.columns if 'Homework' in c]
    qa_cols = [c for c in df_smart.columns if 'Q&A' in c]
    
    if focus_cols and hw_cols and qa_cols:
        df_smart['Engagement Index'] = ((df_smart[focus_cols[0]] + df_smart[hw_cols[0]] + df_smart[qa_cols[0]]) / 30) * 100
    else:
        df_smart['Engagement Index'] = 0.0
        
    # Calculate Avg Test Score
    ut_cols = [c for c in df_smart.columns if 'Unit Test' in c]
    if len(ut_cols) > 0:
        df_smart['Avg Test Score'] = df_smart[ut_cols].mean(axis=1)
    else:
        df_smart['Avg Test Score'] = 0.0
        
    if all(c in df_smart.columns for c in ['Class', 'Section', 'Subject']):
        df_smart['Rank'] = df_smart.groupby(['Class', 'Section', 'Subject'])['Avg Test Score'].rank(ascending=False, method='min').astype(int)
    else:
        df_smart['Rank'] = df_smart['Avg Test Score'].rank(ascending=False, method='min').astype(int)

    def get_intervention(row):
        avg_score = row.get('Avg Test Score', 100)
        ut3_score = row.get('Unit Test 3', 40)
        engagement = row.get('Engagement Index', 100)
        risk_factors = []
        if pd.notnull(avg_score) and avg_score < 50: risk_factors.append("Low Avg Score")
        if pd.notnull(ut3_score) and ut3_score < 16: risk_factors.append("Low UT3")
        if pd.notnull(engagement) and engagement < 40: risk_factors.append("Low Engagement")
        if not risk_factors: return "On Track"
        elif len(risk_factors) >= 2: return "Critical Risk: Parent-Teacher Meeting & Core Remedial"
        elif "Low Engagement" in risk_factors: return "Behavioral Risk: Immediate Focus & Motivation Intervention"
        elif "Low UT3" in risk_factors: return "Recent Drop: Extra Doubt Sessions for UT3 Concepts"
        else: return "Academic Risk: Assign Extra Practice & Monitor"

    df_smart['Intervention Recommendation'] = df_smart.apply(get_intervention, axis=1)

    # --- Interactive Filtering ---
    st.sidebar.header("🎯 Dashboard Filters")
    
    if 'Class' in df_smart.columns:
        classes = sorted(df_smart['Class'].dropna().unique().tolist())
        selected_classes = st.sidebar.multiselect("Select Class", options=classes, default=classes)
    else:
        selected_classes = []
        
    if 'Section' in df_smart.columns:
        sections = sorted(df_smart['Section'].dropna().unique().tolist())
        selected_sections = st.sidebar.multiselect("Select Section", options=sections, default=sections)
    else:
        selected_sections = []
        
    if 'Subject' in df_smart.columns:
        subjects = sorted(df_smart['Subject'].dropna().unique().tolist())
        selected_subjects = st.sidebar.multiselect("Select Subject", options=subjects, default=subjects)
    else:
        selected_subjects = []

    # Apply Filters
    df_filtered = df_smart.copy()
    if selected_classes and 'Class' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Class'].isin(selected_classes)]
    if selected_sections and 'Section' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Section'].isin(selected_sections)]
    if selected_subjects and 'Subject' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Subject'].isin(selected_subjects)]
        
    if df_filtered.empty:
        st.warning("No students match the selected filters.")
    else:
        # --- Insights Construction ---
        st.header("📊 Executive Dashboard")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Filtered Students", len(df_filtered))
        
        avg_att = df_filtered['Attendance (%)'].mean() if 'Attendance (%)' in df_filtered.columns else 0
        col2.metric("Avg Attendance", f"{avg_att:.1f}%")
        
        avg_eng = df_filtered['Engagement Index'].mean()
        col3.metric("Avg Engagement", f"{avg_eng:.1f}/100")
        
        avg_score = df_filtered['Avg Test Score'].mean() if 'Avg Test Score' in df_filtered.columns else 0
        col4.metric("Avg Test Score", f"{avg_score:.1f}%")
        
        if 'Avg Test Score' in df_filtered.columns and not df_filtered.empty:
            mean_score = df_filtered['Avg Test Score'].mean()
            median_score = df_filtered['Avg Test Score'].median()
            mode_scores = df_filtered['Avg Test Score'].mode()
            mode_score = mode_scores.iloc[0] if not mode_scores.empty else 0
            st.caption(f"**Overall Test Score Stats:** Mean: {mean_score:.1f}% | Median: {median_score:.1f}% | Mode: {mode_score:.1f}%")

        st.divider()

        st.subheader("🔍 At-Risk Predictor & Recommendations")
        if 'Intervention Recommendation' in df_filtered.columns:
            at_risk = df_filtered[df_filtered['Intervention Recommendation'] != 'On Track'].copy()
            if not at_risk.empty:
                avail_cols = [c for c in ['ID', 'Name', 'Class', 'Section', 'Subject', 'Avg Test Score', 'Unit Test 3', 'Engagement Index', 'Intervention Recommendation'] if c in at_risk.columns]
                st.dataframe(at_risk[avail_cols], use_container_width=True)
            else:
                st.success("No students are currently marked as at-risk in this filtered view.")
        else:
            st.info("Required columns for Risk Predictor not found.")

        st.divider()

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.subheader("📈 Engagement vs Avg Score Correlation")
            if 'Engagement Index' in df_filtered.columns and 'Avg Test Score' in df_filtered.columns and 'Section' in df_filtered.columns:
                scatter_chart = alt.Chart(df_filtered).mark_circle(size=80).encode(
                    x=alt.X('Engagement Index:Q', scale=alt.Scale(domain=[0, 100]), title='Engagement Index (out of 100)'),
                    y=alt.Y('Avg Test Score:Q', scale=alt.Scale(domain=[0, 100]), title='Average Test Score (%)'),
                    color='Section:N',
                    tooltip=[c for c in ['Name', 'Engagement Index', 'Avg Test Score', 'Section'] if c in df_filtered.columns]
                ).interactive()
                st.altair_chart(scatter_chart, use_container_width=True)
            else:
                st.info("Chart data missing.")

        with col_chart2:
            st.subheader("🔥 Average Engagement by Section")
            if 'Section' in df_filtered.columns:
                bar_chart = alt.Chart(df_filtered).mark_bar().encode(
                    x='Section:N',
                    y='mean(Engagement Index):Q',
                    color='Section:N',
                    tooltip=['Section', 'mean(Engagement Index)']
                )
                st.altair_chart(bar_chart, use_container_width=True)
            else:
                st.info("Chart data missing.")

        st.divider()

        st.subheader("✨ Filtered Engagement Dataset")
        st.markdown("Download the filtered subset along with the calculated Engagement Index and Intervention tracking.")
        
        # Add Intervention Recommendation to main dataframe for download
        
        # Custom sorting: Class DESC, Section ASC, Roll Number ASC, Subject Custom, Name ASC
        if all(col in df_filtered.columns for col in ['Class', 'Section', 'Subject', 'Name', 'Roll Number']):
            # Create a categorical type for Subject to enforce custom order
            subject_order = ['Maths', 'Science', 'English', 'Language', 'Social', 'Arts', 'Physical']
            
            # Identify any existing subjects not in the predefined list and append them
            existing_subjects = df_filtered['Subject'].dropna().unique().tolist()
            for subj in existing_subjects:
                if subj not in subject_order:
                    subject_order.append(subj)

            df_filtered['Subject_Cat'] = pd.Categorical(df_filtered['Subject'], categories=subject_order, ordered=True)
            
            df_filtered = df_filtered.sort_values(
                by=['Class', 'Section', 'Roll Number', 'Subject_Cat', 'Name'],
                ascending=[False, True, True, True, True]
            )
            df_filtered = df_filtered.drop(columns=['Subject_Cat'])
        elif all(col in df_filtered.columns for col in ['Class', 'Section', 'Subject', 'Name']):
            subject_order = ['Maths', 'Science', 'English', 'Language', 'Social', 'Arts', 'Physical']
            existing_subjects = df_filtered['Subject'].dropna().unique().tolist()
            for subj in existing_subjects:
                if subj not in subject_order:
                    subject_order.append(subj)
            df_filtered['Subject_Cat'] = pd.Categorical(df_filtered['Subject'], categories=subject_order, ordered=True)
            df_filtered = df_filtered.sort_values(
                by=['Class', 'Section', 'Subject_Cat', 'Name'],
                ascending=[False, True, True, True]
            )
            df_filtered = df_filtered.drop(columns=['Subject_Cat'])

        df_download = df_filtered.copy().reset_index(drop=True)
        df_download.index = df_download.index + 1
        if 'Attendance (%)' in df_download.columns and 'Unit Test 3' in df_download.columns:
             df_download['Intervention'] = df_download.apply(
                lambda row: ("Schedule Extra Doubt Session" if row.get('Doubt Asking Rate', 1) < 0.4 else "Focus Monitoring") 
                if (row['Attendance (%)'] < 75 or row['Unit Test 3'] < 70) else "On Track", axis=1
             )
        
        st.markdown("**Select a student row below to view their detailed performance radar (Circular Potential Zones):**")
        
        # Add Search Bar for Name or Roll Number
        search_query = st.text_input("🔍 Search Student by Name or Roll Number:")
        
        df_display = df_download.copy()
        if search_query:
            search_query = search_query.lower()
            if 'Roll Number' in df_display.columns:
                df_display = df_display[
                    df_display['Name'].str.lower().str.contains(search_query, na=False) | 
                    df_display['Roll Number'].astype(str).str.contains(search_query, na=False)
                ]
            else:
                df_display = df_display[df_display['Name'].str.lower().str.contains(search_query, na=False)]
        
        event = st.dataframe(
            df_display, 
            use_container_width=True, 
            selection_mode="single-row", 
            on_select="rerun", 
            key="student_selection"
        )
        
        selected_rows = getattr(event, 'selection', event).rows if hasattr(getattr(event, 'selection', event), 'rows') else event.selection.rows # type: ignore
        if selected_rows:
            selected_idx = selected_rows[0]
            if selected_idx < len(df_display):
                student_data = df_display.iloc[selected_idx]
                
                st.markdown("---")
                st.subheader(f"🎯 Detailed View: {student_data.get('Name', 'Unknown')} ({student_data.get('Subject', 'Overall')})")
                
                # Prepare data for Radar Chart
                
                def get_col_val(keyword, default=0):
                    cols = [c for c in student_data.index if keyword in c]
                    return student_data[cols[0]] if cols else default
                    
                # Scale metrics out of 100
                metrics = {
                    'Attendance': get_col_val('Attendance'),
                    'Avg Test Score': student_data.get('Avg Test Score', 0),
                    'Focus (Scaled)': get_col_val('Focus') * 10,
                    'Homework (Scaled)': get_col_val('Homework') * 10,
                    'Q&A (Scaled)': get_col_val('Q&A') * 10,
                    'Exam Prep (Scaled)': get_col_val('Exam Prep') * 10,
                    'Special Problems': get_col_val('Special Problems Completion')
                }
                
                # Filter out any metrics that don't exist in the df
                valid_metrics = {k: v for k, v in metrics.items() if pd.notnull(v)}
                
                if valid_metrics:
                    radar_df = pd.DataFrame(dict(
                        r=list(valid_metrics.values()),
                        theta=list(valid_metrics.keys())
                    ))
                    
                    fig = px.line_polar(radar_df, r='r', theta='theta', line_close=True, range_r=[0,100], 
                                        color_discrete_sequence=['#00f2fe'])
                    fig.update_traces(fill='toself', fillcolor='rgba(0, 242, 254, 0.2)')
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 100])
                        ),
                        showlegend=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    
                    # 2nd Graph: Cross-Subject Scoring
                    all_subjects_df = df_smart[df_smart['Name'] == student_data.get('Name', '')].copy()
                    fig_subj = None
                    
                    if not all_subjects_df.empty and 'Subject' in all_subjects_df.columns and 'Avg Test Score' in all_subjects_df.columns:
                        subj_scores = all_subjects_df.groupby('Subject')['Avg Test Score'].mean().reset_index()
                        
                        fig_subj = px.line_polar(subj_scores, r='Avg Test Score', theta='Subject', line_close=True, range_r=[0,100],
                                                 color_discrete_sequence=['#ff007f'])
                        fig_subj.update_traces(fill='toself', fillcolor='rgba(255, 0, 127, 0.2)')
                        fig_subj.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 100])
                            ),
                            showlegend=False,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                    
                    col_r1, col_r2, col_r3 = st.columns([1.5, 1.5, 1])
                    with col_r1:
                        st.markdown("**Personal Potential Zones**")
                        st.plotly_chart(fig, use_container_width=True)
                    with col_r2:
                        st.markdown("**Cross-Subject Scores**")
                        if fig_subj:
                            st.plotly_chart(fig_subj, use_container_width=True)
                        else:
                            st.info("No cross-subject data available.")
                    with col_r3:
                        st.write("### Quick Stats")
                        st.write(f"**Class:** {student_data.get('Class', 'N/A')}")
                        st.write(f"**Section:** {student_data.get('Section', 'N/A')}")
                        st.write(f"**Subject:** {student_data.get('Subject', 'N/A')}")
                        st.write(f"**Rank:** {student_data.get('Rank', 'N/A')}")
                        st.write(f"**Engagement Index:** {student_data.get('Engagement Index', 0):.1f}/100")
                        st.info(student_data.get('Intervention Recommendation', student_data.get('Intervention', 'On Track')))
                    
                    st.markdown("---")
                    if st.button("✨ Generate AI Action Plan"):
                        api_key = os.getenv("NVIDIA_API_KEY")
                        if not api_key and hasattr(st, "secrets") and "nvidia_api_key" in st.secrets: api_key = st.secrets["nvidia_api_key"]
                        if not api_key: st.error("NVIDIA API Key is missing. Please set NVIDIA_API_KEY in your environment or Streamlit secrets.")
                        else:
                            with st.spinner("Generating personalized AI Action Plan..."):
                                try:
                                    client = openai.OpenAI(api_key=api_key, base_url="https://integrate.api.nvidia.com/v1")
                                    prompt = f"""You are an expert educational AI assistant.
Generate a structured, organized action plan for the student {student_data.get('Name')} (Class {student_data.get('Class')}).
Here are their current metrics:
Attendance: {student_data.get('Attendance (%)', 'N/A')}%
Avg Test Score: {student_data.get('Avg Test Score', 'N/A')}%
Engagement Index: {student_data.get('Engagement Index', 0)}/100
Intervention Status: {student_data.get('Intervention Recommendation', 'On Track')}

Respond EXACTLY in this format (use emojis and bold text):
🏷️ **Student Profile**
Current Status: [Identify them briefly based on metrics]
🌟 **Academic Standing**: [Brief comment on scores]

💬 **Home Actions (For Parents)**
- [Action 1: specific to their metrics]
- [Action 2: specific to their metrics]
- [Action 3: specific to their metrics]

🏫 **School Actions (With Teachers)**
- [Action 1: specific to their metrics]
- [Action 2: specific to their metrics]
- [Action 3: specific to their metrics]"""
                                    response = client.chat.completions.create(
                                        model="meta/llama-3.3-70b-instruct",
                                        messages=[{"role": "user", "content": prompt}],
                                        temperature=0.7, max_tokens=600
                                    )
                                    st.success("Plan Generated Successfully!")
                                    st.markdown(response.choices[0].message.content)
                                except Exception as e: st.error(f"Failed to generate AI plan: {e}")
                else:
                    st.warning("Insufficient data to plot radar chart.")
        
        csv_data = df_download.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Engagement Dataset",
            data=csv_data,
            file_name=f"filtered_engagement_{filename}",
            mime="text/csv",
        )
        
        if st.sidebar.button("🧹 Clear Results"):
            st.session_state.pop('student_results', None)
            st.session_state.pop('student_filename', None)
            st.rerun()
