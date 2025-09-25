import os
import asyncio
from openai import AsyncAzureOpenAI

# Optional agents framework (enhanced tool calling) ---------------------------------
try:  # attempt to import advanced agent framework used in app1.py
    from agents import (
        set_default_openai_client,
        set_tracing_disabled,
        OpenAIChatCompletionsModel,
        Agent,
        Runner,
        function_tool
    )
    HAS_AGENTS = True
except Exception:
    HAS_AGENTS = False
    # Provide no-op decorator so existing functions can be conditionally decorated
    def function_tool(fn):
        return fn

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
from dotenv import load_dotenv
import httpx

# Cleaner agent (optional)
try:
    from cleaner_agent import run_data_cleaning as _run_data_cleaning
except Exception:
    _run_data_cleaning = None

# Predictive analyst (optional risk scoring)
try:
    from predictive_analyst import generate_machine_risk as _generate_machine_risk, save_outputs as _save_risk_outputs
except Exception:
    _generate_machine_risk = None
    _save_risk_outputs = None

load_dotenv()
http_client = httpx.AsyncClient(verify=False)
warnings.filterwarnings('ignore')

# Page config + light styling (from legacy app1) ------------------------------------
try:
    st.set_page_config(
        page_title="Equipment Analytics Dashboard",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception:
    pass  # ignore if already set

st.markdown(
    """
    <style>
    .metric-card {background-color:#f0f2f6;padding:1rem;border-radius:0.5rem;border-left:5px solid #1f77b4;}
    .stTabs [data-baseweb="tab-list"] {gap:1.5rem;}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------------------------------------
# LLM Utilities (direct call, no external agent framework)
# ----------------------------------------------------------------------------------

async def _get_client():
    """Get (and cache) a raw AsyncAzureOpenAI client used for both direct calls and agent model wrapper."""
    if 'azure_client' not in st.session_state:
        st.session_state.azure_client = AsyncAzureOpenAI(
            azure_endpoint=os.getenv("AOAI_ENDPOINT"),
            api_key=os.getenv("AOAI_KEY"),
            api_version=os.getenv("AOAI_API_VERSION"),
            http_client=http_client
        )
    return st.session_state.azure_client

def _get_or_create_agent():
    """Create and cache an Agent instance if the optional agents framework is available."""
    if not HAS_AGENTS:
        return None
    if 'equipment_agent' in st.session_state:
        return st.session_state.equipment_agent
    openai_client = asyncio.run(_get_client())  # ensure client exists (synchronous context)
    try:
        set_default_openai_client(openai_client)
        set_tracing_disabled(True)
        chat_model = OpenAIChatCompletionsModel(
            model=os.getenv("MODEL"),
            openai_client=openai_client
        )
        agent = Agent(
            name="EquipmentAnalysisAgent",
            instructions=(
                "You are a friendly and knowledgeable Equipment Maintenance Assistant for Volvo Group machinery. "
                "Use provided tools for machine / fault / predictive analysis. Keep responses concise, engaging, and actionable."
            ),
            model=chat_model,
            tools=[get_machine_health_summary, analyze_fault_patterns, get_predictive_insights]
        )
        st.session_state.equipment_agent = agent
        return agent
    except Exception:
        return None

async def call_llm(user_query: str, tool_context: str = "") -> str:
    """Direct lightweight LLM call (fallback when agents framework unavailable)."""
    client = await _get_client()
    system_prompt = (
        "You are an Equipment Maintenance Assistant. Use provided TOOL_CONTEXT if present. "
        "When given structured context, integrate it faithfully before adding new reasoning."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    if tool_context:
        messages.append({"role": "system", "content": f"TOOL_CONTEXT:\n{tool_context}"})
    try:
        resp = await client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=messages
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"LLM error: {e}"

# ----------------------------------------------------------------------------------
# Existing cached data loader (will be cleared after cleaner run)
# ----------------------------------------------------------------------------------

@st.cache_data
def load_data():
    try:
        claims_df = pd.read_csv("Data/ActiveCare Current Claims.csv")
        historical_df = pd.read_csv("Data/ActiveCare Historical.csv")
        tech_support_df = pd.read_csv("Data/TechSupport Data.csv")
        matris_df = pd.read_csv("Data/Matris Log Data.csv")
        claims_df['requested_date'] = pd.to_datetime(claims_df['requested_date'], errors='coerce')
        historical_df['requested_date'] = pd.to_datetime(historical_df['requested_date'], errors='coerce')
        tech_support_df['requested_date'] = pd.to_datetime(tech_support_df['requested_date'], errors='coerce')
        if 'Timestamp' in matris_df.columns:
            matris_df['Timestamp'] = pd.to_datetime(matris_df['Timestamp'], errors='coerce')
        return claims_df, historical_df, tech_support_df, matris_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return (pd.DataFrame(),)*4

@st.cache_data
def load_risk_scores():
    """Attempt to load risk scores from processed directory.
    Looks in probable relative locations: '../processed' and 'processed'."""
    candidates = [
        'processed/machine_risk_scores.parquet',
        'processed/machine_risk_scores.csv',
        '../processed/machine_risk_scores.parquet',
        '../processed/machine_risk_scores.csv'
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                if path.endswith('.parquet'):
                    return pd.read_parquet(path), path
                else:
                    return pd.read_csv(path), path
            except Exception:
                continue
    return pd.DataFrame(), None

# ---------------- Helper analytic functions (formerly function tools) ---------------- #

@function_tool
def get_machine_health_summary(chassis_id: str) -> str:
    claims_df, historical_df, tech_support_df, matris_df = load_data()
    if claims_df.empty:
        return "No data loaded."
    summary = [f"Health Summary for Machine {chassis_id}:"]
    current_claims = claims_df[claims_df['chassis_id'] == chassis_id]
    if not current_claims.empty:
        summary.append(f"Current Claims: {len(current_claims)} | Most recent fault: {current_claims.iloc[-1]['fault_code']}")
    historical_claims = historical_df[historical_df['chassis_id'] == chassis_id]
    if not historical_claims.empty:
        summary.append(f"Historical Claims: {len(historical_claims)}")
    tech_cases = pd.DataFrame()
    if not current_claims.empty and 'machine_salesmodel' in current_claims.columns:
        tech_cases = tech_support_df[tech_support_df['machine_salesmodel'].isin(current_claims['machine_salesmodel'].unique())]
    if not tech_cases.empty:
        summary.append(f"Related Tech Support Cases: {len(tech_cases)}")
    telemetry = matris_df[matris_df.get('ChassisId','') == chassis_id] if 'ChassisId' in matris_df.columns else pd.DataFrame()
    if not telemetry.empty and 'MachineHours' in telemetry.columns:
        summary.append(f"Telemetry Records: {len(telemetry)} | Latest Machine Hours: {telemetry['MachineHours'].max()}")
    return "\n".join(summary)

@function_tool
def analyze_fault_patterns(fault_code: str) -> str:
    claims_df, historical_df, tech_support_df, _ = load_data()
    if claims_df.empty:
        return "No data loaded."
    lines = [f"Analysis for Fault Code {fault_code}:"]
    current_fault = claims_df[claims_df['fault_code'] == fault_code]
    if not current_fault.empty:
        lines.append(f"Current Claims: {len(current_fault)} | Models: {', '.join(current_fault['machine_salesmodel'].unique())}")
    historical_fault = historical_df[historical_df['fault_code'] == fault_code]
    if not historical_fault.empty:
        lines.append(f"Historical Claims: {len(historical_fault)}")
    if 'fault_codes' in tech_support_df.columns:
        tech_solutions = tech_support_df[tech_support_df['fault_codes'].astype(str).str.contains(fault_code, na=False)]
        if not tech_solutions.empty:
            lines.append(f"Tech Support Cases Referencing Code: {len(tech_solutions)}")
            for _, case in tech_solutions.head(3).iterrows():
                corr = case.get('correction')
                if isinstance(corr, str) and corr:
                    lines.append(f"- Solution: {corr[:120]}")
    return "\n".join(lines)

@function_tool
def get_predictive_insights(chassis_id: str) -> str:
    claims_df, _, _, matris_df = load_data()
    telemetry = matris_df[matris_df['ChassisId'] == chassis_id] if 'ChassisId' in matris_df.columns else pd.DataFrame()
    if telemetry.empty:
        return f"No telemetry data for {chassis_id}" 
    lines = [f"Predictive Insights for {chassis_id}:"]
    if 'MeasureName' in telemetry.columns and 'MeasureValue' in telemetry.columns:
        brake = telemetry[telemetry['MeasureName'].str.contains('Brake', case=False, na=False)]
        temp = telemetry[telemetry['MeasureName'].str.contains('temp', case=False, na=False)]
        if not brake.empty:
            try:
                avg_brake = pd.to_numeric(brake['MeasureValue'], errors='coerce').mean()
                lines.append(f"Avg Brake Metric: {avg_brake:.2f}")
            except Exception:
                pass
        if not temp.empty:
            try:
                max_temp = pd.to_numeric(temp['MeasureValue'], errors='coerce').max()
                lines.append(f"Max Temp: {max_temp:.1f}")
                if max_temp and max_temp > 100:
                    lines.append("High temperature anomaly detected")
            except Exception:
                pass
    if 'chassis_id' in claims_df.columns:
        machine_model = claims_df[claims_df['chassis_id'] == chassis_id]['machine_salesmodel'].iloc[0] if not claims_df[claims_df['chassis_id'] == chassis_id].empty else None
        if machine_model:
            similar = claims_df[claims_df['machine_salesmodel'] == machine_model]
            top_faults = similar['fault_code'].value_counts().head(3)
            if not top_faults.empty:
                lines.append(f"Common faults for {machine_model}:")
                for f, c in top_faults.items():
                    lines.append(f"- {f}: {c}")
    return "\n".join(lines)

# --- Risk helper functions for chatbot ---

def get_risk_info(chassis_id: str) -> str:
    risk_df, _ = load_risk_scores()
    if risk_df.empty or 'chassis_id' not in risk_df.columns:
        return "No risk scores available. Generate them in the Predictive Analysis tab."
    row = risk_df[risk_df['chassis_id'].astype(str) == str(chassis_id)]
    if row.empty:
        return f"No risk score found for chassis {chassis_id}."
    r = row.iloc[0]
    parts = [f"Risk Summary for {chassis_id}"]
    parts.append(f"Risk Score: {r.get('risk_score','n/a')}")
    parts.append(f"Explanation: {r.get('risk_explanation','n/a')}")
    for col,label in [
        ('claim_freq_z','Claim Frequency Z'),
        ('open_ratio','Open Ratio'),
        ('unique_faults_norm','Fault Diversity'),
        ('recent_spike_flag','Recent Spike Flag'),
        ('severity_signal','Severity Signal')
    ]:
        if col in r and pd.notna(r[col]):
            parts.append(f"{label}: {r[col]}")
    return "\n".join(parts)

def get_top_risks(n: int = 10, min_score: int = 0) -> str:
    risk_df, _ = load_risk_scores()
    if risk_df.empty:
        return "No risk scores available to list."
    rdf = risk_df.sort_values('risk_score', ascending=False)
    if min_score:
        rdf = rdf[rdf['risk_score'] >= min_score]
    rdf = rdf.head(n)
    lines = [f"Top {len(rdf)} Risky Machines (threshold {min_score})"]
    for _, row in rdf.iterrows():
        lines.append(f"{row['chassis_id']}: {row['risk_score']} | {row.get('risk_explanation','')}")
    return "\n".join(lines)

# ---------------- Chatbot (direct tool routing) ---------------- #

def chatbot_ui():
    """Unified chatbot: uses agent framework if available; otherwise manual routing with direct LLM calls.
    Also augments queries with risk score helper outputs when requested."""
    st.header("🤖 Equipment Maintenance Assistant")
    with st.expander("How to ask questions", expanded=False):
        st.markdown(
            """
            Ask about:
            - Machine health: `analyze machine <CHASSIS_ID>` / `health <CHASSIS_ID>`
            - Fault patterns: `fault <FAULT_CODE>` / `analyze fault <FAULT_CODE>`
            - Predictive telemetry insight: `predict <CHASSIS_ID>`
            - Risk score for a machine: `risk <CHASSIS_ID>`
            - Top risky machines: `top 5 risks` / `high risk machines`
            """
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [("Bot", "Hi! 👋 I'm your Equipment Assistant. Ask about a chassis, a fault code, or predictive risks.")]

    # Display prior (chat_message style if available)
    for sender, msg in st.session_state.chat_history:
        if hasattr(st, "chat_message"):
            role = "user" if sender == "You" else "assistant"
            with st.chat_message(role):
                st.markdown(msg)
        else:
            st.markdown(f"**{sender}:** {msg}")

    user_input = st.text_input("Your question")
    send_col1, send_col2 = st.columns([1,0.3])
    with send_col1:
        send_clicked = st.button("Send", type="primary")
    with send_col2:
        if st.button("Clear"):
            st.session_state.chat_history = [("Bot", "Conversation cleared. How can I help now?")]
            st.rerun()

    if not send_clicked or not user_input:
        return

    qlower = user_input.lower().strip()
    tool_output = ""

    # Manual lightweight routing for both modes to supply context
    if any(qlower.startswith(prefix) for prefix in ["analyze machine", "health "]):
        chassis = user_input.split()[-1]
        tool_output = get_machine_health_summary(chassis)
    elif qlower.startswith("predict") or "predictive" in qlower:
        chassis = user_input.split()[-1]
        tool_output = get_predictive_insights(chassis)
    elif qlower.startswith("risk ") or qlower.startswith("risk for") or qlower.startswith("risk score"):
        tokens = user_input.replace(':',' ').split()
        chassis = tokens[-1]
        tool_output = get_risk_info(chassis)
    elif "top" in qlower and "risk" in qlower:
        import re
        m = re.search(r"top\s+(\d+)", qlower)
        n = int(m.group(1)) if m else 10
        tool_output = get_top_risks(n=n)
    elif ("high" in qlower and "risk" in qlower) or ("risk" in qlower and "machines" in qlower):
        tool_output = get_top_risks(n=10, min_score=70)
    elif qlower.startswith("fault ") or qlower.startswith("analyze fault") or "fault code" in qlower:
        token = user_input.split()[-1]
        tool_output = analyze_fault_patterns(token)

    st.session_state.chat_history.append(("You", user_input))

    if HAS_AGENTS:
        agent = _get_or_create_agent()
        if agent:
            composite_query = user_input if not tool_output else f"{user_input}\n\nContext:\n{tool_output}"
            with st.spinner('Processing...'):
                try:
                    async def _run():
                        try:
                            result = await Runner.run(agent, composite_query)
                            return getattr(result, 'final_output', str(result))
                        except Exception as e:  # fallback to direct LLM
                            return f"Agent error: {e}. Fallback answer using provided context:\n{tool_output}" if tool_output else f"Agent error: {e}"
                    answer = asyncio.run(_run())
                except Exception as e:
                    answer = f"Execution error: {e}"
            st.session_state.chat_history.append(("Bot", answer))
            st.rerun()
            return

    # Fallback direct model path
    with st.spinner('Processing...'):
        try:
            answer = asyncio.run(call_llm(user_input, tool_output))
        except Exception as e:
            answer = f"LLM error: {e}\n{tool_output if tool_output else ''}".strip()
    st.session_state.chat_history.append(("Bot", answer))
    st.rerun()

# ---------------- Sidebar Update (add clean button + cache clear) ---------------- #

def sidebar_controls():
    st.sidebar.header("🎛️ Controls & Data")
    if _run_data_cleaning:
        if st.sidebar.button("🧹 Clean / Refresh Data", use_container_width=True):
            with st.spinner("Running data cleaning pipeline..."):
                result = _run_data_cleaning()
                st.session_state.last_clean_result = result
                load_data.clear()  # clear cached data
                if result.get("ok"):
                    st.sidebar.success("Data cleaned and cache cleared")
                else:
                    st.sidebar.error("Cleaning failed")
                    st.sidebar.caption(result.get("message"))
    else:
        st.sidebar.info("Cleaner agent not available")
    if 'last_clean_result' in st.session_state:
        with st.sidebar.expander("Last Clean Run"):
            lr = st.session_state.last_clean_result
            st.write(lr.get('message'))

# ------------------- Main Dashboard Layout ------------------- #

def main():
    st.title("🔧 Equipment Analytics Dashboard")
    st.markdown("### Interactive Analysis of Equipment Claims, Support Cases, and Telemetry Data")
    
    # Load data
    claims_df, historical_df, tech_support_df, matris_df = load_data()
    
    if claims_df is None:
        st.error("Unable to load data files. Please ensure all CSV files are in the Data/ directory.")
        return
    
    # Sidebar branding + filters (use logo image instead of text)
    try:
        st.sidebar.image("volvo.png", use_container_width=True)
    except Exception:
        # Fallback text if image missing
        st.sidebar.markdown("<h1 style='margin-top:0;font-size:2rem;line-height:1'>VOLVO</h1>", unsafe_allow_html=True)
    st.sidebar.header("🎛️ Filters")
    
    # Date range filter
    if not claims_df.empty and 'requested_date' in claims_df.columns:
        min_date = claims_df['requested_date'].min().date()
        max_date = claims_df['requested_date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    
    # Machine model filter
    if 'machine_salesmodel' in claims_df.columns:
        machine_models = ['All'] + sorted(claims_df['machine_salesmodel'].dropna().unique().tolist())
        selected_model = st.sidebar.selectbox("Machine Model", machine_models)
    
    # Data clean button
    if _run_data_cleaning:
        if st.sidebar.button("🧹 Clean / Refresh Data", use_container_width=True):
            with st.spinner("Running data cleaning pipeline..."):
                result = _run_data_cleaning()
                st.session_state.last_clean_result = result
                if result.get("ok"):
                    st.sidebar.success("Data cleaned ✅")
                    load_data.clear()  # Clear cache after successful cleaning
                else:
                    st.sidebar.error("Cleaning failed")
                    st.sidebar.caption(result.get("message"))
    else:
        st.sidebar.info("Cleaner agent not available")

    if 'last_clean_result' in st.session_state:
        with st.sidebar.expander("Last Clean Run"):
            lr = st.session_state.last_clean_result
            st.write(lr.get('message'))

    # --- Chatbot Button ---
    if st.sidebar.button("🤖 Open Chatbot", use_container_width=True):
        st.session_state.show_chatbot = True

    # --- Main Content Switch ---
    if st.session_state.get("show_chatbot", False):
        if st.button("⬅️ Back to Dashboard"):
            st.session_state.show_chatbot = False
            st.rerun()
        else:
            chatbot_ui()
        return
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🚨 Claims Analysis", "🛠️ Tech Support", "📈 Equipment Telemetry", "🔮 Predictive Analysis"])
    
    with tab1:
        st.header("📊 Dashboard Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_claims = len(claims_df)
            st.metric("Total Claims", f"{total_claims:,}")
        
        with col2:
            total_support_cases = len(tech_support_df)
            st.metric("Support Cases", f"{total_support_cases:,}")
        
        with col3:
            unique_machines = claims_df['chassis_id'].nunique()
            st.metric("Unique Machines", f"{unique_machines:,}")
        
        with col4:
            if 'state' in claims_df.columns:
                closed_claims = len(claims_df[claims_df['state'] == 'Closed'])
                closure_rate = (closed_claims / total_claims * 100) if total_claims > 0 else 0
                st.metric("Closure Rate", f"{closure_rate:.1f}%")
        
        # Overview charts
        col1, col2 = st.columns(2)
        
        with col1:
            if 'requested_date' in claims_df.columns:
                # Claims over time
                claims_monthly = claims_df.groupby(claims_df['requested_date'].dt.to_period('M')).size()
                fig_timeline = px.line(
                    x=claims_monthly.index.astype(str), 
                    y=claims_monthly.values,
                    title="Claims Timeline",
                    labels={'x': 'Month', 'y': 'Number of Claims'}
                )
                fig_timeline.update_layout(height=400)
                st.plotly_chart(fig_timeline, use_container_width=True)
        
        with col2:
            if 'machine_salesmodel' in claims_df.columns:
                # Top machine models by claims
                model_counts = claims_df['machine_salesmodel'].value_counts().head(10)
                fig_models = px.bar(
                    x=model_counts.index,
                    y=model_counts.values,
                    title="Top Machine Models by Claims",
                    labels={'x': 'Machine Model', 'y': 'Number of Claims'}
                )
                fig_models.update_layout(height=400)
                st.plotly_chart(fig_models, use_container_width=True)
    
    with tab2:
        st.header("🚨 Claims Analysis")
        
        # Filter data based on selections
        filtered_claims = claims_df.copy()
        if len(date_range) == 2:
            filtered_claims = filtered_claims[
                (filtered_claims['requested_date'].dt.date >= date_range[0]) &
                (filtered_claims['requested_date'].dt.date <= date_range[1])
            ]
        if selected_model != 'All':
            filtered_claims = filtered_claims[filtered_claims['machine_salesmodel'] == selected_model]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Fault code analysis
            if 'fault_code' in filtered_claims.columns:
                fault_counts = filtered_claims['fault_code'].value_counts().head(15)
                fig_faults = px.bar(
                    x=fault_counts.values,
                    y=fault_counts.index,
                    orientation='h',
                    title="Most Common Fault Codes",
                    labels={'x': 'Frequency', 'y': 'Fault Code'}
                )
                fig_faults.update_layout(height=500)
                st.plotly_chart(fig_faults, use_container_width=True)
        
        with col2:
            # State distribution
            if 'state' in filtered_claims.columns:
                state_counts = filtered_claims['state'].value_counts()
                fig_states = px.pie(
                    values=state_counts.values,
                    names=state_counts.index,
                    title="Claims Status Distribution"
                )
                fig_states.update_layout(height=500)
                st.plotly_chart(fig_states, use_container_width=True)
        
        # Machine hours analysis
        if 'machine_hours' in filtered_claims.columns:
            st.subheader("Machine Hours Analysis")
            
            # Remove non-numeric values and convert to numeric
            filtered_claims['machine_hours_numeric'] = pd.to_numeric(filtered_claims['machine_hours'], errors='coerce')
            valid_hours = filtered_claims.dropna(subset=['machine_hours_numeric'])
            
            if not valid_hours.empty:
                fig_hours = px.histogram(
                    valid_hours,
                    x='machine_hours_numeric',
                    nbins=30,
                    title="Distribution of Machine Hours at Time of Claim",
                    labels={'machine_hours_numeric': 'Machine Hours', 'count': 'Number of Claims'}
                )
                fig_hours.update_layout(height=400)
                st.plotly_chart(fig_hours, use_container_width=True)
        
        # Recent claims table
        st.subheader("Recent Claims")
        recent_claims = filtered_claims.head(10)[['chassis_id', 'requested_date', 'machine_salesmodel', 'fault_code', 'state']]
        st.dataframe(recent_claims, use_container_width=True)
    
    with tab3:
        st.header("🛠️ Tech Support Analysis")
        
        # Filter tech support data
        filtered_tech = tech_support_df.copy()
        if len(date_range) == 2:
            filtered_tech = filtered_tech[
                (filtered_tech['requested_date'].dt.date >= date_range[0]) &
                (filtered_tech['requested_date'].dt.date <= date_range[1])
            ]
        if selected_model != 'All':
            filtered_tech = filtered_tech[filtered_tech['machine_salesmodel'] == selected_model]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Support cases by defect code
            if 'defect_code_description' in filtered_tech.columns:
                defect_counts = filtered_tech['defect_code_description'].value_counts().head(10)
                fig_defects = px.bar(
                    x=defect_counts.values,
                    y=defect_counts.index,
                    orientation='h',
                    title="Top Defect Types",
                    labels={'x': 'Frequency', 'y': 'Defect Type'}
                )
                fig_defects.update_layout(height=500)
                st.plotly_chart(fig_defects, use_container_width=True)
        
        with col2:
            # Function group analysis
            if 'function_group' in filtered_tech.columns:
                function_counts = filtered_tech['function_group'].value_counts().head(10)
                fig_functions = px.pie(
                    values=function_counts.values,
                    names=function_counts.index,
                    title="Issues by Function Group"
                )
                fig_functions.update_layout(height=500)
                st.plotly_chart(fig_functions, use_container_width=True)
        
        # Support case timeline
        if 'requested_date' in filtered_tech.columns:
            st.subheader("Support Cases Timeline")
            tech_monthly = filtered_tech.groupby(filtered_tech['requested_date'].dt.to_period('M')).size()
            fig_tech_timeline = px.line(
                x=tech_monthly.index.astype(str),
                y=tech_monthly.values,
                title="Support Cases Over Time",
                labels={'x': 'Month', 'y': 'Number of Cases'}
            )
            fig_tech_timeline.update_layout(height=400)
            st.plotly_chart(fig_tech_timeline, use_container_width=True)
        
        # Recent support cases
        st.subheader("Recent Support Cases")
        recent_tech = filtered_tech.head(10)[['case_ts_number', 'requested_date', 'machine_salesmodel', 'part_description', 'state']]
        st.dataframe(recent_tech, use_container_width=True)
    
    with tab4:
        st.header("📈 Equipment Telemetry")
        
        if not matris_df.empty:
            # Chassis ID selector for telemetry
            chassis_options = ['All'] + sorted(matris_df['ChassisId'].unique().tolist())
            selected_chassis = st.selectbox("Select Chassis ID", chassis_options)
            
            if selected_chassis != 'All':
                filtered_matris = matris_df[matris_df['ChassisId'] == selected_chassis]
            else:
                filtered_matris = matris_df
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Measure types distribution
                measure_counts = filtered_matris['MeasureName'].value_counts().head(15)
                fig_measures = px.bar(
                    x=measure_counts.values,
                    y=measure_counts.index,
                    orientation='h',
                    title="Most Monitored Parameters",
                    labels={'x': 'Frequency', 'y': 'Measure Name'}
                )
                fig_measures.update_layout(height=600)
                st.plotly_chart(fig_measures, use_container_width=True)
            
            with col2:
                # Data source distribution
                if 'Source' in filtered_matris.columns:
                    source_counts = filtered_matris['Source'].value_counts()
                    fig_sources = px.pie(
                        values=source_counts.values,
                        names=source_counts.index,
                        title="Data Sources"
                    )
                    fig_sources.update_layout(height=600)
                    st.plotly_chart(fig_sources, use_container_width=True)
            
            # Time series analysis for selected chassis
            if selected_chassis != 'All':
                st.subheader(f"Telemetry Timeline for {selected_chassis}")
                
                # Select specific measure for time series
                measures = filtered_matris['MeasureName'].unique()
                selected_measure = st.selectbox("Select Parameter", measures)
                
                measure_data = filtered_matris[filtered_matris['MeasureName'] == selected_measure]
                
                if not measure_data.empty and 'MeasureValue' in measure_data.columns:
                    # Convert MeasureValue to numeric
                    measure_data['MeasureValue_numeric'] = pd.to_numeric(measure_data['MeasureValue'], errors='coerce')
                    valid_data = measure_data.dropna(subset=['MeasureValue_numeric'])
                    
                    if not valid_data.empty:
                        fig_timeseries = px.line(
                            valid_data,
                            x='Timestamp',
                            y='MeasureValue_numeric',
                            title=f"{selected_measure} Over Time",
                            labels={'MeasureValue_numeric': 'Value', 'Timestamp': 'Time'}
                        )
                        fig_timeseries.update_layout(height=400)
                        st.plotly_chart(fig_timeseries, use_container_width=True)
            
            # Recent telemetry data
            st.subheader("Recent Telemetry Data")
            recent_telemetry = filtered_matris.head(10)[['ChassisId', 'Timestamp', 'MeasureName', 'MeasureValue', 'Unit']]
            st.dataframe(recent_telemetry, use_container_width=True)
        else:
            st.info("No telemetry data available")
    
    with tab5:
        st.header("🔮 Predictive Analysis")
        risk_df, risk_path = load_risk_scores()
        col_btn1, col_btn2 = st.columns([1,1])
        with col_btn1:
            if _generate_machine_risk and st.button("⚙️ Recompute Risk Scores", use_container_width=True):
                with st.spinner("Computing risk scores..."):
                    try:
                        new_scores = _generate_machine_risk()
                        if _save_risk_outputs:
                            _save_risk_outputs(new_scores)
                        load_risk_scores.clear()
                        risk_df, risk_path = load_risk_scores()
                        st.success("Risk scores recomputed ✅")
                    except Exception as e:
                        st.error(f"Risk computation failed: {e}")
        with col_btn2:
            if st.button("🔁 Refresh View", use_container_width=True):
                load_risk_scores.clear(); risk_df, risk_path = load_risk_scores(); st.info("View refreshed")

        if risk_df.empty:
            st.info("No risk scores available yet. Generate them with the 'Recompute Risk Scores' button if enabled.")
        else:
            st.caption(f"Loaded risk scores from: {risk_path}")
            # Basic metrics
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.metric("Machines Scored", f"{risk_df['chassis_id'].nunique():,}")
            with mcol2:
                st.metric("Avg Risk", f"{risk_df['risk_score'].mean():.1f}")
            with mcol3:
                st.metric("High Risk (>=70)", f"{(risk_df['risk_score']>=70).sum():,}")
            with mcol4:
                st.metric("Max Risk", f"{risk_df['risk_score'].max():.1f}")

            # Filters
            with st.expander("Filters", expanded=True):
                min_score = st.slider("Minimum Risk Score", 0, 100, 0, 1)
                search_chassis = st.text_input("Search Chassis ID contains")
                filtered_risk = risk_df[risk_df['risk_score'] >= min_score].copy()
                if search_chassis:
                    filtered_risk = filtered_risk[filtered_risk['chassis_id'].astype(str).str.contains(search_chassis, case=False, na=False)]

            # Distribution
            dcol1, dcol2 = st.columns([1.4,1])
            with dcol1:
                fig_hist = px.histogram(filtered_risk, x='risk_score', nbins=25, title='Risk Score Distribution')
                fig_hist.update_layout(height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
            with dcol2:
                top_n = filtered_risk.sort_values('risk_score', ascending=False).head(15)
                fig_top = px.bar(top_n, x='risk_score', y='chassis_id', orientation='h', title='Top 15 High-Risk Machines')
                fig_top.update_layout(height=400)
                st.plotly_chart(fig_top, use_container_width=True)

            # Detail selection
            st.subheader("Risk Details")
            chassis_sel = st.selectbox("Select Machine", ['(None)'] + filtered_risk['chassis_id'].astype(str).tolist())
            if chassis_sel != '(None)':
                row = filtered_risk[filtered_risk['chassis_id'].astype(str) == chassis_sel].iloc[0]
                st.markdown(f"**Risk Score:** {row['risk_score']}  ")
                st.markdown(f"**Explanation:** {row.get('risk_explanation','(n/a)')}")
                # Show underlying contributing columns if present
                contrib_cols = [c for c in ['claim_freq_z','open_ratio','unique_faults_norm','recent_spike_flag','severity_signal'] if c in filtered_risk.columns]
                if contrib_cols:
                    st.markdown("**Contributing Factors:**")
                    contrib_df = row[contrib_cols].to_frame(name='value')
                    st.dataframe(contrib_df)

            # Full table
            with st.expander("Full Risk Table", expanded=False):
                display_cols = [c for c in ['chassis_id','risk_score','risk_explanation','open_ratio','unique_faults_norm','recent_spike_flag','severity_signal'] if c in filtered_risk.columns]
                st.dataframe(filtered_risk[display_cols].sort_values('risk_score', ascending=False), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit | Equipment Analytics Dashboard")

if __name__ == "__main__":
    main()