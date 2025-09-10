"""
SustainaBot - 4-Stage SQL Query Pipeline Application
Implements enhanced architecture: Context Retrieval → Table Decider → SQL Generator → Data Transformer
"""

import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime
import logging
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Ensure `src` is on sys.path for package imports
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Import the 4-stage pipeline and LLM client
from data_pipeline.four_stage_pipeline import FourStageQueryPipeline
from llm.llm_client import get_llm_config_summary, test_llm_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="SustainaBot - 4-Stage Pipeline",
    page_icon="PH",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the application
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #0066CC 0%, #FF6B35 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .pipeline-stage {
        background-color: #f8f9fa;
        border-left: 4px solid #0066CC;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .stage-success {
        border-left-color: #28a745;
        background-color: #f8fff8;
    }
    
    .stage-error {
        border-left-color: #dc3545;
        background-color: #fff5f5;
    }
    
    .metrics-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "pipeline_ready" not in st.session_state:
        st.session_state.pipeline_ready = False
    if "openai_api_key" not in st.session_state:
        # Prefill from environment (no sidebar input) with common fallbacks
        st.session_state.openai_api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENAI_KEY")
            or os.getenv("AZURE_OPENAI_API_KEY")
            or ""
        )
    if "enable_vector_search" not in st.session_state:
        st.session_state.enable_vector_search = True

def setup_sidebar() -> Dict[str, Any]:
    """Setup sidebar with table selector. Chat style removed (Data Analyst only)."""
    st.sidebar.title("SustainaBot Pipeline")
    # Always Data Analyst style
    st.session_state.response_style = "Data Analyst"

    # Layer and table selection
    selected_layer = None
    selected_table = None

    if st.session_state.pipeline_ready and st.session_state.pipeline:
        try:
            schema_summary = st.session_state.pipeline.get_schema_summary()
            layers = [k for k in ["gold", "silver", "bronze"] if k in schema_summary.get("by_layer", {})]
            if layers:
                st.sidebar.subheader("Select Layer and Table")
                selected_layer = st.sidebar.radio("Layer", layers, index=0)
                layer_tables = schema_summary["by_layer"][selected_layer]["tables"]
                # Curate choices for quality
                if selected_layer == "gold":
                    preferred_gold = [
                        "gold_co2_trends_visualization",
                        "gold_co2_emissions_summary",
                        "gold_climate_annual_v2",
                    ]
                    # Remove SDG set explicitly if present
                    layer_tables = [t for t in layer_tables if t != "gold_sdg_indicators"]
                    # Order by preference if available
                    ordered = [t for t in preferred_gold if t in layer_tables]
                    remaining = [t for t in layer_tables if t not in ordered]
                    layer_tables = (ordered + remaining)[:3]  # 2–3 max
                elif selected_layer == "silver":
                    # Prefer specific high-quality fact tables
                    preferred_silver = [
                        "silver_fact_energy_consumption",
                        "silver_fact_electricity_pricing",
                        "silver_fact_climate_weather_v2",
                        "silver_fact_economic_indicators",
                        "silver_fact_agricultural_land",
                    ]
                    filtered = [t for t in preferred_silver if t in layer_tables]
                    if filtered:
                        layer_tables = filtered[:6]  # 5–6 max
                    else:
                        # Fallback: keep only fact tables
                        layer_tables = [t for t in layer_tables if ("_fact_" in t) or t.startswith("silver_fact_")][:6]
                elif selected_layer == "bronze":
                    # Curate bronze to NASA and select new_bronze annual series only
                    preferred_bronze = [
                        # NASA daily climate
                        "bronze_nasa_c01_v2",
                        "bronze_nasa_c03_v2",
                        "bronze_nasa_c04_v2",
                        "bronze_nasa_c09_v2",
                        "bronze_nasa_c12_v2",
                        "bronze_nasa_c13_v2",
                        # NASA compact annual
                        "bronze_nasa_c23_v2",
                        # new_bronze annual series (names may be prefixed by bronze_new_bronze_)
                        "bronze_new_bronze_G01_carbon_dioxide_co2_emissions_from_the_power_industry_energy_sector_in_millions_of_tonnes_of_co2_equivalent_1980_2024",
                        "bronze_new_bronze_G03_co2_emissions_total_excluding_lulucf_mt_co2e_1980_2024",
                        "bronze_new_bronze_G06_total_greenhouse_gas_emissions_including_lulucf_mt_co2e_1980_2024",
                        "bronze_new_bronze_L08_agricultural_land_pct_of_land_area_1980_2024",
                        "bronze_new_bronze_L09_agricultural_irrigated_land_pct_of_total_agricultural_land_1980_2024",
                        "bronze_new_bronze_L10_arable_land_pct_of_land_area_1980_2024",
                        "bronze_new_bronze_L11_cereal_yield_kg_per_hectare_1980_2024",
                    ]
                    filtered = [t for t in preferred_bronze if t in layer_tables]
                    # If none matched (name variations), also include keys that start with NASA patterns
                    if not filtered:
                        nasa_like = [t for t in layer_tables if t.startswith("bronze_nasa_c")]
                        filtered = nasa_like
                    picked = filtered or layer_tables
                    # Cap to 10–12
                    layer_tables = picked[:12]
                table_options = ["Auto (let system decide)"] + layer_tables
                table_choice = st.sidebar.selectbox("Table", table_options, index=0)
                if table_choice != "Auto (let system decide)":
                    selected_table = table_choice

                # Removed global all-tables picker and browse expanders per request
        except Exception as e:
            st.sidebar.warning(f"Table list unavailable: {e}")

    # Clear Chat
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # Return options
    return {
        "response_style": st.session_state.response_style,
        "force_layer": selected_layer,
        "force_table": selected_table,
    }

def initialize_pipeline():
    """Initialize the 4-stage pipeline using environment/API defaults (no UI inputs)."""
    # Ensure we pick up env var if present
    if not st.session_state.openai_api_key:
        st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")
    
    try:
        with st.spinner("Initializing 4-Stage Pipeline with RAG..."):
            st.session_state.pipeline = FourStageQueryPipeline(
                api_key=st.session_state.openai_api_key,
                data_path=os.getenv("DATA_PATH"),
                enable_vector_search=bool(st.session_state.get("enable_vector_search", True))
            )
            st.session_state.pipeline_ready = True
        
        st.sidebar.success("SO-bot initialized!")
        logger.info("Pipeline successfully initialized")
        
    except Exception as e:
        st.sidebar.error(f"Pipeline initialization failed: {str(e)}")
        logger.error(f"Pipeline initialization error: {str(e)}")

def show_table_registry():
    """Display table registry information."""
    if not st.session_state.pipeline_ready:
        st.error("Pipeline not initialized")
        return
    
    registry_summary = st.session_state.pipeline.get_table_registry_summary()
    
    st.subheader("Philippine Lakehouse Table Registry")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tables", registry_summary["total_tables"])
    
    with col2:
        bronze_count = registry_summary["by_layer"].get("bronze", 0)
        st.metric("Bronze Tables", bronze_count)
    
    with col3:
        silver_count = registry_summary["by_layer"].get("silver", 0) 
        st.metric("Silver Tables", silver_count)
    
    with col4:
        gold_count = registry_summary["by_layer"].get("gold", 0)
        st.metric("Gold Tables", gold_count)
    
    # Show layer distribution chart
    layer_data = registry_summary["by_layer"]
    if layer_data:
        fig = px.pie(
            values=list(layer_data.values()),
            names=list(layer_data.keys()),
            title="Table Distribution by Layer",
            color_discrete_map={
                "bronze": "#CD7F32",
                "silver": "#C0C0C0", 
                "gold": "#FFD700"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

def run_pipeline_tests():
    """Run pipeline tests and display results."""
    if not st.session_state.pipeline_ready:
        st.error("Pipeline not initialized")
        return
    
    st.subheader("Pipeline Test Results")
    
    with st.spinner("Running pipeline tests..."):
        test_results = st.session_state.pipeline.test_pipeline_with_sample_queries()
    
    # Display test summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Success Rate", f"{test_results['success_rate']:.1f}%")
    
    with col2:
        st.metric("Passed Tests", test_results['successful_tests'])
        
    with col3:
        st.metric("Failed Tests", test_results['failed_tests'])
    
    # Display detailed test results
    st.subheader("Test Details")
    
    for test in test_results['test_details']:
        status_color = "success" if "PASSED" in test['status'] else "error"
        
        with st.container():
            st.markdown(f"""
            <div class="pipeline-stage stage-{status_color}">
                <strong>{test['status']}</strong> {test['query']}<br>
                <small>
                Table: {test.get('table_selected', 'N/A')} | 
                SQL: {'PASS' if test.get('sql_generated') else 'FAIL'} |
                Rows: {test.get('data_returned', 0)} |
                Time: {test.get('execution_time', 0):.2f}s
                </small>
            </div>
            """, unsafe_allow_html=True)
            
            if test.get('error'):
                st.error(f"Error: {test['error']}")

def show_connectivity_check():
    """Show connectivity and system health check."""
    st.subheader("System Connectivity Check")
    
    # LLM Configuration Summary
    st.subheader("LLM Configuration")
    config_summary = get_llm_config_summary()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Provider**: {config_summary['provider']}")
        st.info(f"**Endpoint**: {config_summary['endpoint']}")
        st.info(f"**Model**: {config_summary['model_name']}")
        
    with col2:
        st.info(f"**Deployment**: {config_summary['deployment']}")
        st.info(f"**API Version**: {config_summary['api_version']}")
        api_status = "Present" if config_summary['has_api_key'] else "Missing"
        st.info(f"**API Key**: {api_status}")
    
    # LLM Connection Test
    st.subheader("LLM Connection Test")
    
    with st.spinner("Testing LLM connection..."):
        test_result = test_llm_connection(st.session_state.openai_api_key)
    
    if test_result['status'] == 'success':
        st.success(f"Connection Successful")
        st.write(f"Response: {test_result['response']}")
        st.write(f"Model Used: {test_result.get('model_used', 'Unknown')}")
    else:
        st.error(f"Connection Failed")
        st.error(test_result['message'])
    
    # Data Path Check
    st.subheader("Data Path Status")
    default_repo_data = Path(__file__).resolve().parents[2] / "data"
    data_base = Path(os.getenv("DATA_PATH", str(default_repo_data)))
    paths_to_check = [
        ("Bronze Data", data_base / "raw" / "final-spark-bronze"),
        ("Silver Data", data_base / "processed" / "final-spark-silver"), 
        ("Gold Data", data_base / "outputs" / "final-spark-gold")
    ]
    
    for name, path in paths_to_check:
        if path.exists():
            try:
                file_count = len(list(path.rglob("*.parquet")))
                st.success(f"{name}: {file_count} parquet files found")
            except:
                st.warning(f"{name}: Directory exists but can't count files")
        else:
            st.error(f"{name}: Directory not found at {path}")
    
    # Pipeline Component Status
    if st.session_state.pipeline_ready:
        st.subheader("Pipeline Components")
        status = st.session_state.pipeline.get_pipeline_status()
        
        for component, info in status["components"].items():
            component_name = component.replace("_", " ").title()
            if info["status"] == "ready":
                st.success(f"{component_name}: Ready")
            else:
                st.error(f"{component_name}: {info.get('error', 'Not ready')}")
    else:
        st.warning("Pipeline not initialized - cannot check component status")

def display_message(message: Dict[str, Any], is_user: bool = False):
    """Display message in the chat interface."""
    
    if is_user:
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])
            
            # Display the result table and details when available
            if "pipeline_result" in message and message["pipeline_result"]:
                display_pipeline_result(message["pipeline_result"])

def display_pipeline_result(result: Dict[str, Any]):
    """Display detailed pipeline execution results."""
    
    # Pipeline stages overview
    st.subheader("Pipeline Execution")
    
    # Stage indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stage1_success = result.get("stage_1_table_decision", {}).get("valid", False)
        status1 = "PASS" if stage1_success else "FAIL"
        # Display Stage 0 status
        context_status = "PASS" if result.get("stage_0_context_retrieval") else "SKIP"
        st.markdown(f"**Stage 0: Context Retrieval** {context_status}")
        st.markdown(f"**Stage 1: Table Selection** {status1}")
        
        if stage1_success:
            table_name = result["stage_1_table_decision"]["table_name"]
            confidence = result["stage_1_table_decision"]["confidence"]
            st.write(f"Selected: `{table_name}` ({confidence}%)")
    
    with col2:
        stage2_success = result.get("stage_2_sql_generation", {}).get("sql_statement") is not None
        status2 = "PASS" if stage2_success else "FAIL"
        st.markdown(f"**Stage 2: SQL Generation** {status2}")
        
        if stage2_success:
            estimated_rows = result["stage_2_sql_generation"].get("estimated_rows", "Unknown")
            st.write(f"Est. rows: {estimated_rows}")
    
    with col3:
        stage3_success = result.get("final_dataframe") is not None
        status3 = "PASS" if stage3_success else "FAIL"
        st.markdown(f"**Stage 3: Data Transform** {status3}")
        
        if stage3_success:
            final_rows = len(result["final_dataframe"])
            st.write(f"Final rows: {final_rows:,}")
    
    # Execution time
    if result.get("total_execution_time"):
        st.info(f"Total execution time: {result['total_execution_time']:.2f} seconds")
    
    # Display SQL query
    if result.get("stage_2_sql_generation", {}).get("sql_statement"):
        with st.expander("Generated SQL Query"):
            sql = result["stage_2_sql_generation"]["sql_statement"]
            st.code(sql, language="sql")
    
    # Display final dataframe
    if result.get("final_dataframe") is not None:
        df = result["final_dataframe"]
        
        with st.expander(f"Results ({len(df):,} rows × {len(df.columns)} columns)", expanded=True):
            st.dataframe(df, use_container_width=True)
            
            # Data insights
            if result.get("stage_3_data_transformation", {}).get("insights"):
                insights = result["stage_3_data_transformation"]["insights"]
                
                st.subheader("Key Insights")
                for insight in insights[:5]:  # Show up to 5 insights
                    st.write(f"• {insight['insight']}")
    
    # Natural language summary
    if result.get("natural_language_summary"):
        st.info(f"Summary: {result['natural_language_summary']}")
    
    # Transformation log (guard against None)
    stage3 = result.get("stage_3_data_transformation") or {}
    if stage3.get("transformation_log"):
        with st.expander("Data Transformations Applied"):
            for transformation in stage3.get("transformation_log", []):
                st.write(f"• {transformation}")

def process_user_input(user_input: str, query_options: Dict[str, Any]) -> Dict[str, Any]:
    """Process user input through the 4-stage pipeline."""
    
    if not st.session_state.pipeline_ready:
        return {
            "content": "Pipeline not initialized. Please click 'Initialize 4-Stage Pipeline' in the sidebar.",
            "error": "Pipeline not ready"
        }
    
    try:
        # Execute the 4-stage pipeline
        with st.spinner("Executing 4-Stage Pipeline..."):
            result = st.session_state.pipeline.execute_query_pipeline(user_input, query_options)
        
        if result["pipeline_success"]:
            # Generate response content
            df = result["final_dataframe"]
            table_name = result["stage_1_table_decision"]["table_name"]
            layer = result["stage_1_table_decision"]["layer"]

            style = query_options.get("response_style") or st.session_state.get("response_style", "Data Analyst")
            nl_summary = result.get('natural_language_summary')

            # Derive quick insights from the dataframe
            def _year_column(dframe):
                for c in dframe.columns:
                    lc = c.lower()
                    if lc == 'year' or ('year' in lc and str(dframe[c].dtype).startswith('int')):
                        return c
                return None
            def _numeric_cols(dframe):
                bad_tokens = ('year','date','month','quarter')
                cols = []
                for c in dframe.columns:
                    lc = c.lower()
                    if any(tok in lc for tok in bad_tokens):
                        continue
                    if lc.endswith('_id'):
                        continue
                    if 'formatted' in lc:
                        continue
                    dt = str(dframe[c].dtype)
                    if dt.startswith('int') or dt.startswith('float'):
                        cols.append(c)
                return cols
            insights = []
            try:
                year_col = _year_column(df)
                num_cols = _numeric_cols(df)
                if year_col and len(num_cols) > 0:
                    focus = num_cols[0]
                    # Aggregate by year
                    ts = (
                        df[[year_col, focus]]
                        .dropna()
                        .groupby(year_col, as_index=False)
                        .mean()
                        .sort_values(year_col)
                    )
                    years = ts[year_col].tolist()
                    vals = ts[focus].tolist()
                    if years and vals:
                        latest = years[-1]
                        latest_val = float(vals[-1])
                        # YoY change
                        if len(vals) > 1 and vals[-2] != 0:
                            yoy = ((vals[-1] - vals[-2]) / abs(vals[-2])) * 100.0
                            insights.append(
                                f"{focus} in {latest}: {latest_val:,.2f} ({yoy:+.1f}% vs {years[-2]})"
                            )
                        else:
                            insights.append(f"{focus} in {latest}: {latest_val:,.2f}")
                        # Range and period length
                        vmin, vmax = float(min(vals)), float(max(vals))
                        insights.append(
                            f"Range of {focus}: {vmin:,.2f} to {vmax:,.2f} across {len(years)} years"
                        )
                        # CAGR over the full period (guard zeros/negatives)
                        if vals[0] not in (0, None) and vals[0] > 0 and vals[-1] > 0 and len(years) > 1:
                            n = years[-1] - years[0]
                            if n > 0:
                                cagr = ((vals[-1] / vals[0]) ** (1.0 / n) - 1.0) * 100.0
                                insights.append(f"Compound annual growth (approx): {cagr:+.2f}% per year")
                        # Volatility (coefficient of variation)
                        import numpy as _np
                        mean_v = float(_np.mean(vals)) if vals else None
                        std_v = float(_np.std(vals)) if vals else None
                        if mean_v and mean_v != 0 and std_v is not None:
                            cv = (std_v / abs(mean_v)) * 100.0
                            insights.append(f"Volatility (CV): {cv:.1f}% over the period")
                        # Best/Worst year
                        import numpy as _np2
                        idx_max = int(_np2.argmax(vals))
                        idx_min = int(_np2.argmin(vals))
                        insights.append(
                            f"Best year for {focus}: {years[idx_max]} ({vals[idx_max]:,.2f}); Worst: {years[idx_min]} ({vals[idx_min]:,.2f})"
                        )
                # Add a second metric summary if available
                m2 = None
                if len(_numeric_cols(df)) > 1:
                    m2 = _numeric_cols(df)[1]
                if m2:
                    m2_mean = float(df[m2].dropna().mean()) if not df[m2].dropna().empty else None
                    if m2_mean is not None:
                        insights.append(f"Average {m2}: {m2_mean:,.2f}")
            except Exception:
                pass

            if style == "Data Analyst":
                content_lines = [
                    "Analysis Result",
                    f"Query: {user_input}",
                    f"Table: {table_name} ({layer})",
                    f"Rows: {len(df):,}  Columns: {len(df.columns) if df is not None else 0}",
                ]
                if insights:
                    content_lines.append("Key Insights:")
                    for s in insights[:3]:
                        content_lines.append(f"- {s}")
                if nl_summary:
                    content_lines.append(f"Summary: {nl_summary}")
                content = "\n".join(content_lines) + "\n"

            # Removed Layer EDA Overview from the final message per request

            return {
                "content": content,
                "pipeline_result": result,
                "success": True
            }
        
        else:
            error_msg = result.get("error", "Unknown pipeline error")
            # Attempt a general response (not tied to tables) if LLM available
            general_answer = None
            try:
                from llm.response_generator_llm import SustainaBotResponseGenerator
                rg = SustainaBotResponseGenerator(api_key=st.session_state.openai_api_key)
                gr = rg.generate_response(user_input, analysis_type="general")
                if not gr.get('error'):
                    general_answer = gr.get('answer')
            except Exception:
                general_answer = None

            content = f"Pipeline Error\n\n{error_msg}"
            if general_answer:
                content += f"\n\nGeneral Answer:\n{general_answer}"

            return {
                "content": content,
                "pipeline_result": result,
                "error": error_msg
            }
    
    except Exception as e:
        logger.error(f"Error processing user input: {str(e)}")
        return {
            "content": f"System Error\n\n{str(e)}",
            "error": str(e)
        }

def show_sample_queries():
    """Display layer-aligned sample queries for reliable Stage 2."""
    st.subheader("Sample Queries")
    st.markdown("Layer-specific examples that map cleanly to quality datasets:")

    samples = [
        # Gold (curated)
        ("Gold", "Show annual CO2 emission trends"),
        ("Gold", "Show climate annual trends for the Philippines"),
        # Silver (facts)
        ("Silver", "Show energy consumption by region"),
        ("Silver", "Show average electricity pricing by year"),
        ("Silver", "Show climate metrics by year"),
        # Bronze (NASA)
        ("Bronze", "Show yearly breakdown from NASA climate temperature data"),
    ]

    cols = st.columns(2)
    for i, (layer, query) in enumerate(samples):
        label = f"[{layer}] {query}"
        col = cols[i % 2]
        with col:
            if st.button(label, key=f"sample_{i}", use_container_width=True):
                st.session_state.messages.append({"content": query, "is_user": True})
                st.rerun()

def main():
    """Main application function."""
    
    # Initialize session state
    initialize_session_state()
    
    # Auto-initialize pipeline (no UI buttons) before sidebar so we can list tables
    if not st.session_state.pipeline_ready:
        initialize_pipeline()

    # Setup sidebar and get query options
    query_options = setup_sidebar()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>SustainaBot</h1>
        <h3>Philippine Sustainability Data Assistant</h3>
        <p><em>Ask questions about Philippine climate, economic, and sustainability data</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # About section removed per request
    
    # Chat interface
    st.subheader("Chat with SustainaBot")
    
    with st.container():
        # Display chat history
        for message in st.session_state.messages:
            display_message(message, is_user=message.get("is_user", False))
        
        # Chat input
        if user_input := st.chat_input("Ask about Philippine climate, economic, or sustainability data..."):
            # Add user message
            user_message = {"content": user_input, "is_user": True}
            st.session_state.messages.append(user_message)
            display_message(user_message, is_user=True)
            
            # Process through pipeline
            with st.spinner("Processing through 4-stage pipeline..."):
                response = process_user_input(user_input, query_options)
            
            # Add assistant response
            st.session_state.messages.append(response)
            display_message(response)
    
    # Sample queries section
    if len(st.session_state.messages) == 0:
        show_sample_queries()
    
    # Footer with pipeline info
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        <p>SustainaBot | 4-Stage SQL Query Pipeline for Philippine Data Analysis</p>
        <p>Bronze Layer (Raw Data) • Silver Layer (Processed Data) • Gold Layer (Analytics Ready)</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
