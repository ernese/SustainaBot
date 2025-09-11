# SustainaBot: Philippine Sustainability Analytics Platform

## Overview

**SustainaBot** is an intelligent data analytics platform designed for Philippine climate, economic, and sustainability data analysis. This proof-of-concept demonstrates advanced data engineering, LLM integration, and automated exploratory data analysis capabilities.

### Key Features

- **Medallion Architecture**: Bronze/Silver/Gold data layers for scalable data processing
- **LLM Integration**: Azure OpenAI + LangChain for natural language data queries
- **Automated EDA**: Intelligent exploratory data analysis with statistical insights
- **Multi-Source Data**: Government APIs (PSA, DOE), NASA climate data, World Bank indicators
- **Interactive Analytics**: Streamlit-based dashboard with real-time visualizations

## Technology Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python, PySpark, SQLAlchemy |
| **Frontend** | Streamlit, Plotly, D-Tale |
| **LLM** | Azure OpenAI, LangChain |
| **Data Processing** | Apache Spark, Pandas, NumPy |
| **Analytics** | ydata-profiling, SciPy, scikit-learn |
| **Storage** | Delta Lake, Parquet |

## Data Sources

- **Philippine Statistics Authority (PSA)**: Population, economics, labor
- **Department of Energy (DOE)**: Energy consumption, pricing
- **NASA**: Climate and weather data
- **World Bank**: International development indicators
  
## Key Components

### 1. Data Pipeline (`src/data_pipeline/`)
- **Bronze Layer**: Raw data ingestion with schema discovery
- **Silver Layer**: Data cleaning, transformation, and quality checks
- **Gold Layer**: Analytics-ready datasets with KPIs and aggregations

### 2. LLM Integration (`src/llm/`)
- Natural language to SQL translation
- Context-aware response generation
- Intelligent table selection and routing

### 3. Analytics Engine (`src/core/`)
- Automated exploratory data analysis
- Statistical profiling and pattern detection
- Interactive visualization generation

## Sample Analytics

- **Climate Analysis**: Typhoon patterns, monsoon detection, temperature trends
- **Economic Insights**: Regional GDP analysis, sector performance
- **Sustainability Tracking**: SDG progress monitoring, carbon footprint analysis
- **Geographic Intelligence**: Province/city level comparative analysis
