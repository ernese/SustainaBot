"""
Stage 3: Data Transformation Pipeline for SustainaBot
Executes SQL queries and transforms raw results into user-friendly, 
analysis-ready dataframes with Philippine context and business insights.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
from datetime import datetime, timedelta
import warnings
import re
from llm.llm_client import get_llm_client, get_model_name
import os

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataTransformer:
    """
    Data execution and transformation pipeline that takes SQL queries,
    executes them against Philippine lakehouse data, and transforms results 
    into user-friendly, analysis-ready formats.
    """
    
    def __init__(self, api_key: Optional[str] = None, data_path: Optional[str] = None):
        """Initialize the Data Transformer with data access and enhancement capabilities."""
        
        self.api_key = api_key
        self.client = get_llm_client(api_key=api_key) if api_key else None
        
        # Data paths for lakehouse access
        default_repo_data = Path(__file__).resolve().parents[2] / "data"
        self.data_path = Path(data_path or os.getenv("DATA_PATH", str(default_repo_data)))
        
        # Initialize data source mappings
        self._initialize_data_mappings()
        
        # Philippine context for transformations
        self.philippine_context = self._load_philippine_context()
        
        logger.info("Data Transformer initialized for Philippine lakehouse")
    
    def _initialize_data_mappings(self):
        """Initialize mappings between table names and actual data files."""
        
        self.data_mappings = {
            # Bronze layer mappings
            "bronze_psa_income_consumption": {
                "file_patterns": ["*income*consumption*.parquet", "*psa*income*.parquet"],
                "directories": ["final-spark-bronze", "bronze_income_consumption"]
            },
            "bronze_psa_employment_labor": {
                "file_patterns": ["*employment*.parquet", "*labor*.parquet"],
                "directories": ["final-spark-bronze", "bronze_labor"]
            },
            "bronze_nasa_climate_temperature": {
                "file_patterns": ["*nasa*climate*.parquet", "*temperature*.parquet"],
                "directories": ["final-spark-bronze", "bronze_nasa_c23_v2"]
            },
            "bronze_doe_energy_consumption": {
                "file_patterns": ["*energy*consumption*.parquet", "*doe*.parquet"],
                "directories": ["final-spark-bronze", "bronze_energy"]
            },
            
            # Silver layer mappings  
            "silver_fact_climate_weather": {
                "file_patterns": ["fact_climate_weather*.parquet", "*climate*.parquet"],
                "directories": ["vectorization_data_silver/silver", "final-spark-silver"]
            },
            "silver_fact_economic_indicators": {
                "file_patterns": ["part-00000-*.snappy.parquet"],
                "directories": ["final-spark-gold", "gold_economic_performance_dashboard"]
            },
            "silver_fact_agricultural_land": {
                "file_patterns": ["fact_agricultural_land*.parquet", "*agricultural*.parquet"],
                "directories": ["vectorization_data_silver/silver", "final-spark-silver"]
            },
            "silver_fact_energy_consumption": {
                "file_patterns": ["fact_energy_consumption*.parquet", "*energy*.parquet"],
                "directories": ["vectorization_data_silver/silver", "final-spark-silver"]
            },
            "silver_dim_location": {
                "file_patterns": ["dim_location*.parquet", "*location*.parquet"],
                "directories": ["vectorization_data_silver/silver", "final-spark-silver"]
            },
            "silver_dim_time": {
                "file_patterns": ["dim_time*.parquet"],
                "directories": ["vectorization_data_silver/silver", "final-spark-silver"]
            },
            
            # Gold layer mappings
            "gold_sustainability_scorecard": {
                "file_patterns": ["*sustainability*scorecard*", "*sustainability*.parquet"],
                "directories": ["final-spark-gold", "gold_sustainability_scorecard"]
            },
            "gold_integrated_dashboard": {
                "file_patterns": ["*integrated*dashboard*", "*dashboard*.parquet"],
                "directories": ["final-spark-gold", "gold_integrated_dashboard"]
            },
            
            # Additional Gold layer mappings for existing files
            "gold_economic_performance_dashboard": {
                "file_patterns": ["part-00000-*.snappy.parquet"],
                "directories": ["final-spark-gold", "gold_economic_performance_dashboard"]
            },
            "gold_climate_annual_v2": {
                "file_patterns": ["part-00000-*.csv"],
                "directories": ["final-spark-gold", "gold_climate_annual_v2"]
            },
            "gold_sdg_indicators": {
                "file_patterns": ["part-00000-*.csv"],
                "directories": ["final-spark-gold", "gold_sdg_indicators"]
            },
            "gold_sustainability_scorecard": {
                "file_patterns": ["part-00000-*.csv"],
                "directories": ["final-spark-gold", "gold_sustainability_scorecard"]
            }
        }
    
    def _load_philippine_context(self) -> Dict[str, Any]:
        """Load Philippine-specific context for data transformations."""
        
        return {
            "regions": {
                "NCR": "National Capital Region",
                "CAR": "Cordillera Administrative Region", 
                "Region I": "Ilocos Region",
                "Region II": "Cagayan Valley",
                "Region III": "Central Luzon",
                "Region IV-A": "CALABARZON",
                "Region IV-B": "MIMAROPA",
                "Region V": "Bicol Region",
                "Region VI": "Western Visayas",
                "Region VII": "Central Visayas", 
                "Region VIII": "Eastern Visayas",
                "Region IX": "Zamboanga Peninsula",
                "Region X": "Northern Mindanao",
                "Region XI": "Davao Region",
                "Region XII": "SOCCSKSARGEN",
                "Region XIII": "Caraga"
            },
            "island_groups": {
                "Luzon": ["NCR", "CAR", "Region I", "Region II", "Region III", "Region IV-A", "Region IV-B", "Region V"],
                "Visayas": ["Region VI", "Region VII", "Region VIII"],
                "Mindanao": ["Region IX", "Region X", "Region XI", "Region XII", "Region XIII"]
            },
            "currency_format": "PHP",
            "date_format": "YYYY-MM-DD",
            "fiscal_year_start": 7,  # July
            "seasons": {
                "wet": [6, 7, 8, 9, 10],  # June to October
                "dry": [11, 12, 1, 2, 3, 4, 5]  # November to May
            },
            "unit_conversions": {
                "temperature": {"celsius": "°C", "fahrenheit": "°F"},
                "area": {"km2": "km²", "hectare": "ha"},
                "currency": {"php": "₱", "usd": "$"}
            }
        }
    
    def execute_and_transform(self, sql_statement: str, table_name: str, 
                            user_query: str, transformation_options: Optional[Dict] = None,
                            schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute SQL query and transform results into user-friendly format.
        
        Args:
            sql_statement: SQL query to execute
            table_name: Target table name
            user_query: Original user query for context
            transformation_options: Optional transformation preferences
            
        Returns:
            Dict containing original dataframe, transformed dataframe, and metadata
        """
        try:
            # Step 1: Execute SQL and get raw dataframe
            raw_dataframe = self._execute_sql(sql_statement, table_name, schema)

            if raw_dataframe is None:
                return {
                    "raw_dataframe": None,
                    "transformed_dataframe": None,
                    "error": "Failed to execute SQL query",
                    "sql_statement": sql_statement
                }

            # Optional: improve column names using schema descriptions when available
            try:
                if schema and isinstance(schema, dict):
                    raw_dataframe = self._apply_schema_aliases(raw_dataframe, schema)
            except Exception as e:
                logger.warning(f"Could not apply schema aliases: {e}")
            
            # Step 2: Apply transformations to create altered dataframe
            transformation_options = transformation_options or {}
            transformed_dataframe = self._apply_transformations(
                raw_dataframe, table_name, user_query, transformation_options
            )
            
            # Step 3: Generate insights and metadata
            insights = self._generate_data_insights(transformed_dataframe, user_query)
            
            # Step 4: Create user-friendly summary
            summary = self._create_user_summary(transformed_dataframe, user_query, insights)
            
            logger.info(f"Successfully transformed data: {len(raw_dataframe)} → {len(transformed_dataframe)} rows")
            
            return {
                "raw_dataframe": raw_dataframe,
                "transformed_dataframe": transformed_dataframe,
                "insights": insights,
                "summary": summary,
                "transformation_log": self._get_transformation_log(),
                "sql_statement": sql_statement,
                "table_name": table_name,
                "user_query": user_query,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error in data transformation: {str(e)}")
            return {
                "raw_dataframe": None,
                "transformed_dataframe": None,
                "error": str(e),
                "sql_statement": sql_statement,
                "table_name": table_name,
                "user_query": user_query
            }

    def _apply_schema_aliases(self, df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
        """Rename opaque column names (e.g., col-xxxxxxxx) using schema descriptions if present."""
        try:
            cols = schema.get('columns', [])
            mapping = {}
            for col in cols:
                if not isinstance(col, dict):
                    continue
                name = col.get('name')
                desc = (col.get('description') or '').strip()
                if not name or not desc:
                    continue
                # Opaque column pattern or generic codes
                if name.startswith('col-') or all(ch.isalnum() or ch in '-_' for ch in name) and name.startswith('col'):
                    # Build a concise alias from description (first 4 words, snake_case)
                    alias = desc
                    # remove non-alnum
                    import re as _re
                    alias = _re.sub(r"[^A-Za-z0-9\s]", "", alias)
                    words = [w for w in alias.strip().split() if w]
                    if not words:
                        continue
                    alias = "_".join(words[:4]).lower()
                    # ensure uniqueness
                    base = alias
                    i = 1
                    while alias in mapping.values() or alias in df.columns:
                        alias = f"{base}_{i}"
                        i += 1
                    mapping[name] = alias
            if mapping:
                return df.rename(columns=mapping)
        except Exception as e:
            logger.warning(f"Alias mapping failed: {e}")
        return df
    
    def _execute_sql(self, sql_statement: str, table_name: str, schema: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
        """
        Execute SQL statement against lakehouse data files.
        Since we're working with parquet files, we'll simulate SQL execution with pandas.
        """
        try:
            # Load the target table data
            df = self._load_table_data(table_name, schema, sql_statement)
            
            if df is None:
                logger.error(f"Could not load data for table {table_name}")
                return None
            
            # Simulate SQL execution with pandas operations
            result_df = self._simulate_sql_execution(sql_statement, df, table_name)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error executing SQL: {str(e)}")
            return None
    
    def _load_table_data(self, table_name: str, schema: Optional[Dict[str, Any]] = None, sql_statement: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load data for the specified table from lakehouse files."""

        # If schema contains a discovered data file/path, use it directly
        try:
            if schema:
                data_file = schema.get('data_file')
                if data_file:
                    return self._read_partitioned_if_needed(Path(data_file), sql_statement)
                file_path = schema.get('file_path')
                if file_path:
                    # Try to load a representative file from this directory
                    dir_path = Path(file_path)
                    for ext in ('*.parquet', '*.csv'):
                        files = list(dir_path.rglob(ext))
                        if files:
                            return self._read_partitioned_if_needed(files[0], sql_statement)
        except Exception as e:
            logger.warning(f"Schema-based data load failed: {e}")
        
        if table_name not in self.data_mappings:
            logger.warning(f"No mapping found for table {table_name}")
            return self._discover_and_load_data(table_name)
        
        mapping = self.data_mappings[table_name]
        
        # Try each directory and file pattern
        for directory in mapping["directories"]:
            dir_path = self.data_path / directory
            
            if not dir_path.exists():
                continue
            
            for pattern in mapping["file_patterns"]:
                # Search recursively to support partitioned datasets (e.g., year=.../category=.../*.parquet)
                files = list(dir_path.rglob(pattern))
                
                if files:
                    try:
                        # Load up to N files to provide a more representative sample
                        max_files = 10
                        loaded = []
                        for f in files[:max_files]:
                            loaded.append(self._read_partitioned_if_needed(f, sql_statement))
                        if len(loaded) == 1:
                            return loaded[0]
                        df_concat = pd.concat(loaded, ignore_index=True, sort=False)
                        logger.info(f"Loaded {len(df_concat)} rows from {min(len(files), max_files)} files in {dir_path}")
                        return df_concat
                    except Exception as e:
                        logger.warning(f"Could not load {files[0]}: {str(e)}")
                        continue
        
        # If no exact match, try discovery
        return self._discover_and_load_data(table_name)

    def _read_partitioned_if_needed(self, f: Path, sql_statement: Optional[str]) -> pd.DataFrame:
        """If query groups by year and file path contains partition 'year=', combine across years."""
        try:
            if sql_statement and re.search(r"GROUP BY\s+year", sql_statement, re.IGNORECASE):
                # Find parent dir that has year=* subdirs
                root = None
                for p in [f.parent, *f.parents]:
                    try:
                        if any(child.is_dir() and child.name.startswith('year=') for child in p.iterdir() if child.is_dir()):
                            root = p
                            break
                    except Exception:
                        continue
                if root is not None:
                    dfs = []
                    for ydir in sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith('year=')]):
                        year_val = ydir.name.split('=',1)[-1]
                        files = list(ydir.rglob('*.parquet')) or list(ydir.rglob('*.csv'))
                        if not files:
                            continue
                        dfy = self._read_data_file(files[0])
                        if 'year' not in dfy.columns:
                            try:
                                dfy['year'] = int(year_val) if str(year_val).isdigit() else year_val
                            except Exception:
                                dfy['year'] = year_val
                        dfs.append(dfy)
                    if dfs:
                        combined = pd.concat(dfs, ignore_index=True)
                        logger.info(f"Loaded {len(combined)} rows across {len(dfs)} year partitions from {root}")
                        return combined
        except Exception as e:
            logger.warning(f"Partition-aware read failed: {e}")
        # Fallback to single file
        return self._read_data_file(f)
    
    def _discover_and_load_data(self, table_name: str) -> Optional[pd.DataFrame]:
        """Discover and load data when exact mapping is not available."""
        
        # Extract key terms from table name for matching
        key_terms = table_name.replace("_", " ").replace("bronze", "").replace("silver", "").replace("gold", "").strip().split()
        
        # Search in all data directories
        search_dirs = [
            "final-spark-bronze",
            "final-spark-silver", 
            "final-spark-gold",
            "vectorization_data_silver/silver"
        ]
        
        for search_dir in search_dirs:
            dir_path = self.data_path / search_dir
            
            if not dir_path.exists():
                continue
            
            # Find files matching key terms (recursive)
            for data_file in list(dir_path.rglob("*.parquet")) + list(dir_path.rglob("*.csv")):
                file_name_lower = data_file.name.lower()
                
                if any(term.lower() in file_name_lower for term in key_terms):
                    try:
                        df = self._read_data_file(data_file)
                        logger.info(f"Discovered and loaded {len(df)} rows from {data_file}")
                        return df
                    except Exception as e:
                        logger.warning(f"Could not load discovered file {data_file}: {str(e)}")
        
        logger.error(f"Could not discover data for table {table_name}")
        return None

    def _read_data_file(self, f: Path) -> pd.DataFrame:
        """Helper to read a parquet or csv file into a DataFrame."""
        if f.suffix == '.parquet':
            df = pd.read_parquet(f)
        elif f.suffix == '.csv':
            df = pd.read_csv(f)
        else:
            raise ValueError(f"Unsupported file type: {f.suffix}")
        logger.info(f"Loaded {len(df)} rows from {f}")
        return df
    
    def _simulate_sql_execution(self, sql_statement: str, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """
        Simulate SQL execution using pandas operations.
        This is a simplified approach - in production, use a proper SQL engine.
        """
        try:
            result_df = df.copy()
            
            # Parse SQL components (simplified)
            sql_upper = sql_statement.upper()
            
            # Apply WHERE clauses (simplified)
            if "WHERE" in sql_upper:
                result_df = self._apply_where_conditions(result_df, sql_statement)
            
            # Apply GROUP BY (simplified)
            if "GROUP BY" in sql_upper:
                result_df = self._apply_group_by(result_df, sql_statement)
            
            # Apply ORDER BY (simplified)
            if "ORDER BY" in sql_upper:
                result_df = self._apply_order_by(result_df, sql_statement)
            
            # Apply LIMIT (simplified)
            if "LIMIT" in sql_upper:
                result_df = self._apply_limit(result_df, sql_statement)
            
            # Apply column selection (simplified)
            result_df = self._apply_column_selection(result_df, sql_statement)
            
            return result_df
            
        except Exception as e:
            logger.warning(f"Error in SQL simulation: {str(e)}, returning original data with limit")
            # Fallback: return limited original data
            return df.head(1000)
    
    def _apply_where_conditions(self, df: pd.DataFrame, sql_statement: str) -> pd.DataFrame:
        """Apply basic WHERE conditions (simplified implementation)."""
        
        try:
            # This is a very simplified WHERE clause handler
            # In production, use a proper SQL parser
            
            # Handle basic year filters
            if re.search(r"year\s*=\s*(\d{4})", sql_statement, re.IGNORECASE):
                year_match = re.search(r"year\s*=\s*(\d{4})", sql_statement, re.IGNORECASE)
                year = int(year_match.group(1))
                
                if 'year' in df.columns:
                    df = df[df['year'] == year]
                elif 'date' in df.columns:
                    df = df[pd.to_datetime(df['date'], errors='coerce').dt.year == year]
            
            # Handle basic region filters
            if re.search(r"region.*=.*'([^']*)'", sql_statement, re.IGNORECASE):
                region_match = re.search(r"region.*=.*'([^']*)'", sql_statement, re.IGNORECASE)
                region = region_match.group(1)
                
                region_cols = [col for col in df.columns if 'region' in col.lower()]
                if region_cols:
                    df = df[df[region_cols[0]].str.contains(region, case=False, na=False)]
            
            return df
            
        except Exception as e:
            logger.warning(f"Error applying WHERE conditions: {str(e)}")
            return df
    
    def _apply_group_by(self, df: pd.DataFrame, sql_statement: str) -> pd.DataFrame:
        """Apply basic GROUP BY operations (simplified)."""
        
        try:
            # Extract GROUP BY columns (simplified)
            group_match = re.search(r"GROUP BY\s+([\w\s,]+)", sql_statement, re.IGNORECASE)
            if not group_match:
                return df
            
            group_cols_str = group_match.group(1)
            group_cols = [col.strip() for col in group_cols_str.split(',')]
            
            # Derive year from a date-like column if requested
            if 'year' in [c.lower() for c in group_cols] and 'year' not in df.columns:
                # Try to create a year column from any date-like column
                date_cols = [c for c in df.columns if 'date' in c.lower()]
                if date_cols:
                    try:
                        df = df.copy()
                        df['year'] = pd.to_datetime(df[date_cols[0]], errors='coerce').dt.year
                    except Exception:
                        pass

            # Filter to existing columns (post-derivation)
            existing_group_cols = [col for col in group_cols if col in df.columns]
            
            if not existing_group_cols:
                return df
            
            # Parse SELECT to detect explicit aggregations (AVG, SUM, COUNT)
            select_match = re.search(r"SELECT\s+(.*?)\s+FROM", sql_statement, re.IGNORECASE | re.DOTALL)
            select_clause = select_match.group(1) if select_match else ""
            agg_map = {}
            # Detect AVG(col)
            for m in re.finditer(r"AVG\(([^)]+)\)", select_clause, re.IGNORECASE):
                col = m.group(1).strip().split()[-1].split('.')[-1]
                if col in df.columns:
                    agg_map[col] = 'mean'
            # Detect SUM(col)
            for m in re.finditer(r"SUM\(([^)]+)\)", select_clause, re.IGNORECASE):
                col = m.group(1).strip().split()[-1].split('.')[-1]
                if col in df.columns:
                    agg_map[col] = 'sum'
            # Detect COUNT(*) or COUNT(col)
            count_detected = bool(re.search(r"COUNT\(\s*\*\s*\)", select_clause, re.IGNORECASE))
            count_cols = []
            for m in re.finditer(r"COUNT\(([^*)]+)\)", select_clause, re.IGNORECASE):
                col = m.group(1).strip().split()[-1].split('.')[-1]
                if col in df.columns:
                    count_cols.append(col)
            
            # Default aggregation for remaining numeric columns not in group
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in existing_group_cols:
                    continue
                if col not in agg_map:
                    # Default to sum for numeric columns
                    agg_map[col] = 'sum'
            
            result_df = df.groupby(existing_group_cols).agg(agg_map).reset_index() if agg_map else df.groupby(existing_group_cols).size().reset_index(name='count')
            
            # If COUNT(*) requested and not present, add a 'count' column
            if count_detected and 'count' not in result_df.columns:
                cnt = df.groupby(existing_group_cols).size().reset_index(name='count')
                result_df = result_df.merge(cnt, on=existing_group_cols, how='left')
            
            return result_df
            
        except Exception as e:
            logger.warning(f"Error applying GROUP BY: {str(e)}")
            return df
    
    def _apply_order_by(self, df: pd.DataFrame, sql_statement: str) -> pd.DataFrame:
        """Apply ORDER BY clause (simplified)."""
        
        try:
            order_match = re.search(r"ORDER BY\s+([\w\s,]+?)(?:\s+(?:ASC|DESC))?\s*(?:LIMIT|$)", sql_statement, re.IGNORECASE)
            if not order_match:
                return df
            
            order_col = order_match.group(1).strip().split(',')[0].strip()
            
            if order_col in df.columns:
                ascending = "DESC" not in sql_statement.upper()
                df = df.sort_values(order_col, ascending=ascending)
            
            return df
            
        except Exception as e:
            logger.warning(f"Error applying ORDER BY: {str(e)}")
            return df
    
    def _apply_limit(self, df: pd.DataFrame, sql_statement: str) -> pd.DataFrame:
        """Apply LIMIT clause."""
        
        try:
            limit_match = re.search(r"LIMIT\s+(\d+)", sql_statement, re.IGNORECASE)
            if limit_match:
                limit = int(limit_match.group(1))
                df = df.head(limit)
            
            return df
            
        except Exception as e:
            logger.warning(f"Error applying LIMIT: {str(e)}")
            return df
    
    def _apply_column_selection(self, df: pd.DataFrame, sql_statement: str) -> pd.DataFrame:
        """Apply SELECT column filtering (simplified)."""
        
        try:
            # Extract SELECT clause
            select_match = re.search(r"SELECT\s+(.*?)\s+FROM", sql_statement, re.IGNORECASE | re.DOTALL)
            if not select_match:
                return df
            
            select_clause = select_match.group(1).strip()
            
            # If SELECT *, return all columns
            if '*' in select_clause:
                return df
            
            # Parse selected columns (simplified)
            selected_cols = []
            for col_expr in select_clause.split(','):
                col_expr = col_expr.strip()
                
                # Handle basic column names and aliases
                if ' AS ' in col_expr.upper():
                    parts = col_expr.upper().split(' AS ')
                    alias = parts[1].strip()
                    selected_cols.append(alias)
                else:
                    # Remove table prefixes and functions
                    clean_col = re.sub(r'.*\.', '', col_expr)
                    clean_col = re.sub(r'\w+\((.*)\)', r'\1', clean_col)
                    selected_cols.append(clean_col.strip())
            
            # Filter to existing columns
            existing_cols = [col for col in selected_cols if col in df.columns]
            
            if existing_cols:
                return df[existing_cols]
            else:
                return df
            
        except Exception as e:
            logger.warning(f"Error applying column selection: {str(e)}")
            return df
    
    def _apply_transformations(self, df: pd.DataFrame, table_name: str, 
                             user_query: str, options: Dict[str, Any]) -> pd.DataFrame:
        """Apply data transformations to create user-friendly output."""
        
        transformed_df = df.copy()
        self.transformation_log = []
        
        try:
            # 1. Standardize column names
            transformed_df = self._standardize_column_names(transformed_df)
            
            # 2. Apply Philippine context enhancements
            transformed_df = self._apply_philippine_context(transformed_df)
            
            # 3. Format data types appropriately
            transformed_df = self._format_data_types(transformed_df)
            
            # 4. Add calculated columns if beneficial
            transformed_df = self._add_calculated_columns(transformed_df, user_query)
            
            # 5. Apply user-friendly formatting
            transformed_df = self._apply_user_formatting(transformed_df)
            
            # 6. Sort data meaningfully
            transformed_df = self._apply_meaningful_sorting(transformed_df, user_query)
            
            # 7. Limit output size for user experience
            transformed_df = self._limit_output_size(transformed_df, options)
            
            return transformed_df
            
        except Exception as e:
            logger.error(f"Error in data transformations: {str(e)}")
            return df  # Return original data if transformations fail
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to be user-friendly."""
        
        column_mapping = {}
        
        for col in df.columns:
            # Convert to title case and replace underscores
            new_name = col.replace('_', ' ').title()
            
            # Handle common abbreviations
            new_name = new_name.replace('Gdp', 'GDP')
            new_name = new_name.replace('Sdg', 'SDG') 
            new_name = new_name.replace('Ph ', 'Philippine ')
            new_name = new_name.replace('Ncr', 'NCR')
            
            if new_name != col:
                column_mapping[col] = new_name
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
            self.transformation_log.append(f"Standardized {len(column_mapping)} column names")
        
        return df
    
    def _apply_philippine_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Philippine context enhancements."""
        
        # Expand region codes to full names
        region_cols = [col for col in df.columns if 'region' in col.lower()]
        
        for col in region_cols:
            if col in df.columns and df[col].dtype == 'object':
                # Map region codes to full names
                region_mapping = {}
                for code, name in self.philippine_context["regions"].items():
                    if code in df[col].values:
                        region_mapping[code] = f"{name} ({code})"
                
                if region_mapping:
                    df[f"{col}_Full_Name"] = df[col].map(region_mapping).fillna(df[col])
                    self.transformation_log.append(f"Added full region names for {col}")
        
        # Add island group classification
        if any('region' in col.lower() for col in df.columns):
            region_col = next(col for col in df.columns if 'region' in col.lower())
            
            def get_island_group(region):
                for island, regions in self.philippine_context["island_groups"].items():
                    if region in regions:
                        return island
                return "Unknown"
            
            df['Island_Group'] = df[region_col].apply(get_island_group)
            self.transformation_log.append("Added island group classification")
        
        return df
    
    def _format_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format data types appropriately for display."""
        
        # Format currency columns
        currency_cols = [col for col in df.columns if any(term in col.lower() 
                        for term in ['income', 'expenditure', 'cost', 'price', 'gdp', 'salary'])]
        
        for col in currency_cols:
            if df[col].dtype in ['int64', 'float64']:
                # Format as Philippine Peso
                df[f"{col}_Formatted"] = df[col].apply(lambda x: f"₱{x:,.2f}" if pd.notna(x) else "N/A")
                self.transformation_log.append(f"Formatted {col} as currency")
        
        # Format percentage columns
        pct_cols = [col for col in df.columns if any(term in col.lower() 
                   for term in ['rate', 'percent', 'ratio']) and df[col].dtype in ['int64', 'float64']]
        
        for col in pct_cols:
            # Assume values are already in percentage form
            df[f"{col}_Formatted"] = df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
            self.transformation_log.append(f"Formatted {col} as percentage")
        
        # Format date columns
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        for col in date_cols:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                    self.transformation_log.append(f"Converted {col} to datetime")
                except:
                    pass
        
        return df
    
    def _add_calculated_columns(self, df: pd.DataFrame, user_query: str) -> pd.DataFrame:
        """Add calculated columns that might be useful for analysis."""
        
        # Add calculated columns based on available data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Growth rate calculations if year data is available
        if 'year' in df.columns and len(numeric_cols) > 0:
            df_sorted = df.sort_values('year')
            
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if col != 'year':
                    growth_col = f"{col}_Growth_Rate"
                    df_sorted[growth_col] = df_sorted.groupby(['region'] if 'region' in df.columns else [True])[col].pct_change() * 100
                    self.transformation_log.append(f"Added growth rate calculation for {col}")
            
            df = df_sorted
        
        # Add ranking columns for comparative analysis
        if "compare" in user_query.lower() or "rank" in user_query.lower():
            for col in numeric_cols[:2]:  # Rank top 2 numeric columns
                if col != 'year':
                    rank_col = f"{col}_Rank"
                    df[rank_col] = df[col].rank(method='dense', ascending=False)
                    self.transformation_log.append(f"Added ranking for {col}")
        
        return df
    
    def _apply_user_formatting(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply final user-friendly formatting."""
        
        # Round numeric columns to reasonable precision
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].max() > 1000:
                df[col] = df[col].round(0)  # Round large numbers to whole numbers
            else:
                df[col] = df[col].round(2)  # Round smaller numbers to 2 decimal places
        
        return df
    
    def _apply_meaningful_sorting(self, df: pd.DataFrame, user_query: str) -> pd.DataFrame:
        """Sort data in a meaningful way based on the user query."""
        
        query_lower = user_query.lower()
        
        # Sort by date if available for trend queries
        if any(word in query_lower for word in ['trend', 'over time', 'recent', 'latest']):
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'year' in col.lower()]
            if date_cols:
                df = df.sort_values(date_cols[0], ascending=False)
                self.transformation_log.append(f"Sorted by {date_cols[0]} (newest first)")
                return df
        
        # Sort by top values for comparison queries
        if any(word in query_lower for word in ['top', 'highest', 'best', 'most']):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                sort_col = numeric_cols[0]
                df = df.sort_values(sort_col, ascending=False)
                self.transformation_log.append(f"Sorted by {sort_col} (highest first)")
                return df
        
        # Sort by region for geographic queries
        if 'region' in df.columns:
            df = df.sort_values('region')
            self.transformation_log.append("Sorted by region")
        
        return df
    
    def _limit_output_size(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """Limit output size for optimal user experience."""
        
        max_rows = options.get('max_rows', 500)
        max_cols = options.get('max_columns', 15)
        
        # Limit rows
        if len(df) > max_rows:
            df = df.head(max_rows)
            self.transformation_log.append(f"Limited output to {max_rows} rows")
        
        # Limit columns (prioritize meaningful columns)
        if len(df.columns) > max_cols:
            # Prioritize non-formatted columns
            priority_cols = [col for col in df.columns if not col.endswith('_Formatted')]
            priority_cols = priority_cols[:max_cols-2]  # Leave room for a couple formatted columns
            
            # Add some formatted columns if available
            formatted_cols = [col for col in df.columns if col.endswith('_Formatted')][:2]
            
            final_cols = priority_cols + formatted_cols
            df = df[final_cols]
            self.transformation_log.append(f"Limited output to {len(final_cols)} columns")
        
        return df
    
    def _generate_data_insights(self, df: pd.DataFrame, user_query: str) -> List[Dict[str, Any]]:
        """Generate insights about the transformed data."""
        
        insights = []
        
        try:
            # 1) Overview
            insights.append({
                "type": "data_overview",
                "insight": f"Dataset contains {len(df):,} rows and {len(df.columns)} columns",
                "priority": "info"
            })

            # Helper: choose a meaningful numeric column (exclude date/year/id/code)
            num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
            def is_bad_numeric(name: str) -> bool:
                n = name.lower()
                return any(tok in n for tok in ["year","date","month","quarter","_id","id","code","key","index"]) or n.endswith('_id')
            num_cols = [c for c in num_cols_all if not is_bad_numeric(c)]

            # 2) Numeric summary on a meaningful metric
            if num_cols:
                main = num_cols[0]
                mean_val = float(df[main].dropna().mean()) if not df[main].dropna().empty else None
                median_val = float(df[main].dropna().median()) if not df[main].dropna().empty else None
                if mean_val is not None and median_val is not None:
                    insights.append({
                        "type": "statistical_summary",
                        "insight": f"Average {main}: {mean_val:,.2f}, Median: {median_val:,.2f}",
                        "priority": "medium"
                    })

            # 3) Missing data
            total_cells = max(len(df) * max(len(df.columns), 1), 1)
            missing_data = int(df.isnull().sum().sum())
            if missing_data > 0:
                missing_pct = (missing_data / total_cells) * 100
                insights.append({
                    "type": "data_quality",
                    "insight": f"Dataset has {missing_pct:.1f}% missing values",
                    "priority": "high" if missing_pct > 10 else "low"
                })

            # 4) Temporal coverage
            date_cols = [col for col in df.columns if 'year' in col.lower() or 'date' in col.lower()]
            if date_cols:
                date_col = date_cols[0]
                # If it's a year-like numeric/string column
                if 'year' in date_col.lower():
                    try:
                        yr_min = int(pd.to_numeric(df[date_col], errors='coerce').min())
                        yr_max = int(pd.to_numeric(df[date_col], errors='coerce').max())
                        n_years = int(pd.to_numeric(df[date_col], errors='coerce').nunique())
                        insights.append({
                            "type": "temporal",
                            "insight": f"Data spans {yr_min}-{yr_max} ({n_years} years)",
                            "priority": "medium"
                        })
                    except Exception:
                        pass

            # 5) Simple geography note, if present
            if 'Island_Group' in df.columns:
                island_counts = df['Island_Group'].value_counts()
                if not island_counts.empty:
                    top_island = island_counts.index[0]
                    insights.append({
                        "type": "geographic",
                        "insight": f"Most data points are from {top_island} ({int(island_counts[top_island])} records)",
                        "priority": "medium"
                    })

        except Exception as e:
            logger.warning(f"Error generating insights: {str(e)}")
        
        return insights
    
    def _create_user_summary(self, df: pd.DataFrame, user_query: str, insights: List[Dict]) -> Dict[str, Any]:
        """Create a user-friendly summary of the results."""
        
        summary = {
            "query_fulfilled": user_query,
            "result_count": len(df),
            "column_count": len(df.columns),
            "key_findings": [],
            "data_period": "Unknown",
            "geographic_coverage": "Unknown"
        }
        
        try:
            # Extract key findings from insights
            high_priority_insights = [i for i in insights if i.get("priority") == "high"]
            medium_priority_insights = [i for i in insights if i.get("priority") == "medium"]
            
            summary["key_findings"] = [i["insight"] for i in high_priority_insights + medium_priority_insights[:3]]
            
            # Determine data period
            date_cols = [col for col in df.columns if 'year' in col.lower()]
            if date_cols:
                date_col = date_cols[0]
                summary["data_period"] = f"{df[date_col].min()}-{df[date_col].max()}"
            
            # Determine geographic coverage
            region_cols = [col for col in df.columns if 'region' in col.lower()]
            if region_cols:
                region_count = df[region_cols[0]].nunique()
                summary["geographic_coverage"] = f"{region_count} regions/areas"
            
        except Exception as e:
            logger.warning(f"Error creating user summary: {str(e)}")
        
        return summary
    
    def _get_transformation_log(self) -> List[str]:
        """Get log of transformations applied."""
        return getattr(self, 'transformation_log', [])
    
    def create_natural_language_summary(self, result: Dict[str, Any]) -> Optional[str]:
        """Generate natural language summary of the results using LLM."""
        
        if not self.client or result.get("error"):
            return None
        
        try:
            df = result["transformed_dataframe"]
            insights = result["insights"]
            user_query = result["user_query"]
            
            # Prepare context for LLM
            context = {
                "user_query": user_query,
                "row_count": len(df),
                "column_names": list(df.columns),
                "insights": [i["insight"] for i in insights],
                "sample_data": df.head(3).to_dict('records') if len(df) > 0 else []
            }
            
            prompt = f"""Based on the following data analysis results for a Philippine dataset, create a concise, user-friendly summary:

User Query: {user_query}
Results: {json.dumps(context, indent=2, default=str)}

Provide a 2-3 sentence summary that:
1. Confirms what data was found
2. Highlights key insights or patterns  
3. Mentions the Philippine context where relevant

Keep it conversational and accessible."""

            response = self.client.chat.completions.create(
                model=get_model_name("gpt-4"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"Could not generate natural language summary: {str(e)}")
            return None
