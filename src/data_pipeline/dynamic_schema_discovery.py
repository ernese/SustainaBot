"""
Dynamic Schema Discovery System
Automatically detects data files and generates schemas instead of hardcoding
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DynamicSchemaDiscovery:
    """Automatically discovers and builds schemas from actual data files."""
    
    def __init__(self, base_path: Optional[str] = None):
        # Resolve base path: arg > DATA_PATH env > repo data folder
        default_repo_data = Path(__file__).resolve().parents[2] / "data"
        self.base_path = Path(base_path or os.getenv("DATA_PATH", str(default_repo_data)))

        def resolve_layer(layer: str) -> Path:
            candidates = [
                self.base_path / f"final-spark-{layer}",
                self.base_path / "raw" / "final-spark-bronze" if layer == "bronze" else self.base_path,
                self.base_path / "processed" / "final-spark-silver" if layer == "silver" else self.base_path,
                self.base_path / "outputs" / "final-spark-gold" if layer == "gold" else self.base_path,
            ]
            for p in candidates:
                if p and isinstance(p, Path) and p.name.startswith("final-spark-") and p.exists():
                    return p
            # Fallback to the first conventional location even if missing
            return self.base_path / f"final-spark-{layer}"

        self.layer_paths = {
            "bronze": resolve_layer("bronze"),
            "silver": resolve_layer("silver"),
            "gold": resolve_layer("gold"),
        }
        
    def discover_all_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Discover schemas from all layers."""
        schemas = {}
        
        for layer_name, layer_path in self.layer_paths.items():
            if layer_path.exists():
                layer_schemas = self._discover_layer_schemas(layer_name, layer_path)
                schemas.update(layer_schemas)
                logger.info(f"Discovered {len(layer_schemas)} schemas in {layer_name} layer")
            else:
                logger.warning(f"{layer_name} layer not found at {layer_path}")
        
        return schemas
    
    def _discover_layer_schemas(self, layer_name: str, layer_path: Path) -> Dict[str, Dict[str, Any]]:
        """Discover schemas in a specific layer."""
        schemas = {}
        
        for item in layer_path.iterdir():
            if not item.is_dir() or item.name.startswith('.'):
                continue

            # Always include a rolled-up entry for the item itself
            schema = self._analyze_table_directory(layer_name, item)
            if schema:
                table_name = f"{layer_name}_{item.name}" if not item.name.startswith(layer_name) else item.name
                schemas[table_name] = schema

            # For bronze, also expose meaningful subdirectories as separate tables
            if layer_name == "bronze":
                for sub in item.iterdir():
                    if not sub.is_dir() or sub.name.startswith('.') or sub.name.startswith('_') or '=' in sub.name:
                        continue
                    sub_schema = self._analyze_table_directory(layer_name, sub)
                    if sub_schema:
                        base = item.name
                        base_name = base if base.startswith(f"{layer_name}_") else f"{layer_name}_{base}"
                        sub_table = f"{base_name}_{sub.name}"
                        sub_schema["layer"] = layer_name
                        schemas[sub_table] = sub_schema
        
        return schemas
    
    def _analyze_table_directory(self, layer_name: str, table_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a table directory to extract schema."""
        try:
            # Find data files
            data_file = self._find_sample_data_file(table_path)
            if not data_file:
                return None
            
            # Read sample data
            df = self._read_sample_data(data_file)
            if df is None:
                return None
            
            # Build schema
            schema = {
                "table_name": table_path.name,
                "layer": layer_name,
                "columns": self._extract_column_info(df),
                "row_count": len(df),
                "file_path": str(table_path),
                "data_file": str(data_file)
            }
            
            # Add primary key detection
            schema["primary_key"] = self._detect_primary_key(df)
            
            return schema
            
        except Exception as e:
            logger.error(f"Error analyzing {table_path}: {e}")
            return None
    
    def _find_sample_data_file(self, table_path: Path) -> Optional[Path]:
        """Find a data file to sample from."""
        
        # Look for parquet files first
        parquet_files = list(table_path.rglob("*.parquet"))
        if parquet_files:
            return parquet_files[0]
        
        # Look for CSV files
        csv_files = list(table_path.rglob("*.csv"))
        if csv_files:
            return csv_files[0]
        
        # Look in partitioned directories
        for subdir in table_path.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('_'):
                parquet_files = list(subdir.rglob("*.parquet"))
                if parquet_files:
                    return parquet_files[0]
        
        return None
    
    def _read_sample_data(self, file_path: Path, sample_size: int = 1000) -> Optional[pd.DataFrame]:
        """Read a sample of data from a file."""
        try:
            if file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            else:
                return None
            
            # Sample if too large
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            
            return df
            
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None
    
    def _extract_column_info(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Extract column information from dataframe."""
        columns = []
        
        for col_name in df.columns:
            col_type = self._pandas_to_sql_type(df[col_name].dtype)
            description = self._generate_column_description(col_name, df[col_name])
            
            columns.append({
                "name": col_name,
                "type": col_type,
                "description": description
            })
        
        return columns
    
    def _pandas_to_sql_type(self, pandas_dtype) -> str:
        """Convert pandas dtype to SQL type."""
        dtype_str = str(pandas_dtype)
        
        if 'int' in dtype_str:
            return 'INTEGER'
        elif 'float' in dtype_str:
            return 'FLOAT'
        elif 'datetime' in dtype_str:
            return 'TIMESTAMP'
        elif 'bool' in dtype_str:
            return 'BOOLEAN'
        else:
            return 'VARCHAR(255)'
    
    def _generate_column_description(self, col_name: str, series: pd.Series) -> str:
        """Generate description for a column based on its name and data."""
        col_lower = col_name.lower()
        
        # Common patterns
        if 'id' in col_lower and col_lower.endswith('id'):
            return f"Unique identifier for {col_name.replace('_id', '').replace('id', '')}"
        elif 'year' in col_lower:
            return "Calendar year"
        elif 'date' in col_lower:
            return "Date field"
        elif 'score' in col_lower:
            return f"Score or rating for {col_name.replace('_score', '')}"
        elif 'rate' in col_lower or 'pct' in col_lower or 'percent' in col_lower:
            return f"Percentage or rate for {col_name}"
        elif 'avg' in col_lower or 'average' in col_lower:
            return f"Average value of {col_name}"
        elif 'count' in col_lower:
            return f"Count of {col_name.replace('_count', '')}"
        elif 'value' in col_lower:
            return f"Value measurement for {col_name.replace('_value', '')}"
        else:
            return f"Data field for {col_name.replace('_', ' ')}"
    
    def _detect_primary_key(self, df: pd.DataFrame) -> List[str]:
        """Detect likely primary key columns."""
        candidates = []
        if len(df) == 0:
            return candidates
        
        for col_name in df.columns:
            col_lower = col_name.lower()
            try:
                unique_ratio = df[col_name].nunique(dropna=True) / max(len(df), 1)
            except Exception:
                unique_ratio = 0
            # High uniqueness + ID pattern = likely primary key
            if unique_ratio > 0.95 and ('id' in col_lower or 'key' in col_lower):
                candidates.append(col_name)
        
        return candidates[:1]  # Return first candidate
    
    def get_table_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Generate data mappings for the data transformer."""
        mappings = {}
        
        for layer_name, layer_path in self.layer_paths.items():
            if layer_path.exists():
                for item in layer_path.iterdir():
                    if not item.is_dir() or item.name.startswith('.'):
                        continue

                    # Rolled-up mapping for the item itself
                    table_name = f"{layer_name}_{item.name}" if not item.name.startswith(layer_name) else item.name
                    file_patterns = []
                    if list(item.rglob("*.parquet")):
                        file_patterns.append("*.parquet")
                    if list(item.rglob("*.csv")):
                        file_patterns.append("*.csv")
                    mappings[table_name] = {
                        "file_patterns": file_patterns,
                        "directories": [str(layer_path.name), item.name]
                    }

                    # For bronze, also expose meaningful subdirectories as separate tables
                    if layer_name == "bronze":
                        for sub in item.iterdir():
                            if not sub.is_dir() or sub.name.startswith('.') or sub.name.startswith('_') or '=' in sub.name:
                                continue
                            base = item.name
                            base_name = base if base.startswith(f"{layer_name}_") else f"{layer_name}_{base}"
                            sub_table = f"{base_name}_{sub.name}"
                            sub_patterns = []
                            if list(sub.rglob("*.parquet")):
                                sub_patterns.append("*.parquet")
                            if list(sub.rglob("*.csv")):
                                sub_patterns.append("*.csv")
                            mappings[sub_table] = {
                                "file_patterns": sub_patterns,
                                "directories": [str(layer_path.name), item.name, sub.name]
                            }
        
        return mappings


def get_dynamic_schemas() -> Dict[str, Dict[str, Any]]:
    """Convenience function to get all discovered schemas."""
    discovery = DynamicSchemaDiscovery()
    return discovery.discover_all_schemas()


def get_dynamic_mappings() -> Dict[str, Dict[str, Any]]:
    """Convenience function to get all data mappings."""
    discovery = DynamicSchemaDiscovery()
    return discovery.get_table_mappings()


if __name__ == "__main__":
    # Test the discovery system
    logging.basicConfig(level=logging.INFO)
    
    discovery = DynamicSchemaDiscovery()
    schemas = discovery.discover_all_schemas()
    
    print(f"Discovered {len(schemas)} tables:")
    for table_name, schema in schemas.items():
        print(f"  {table_name}: {len(schema['columns'])} columns, {schema['row_count']} rows")
