"""
Visualization Generator for SustainaBot
Creates interactive charts and dashboards for Philippine data analysis
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import streamlit as st
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime
import json
import base64
import io

logger = logging.getLogger(__name__)

class SustainaBotVisualizer:
    """
    Advanced visualization generator for Philippine sustainability data
    """
    
    def __init__(self):
        self.philippine_colors = {
            "primary": "#0066CC",      # Philippine Blue
            "secondary": "#FF6B35",    # Philippine Red/Orange
            "success": "#28A745",      # Green for positive indicators
            "warning": "#FFC107",      # Yellow for caution
            "danger": "#DC3545",       # Red for critical issues
            "info": "#17A2B8",         # Light blue for information
            "regions": px.colors.qualitative.Set3,  # For regional comparisons
            "climate": ["#FF6B35", "#FFA500", "#FFD700", "#90EE90", "#4169E1", "#8A2BE2"],  # Climate zones
            "sdg": px.colors.qualitative.Plotly  # For SDG indicators
        }
        
        self.chart_templates = {
            "default": "plotly_white",
            "dark": "plotly_dark", 
            "minimal": "simple_white"
        }
    
    def create_dashboard(self, data: Dict[str, Any], dashboard_type: str = "executive") -> go.Figure:
        """
        Create comprehensive dashboard based on analysis results
        """
        if dashboard_type == "executive":
            return self._create_executive_dashboard(data)
        elif dashboard_type == "climate":
            return self._create_climate_dashboard(data)
        elif dashboard_type == "economic":
            return self._create_economic_dashboard(data)
        elif dashboard_type == "sustainability":
            return self._create_sustainability_dashboard(data)
        else:
            return self._create_general_dashboard(data)
    
    def _create_executive_dashboard(self, data: Dict[str, Any]) -> go.Figure:
        """Create executive-level dashboard with key metrics"""
        
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                "Overall Performance", "Regional Comparison", "Climate Trends",
                "Economic Indicators", "SDG Progress", "Priority Areas",
                "Risk Assessment", "Action Items Status", "Forecast Summary"
            ),
            specs=[
                [{"type": "indicator"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "sunburst"}, {"type": "bar"}],
                [{"type": "heatmap"}, {"type": "table"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Overall Performance (KPI Gauge)
        overall_score = data.get("composite_score", 75)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=overall_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Sustainability Score"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': self._get_score_color(overall_score)},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgray"},
                           {'range': [50, 80], 'color': "lightyellow"},
                           {'range': [80, 100], 'color': "lightgreen"}
                       ],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ),
            row=1, col=1
        )
        
        # Add other dashboard components
        self._add_regional_comparison(fig, data, row=1, col=2)
        self._add_climate_trends(fig, data, row=1, col=3)
        self._add_economic_indicators(fig, data, row=2, col=1)
        self._add_sdg_progress(fig, data, row=2, col=2)
        self._add_priority_areas(fig, data, row=2, col=3)
        self._add_risk_heatmap(fig, data, row=3, col=1)
        
        fig.update_layout(
            height=1200,
            title_text="SustainaBot Executive Dashboard",
            title_x=0.5,
            template=self.chart_templates["default"],
            showlegend=False
        )
        
        return fig
    
    def create_climate_visualization(self, df: pd.DataFrame, chart_type: str = "auto") -> go.Figure:
        """Create climate-specific visualizations"""
        
        if chart_type == "auto":
            chart_type = self._auto_detect_climate_chart(df)
        
        if chart_type == "temperature_trends":
            return self._create_temperature_trends(df)
        elif chart_type == "rainfall_patterns":
            return self._create_rainfall_patterns(df)
        elif chart_type == "extreme_events":
            return self._create_extreme_events_chart(df)
        elif chart_type == "regional_climate":
            return self._create_regional_climate_comparison(df)
        else:
            return self._create_general_climate_overview(df)
    
    def create_economic_visualization(self, df: pd.DataFrame, chart_type: str = "auto") -> go.Figure:
        """Create economic-specific visualizations"""
        
        if chart_type == "auto":
            chart_type = self._auto_detect_economic_chart(df)
        
        if chart_type == "gdp_trends":
            return self._create_gdp_trends(df)
        elif chart_type == "regional_economy":
            return self._create_regional_economic_comparison(df)
        elif chart_type == "sector_analysis":
            return self._create_sector_analysis(df)
        elif chart_type == "poverty_indicators":
            return self._create_poverty_indicators(df)
        else:
            return self._create_general_economic_overview(df)
    
    def create_sustainability_visualization(self, df: pd.DataFrame, chart_type: str = "auto") -> go.Figure:
        """Create sustainability-specific visualizations"""
        
        if chart_type == "auto":
            chart_type = self._auto_detect_sustainability_chart(df)
        
        if chart_type == "sdg_scorecard":
            return self._create_sdg_scorecard(df)
        elif chart_type == "sustainability_trends":
            return self._create_sustainability_trends(df)
        elif chart_type == "carbon_footprint":
            return self._create_carbon_footprint_chart(df)
        elif chart_type == "energy_mix":
            return self._create_energy_mix_chart(df)
        else:
            return self._create_general_sustainability_overview(df)
    
    def create_comparison_chart(self, data: Dict[str, pd.DataFrame], comparison_type: str) -> go.Figure:
        """Create comparison charts across different dimensions"""
        
        if comparison_type == "regional":
            return self._create_multi_regional_comparison(data)
        elif comparison_type == "temporal":
            return self._create_temporal_comparison(data)
        elif comparison_type == "sectoral":
            return self._create_sectoral_comparison(data)
        else:
            return self._create_general_comparison(data)
    
    def create_correlation_matrix(self, df: pd.DataFrame, focus_variables: Optional[List[str]] = None) -> go.Figure:
        """Create interactive correlation matrix"""
        
        # Select numeric columns
        if focus_variables:
            numeric_cols = [col for col in focus_variables if col in df.columns and df[col].dtype in [np.float64, np.int64]]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return self._create_no_data_chart("Insufficient numeric data for correlation analysis")
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            title="Variable Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto",
            labels=dict(color="Correlation")
        )
        
        fig.update_layout(
            template=self.chart_templates["default"],
            height=600,
            title_x=0.5
        )
        
        return fig
    
    def create_time_series_chart(self, df: pd.DataFrame, date_col: str, value_cols: List[str],
                                title: str = "Time Series Analysis") -> go.Figure:
        """Create interactive time series visualization"""
        
        fig = go.Figure()
        
        # Add traces for each value column
        colors = px.colors.qualitative.Set1
        
        for i, col in enumerate(value_cols):
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df[date_col],
                    y=df[col],
                    mode='lines+markers',
                    name=col,
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f"{col}: %{{y:.2f}}<br>Date: %{{x}}<extra></extra>"
                ))
        
        fig.update_layout(
            title=title,
            title_x=0.5,
            xaxis_title="Date",
            yaxis_title="Value",
            template=self.chart_templates["default"],
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def create_geographic_map(self, df: pd.DataFrame, location_col: str, value_col: str,
                             map_type: str = "choropleth") -> go.Figure:
        """Create Philippine geographic visualizations"""
        
        # Philippine region mapping (simplified for demo)
        philippines_regions = {
            "NCR": {"lat": 14.6042, "lon": 120.9822},
            "CAR": {"lat": 16.4023, "lon": 120.5960},
            "Region I": {"lat": 15.8500, "lon": 120.2833},
            "Region II": {"lat": 17.6129, "lon": 121.7270},
            "Region III": {"lat": 15.0794, "lon": 120.6200},
            # Add more regions as needed
        }
        
        if map_type == "choropleth":
            # For now, create a bubble map since we don't have shape files
            return self._create_bubble_map(df, location_col, value_col, philippines_regions)
        else:
            return self._create_bubble_map(df, location_col, value_col, philippines_regions)
    
    def _create_bubble_map(self, df: pd.DataFrame, location_col: str, value_col: str,
                          coordinates: Dict[str, Dict[str, float]]) -> go.Figure:
        """Create bubble map for Philippine regions"""
        
        # Merge data with coordinates
        map_data = []
        for _, row in df.iterrows():
            location = row[location_col]
            if location in coordinates:
                map_data.append({
                    'location': location,
                    'lat': coordinates[location]['lat'],
                    'lon': coordinates[location]['lon'],
                    'value': row[value_col],
                    'size': max(10, min(50, row[value_col] / df[value_col].max() * 50))
                })
        
        if not map_data:
            return self._create_no_data_chart("No location data available for mapping")
        
        map_df = pd.DataFrame(map_data)
        
        fig = px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="lon",
            size="size",
            color="value",
            hover_name="location",
            hover_data={"value": True, "lat": False, "lon": False, "size": False},
            mapbox_style="open-street-map",
            title=f"Geographic Distribution: {value_col}",
            height=600,
            zoom=5,
            center={"lat": 14.5995, "lon": 120.9842}  # Philippines center
        )
        
        fig.update_layout(
            template=self.chart_templates["default"],
            title_x=0.5
        )
        
        return fig
    
    def export_chart(self, fig: go.Figure, format: str = "png", filename: str = None) -> str:
        """Export chart in specified format"""
        
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"sustainabot_chart_{timestamp}"
            
            if format.lower() == "png":
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                with open(f"{filename}.png", "wb") as f:
                    f.write(img_bytes)
                return f"{filename}.png"
            
            elif format.lower() == "html":
                fig.write_html(f"{filename}.html")
                return f"{filename}.html"
            
            elif format.lower() == "pdf":
                img_bytes = fig.to_image(format="pdf", width=1200, height=800)
                with open(f"{filename}.pdf", "wb") as f:
                    f.write(img_bytes)
                return f"{filename}.pdf"
            
        except Exception as e:
            logger.error(f"Error exporting chart: {str(e)}")
            return ""
    
    def get_chart_recommendations(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Recommend appropriate chart types based on data characteristics"""
        
        recommendations = []
        
        # Data characteristics
        num_rows = len(df)
        num_cols = len(df.columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        date_cols = df.select_dtypes(include=['datetime64']).columns
        
        # Time series recommendation
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            recommendations.append({
                "chart_type": "time_series",
                "title": "Time Series Analysis",
                "description": "Track trends over time",
                "best_for": "Temporal patterns and forecasting"
            })
        
        # Correlation analysis
        if len(numeric_cols) > 2:
            recommendations.append({
                "chart_type": "correlation_matrix",
                "title": "Correlation Analysis",
                "description": "Understand relationships between variables",
                "best_for": "Identifying patterns and dependencies"
            })
        
        # Distribution analysis
        if len(numeric_cols) > 0:
            recommendations.append({
                "chart_type": "distribution",
                "title": "Distribution Analysis",
                "description": "Examine data distributions and outliers",
                "best_for": "Data quality assessment and outlier detection"
            })
        
        # Categorical comparison
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            recommendations.append({
                "chart_type": "comparison",
                "title": "Comparative Analysis",
                "description": "Compare performance across categories",
                "best_for": "Regional or sectoral comparisons"
            })
        
        return recommendations
    
    # Helper methods for specific chart types
    def _create_temperature_trends(self, df: pd.DataFrame) -> go.Figure:
        """Create temperature trend visualization"""
        temp_cols = [col for col in df.columns if 'temp' in col.lower()]
        date_cols = df.select_dtypes(include=['datetime64']).columns
        
        if not temp_cols or len(date_cols) == 0:
            return self._create_no_data_chart("No temperature trend data available")
        
        return self.create_time_series_chart(df, date_cols[0], temp_cols[:2], "Temperature Trends")
    
    def _create_rainfall_patterns(self, df: pd.DataFrame) -> go.Figure:
        """Create rainfall pattern visualization"""
        rain_cols = [col for col in df.columns if 'rain' in col.lower() or 'precipitation' in col.lower()]
        
        if not rain_cols:
            return self._create_no_data_chart("No rainfall data available")
        
        # Create monthly rainfall pattern
        fig = px.bar(
            x=range(1, 13),
            y=df[rain_cols[0]].values[:12] if len(df) >= 12 else df[rain_cols[0]].values,
            title="Monthly Rainfall Patterns",
            labels={"x": "Month", "y": "Rainfall (mm)"}
        )
        
        fig.update_layout(
            template=self.chart_templates["default"],
            title_x=0.5,
            height=400
        )
        
        return fig
    
    def _create_sdg_scorecard(self, df: pd.DataFrame) -> go.Figure:
        """Create SDG scorecard visualization"""
        sdg_cols = [col for col in df.columns if 'sdg' in col.lower()]
        
        if not sdg_cols:
            return self._create_no_data_chart("No SDG data available")
        
        # Create radar chart for SDG scores
        sdg_scores = [df[col].mean() if col in df.columns else 50 for col in sdg_cols]
        sdg_names = [col.replace('_', ' ').title() for col in sdg_cols]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=sdg_scores,
            theta=sdg_names,
            fill='toself',
            name='SDG Performance',
            line_color=self.philippine_colors["primary"]
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="SDG Performance Scorecard",
            title_x=0.5,
            template=self.chart_templates["default"]
        )
        
        return fig
    
    def _create_no_data_chart(self, message: str) -> go.Figure:
        """Create placeholder chart when no data is available"""
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5, y=0.5,
            text=message,
            showarrow=False,
            font=dict(size=16, color="gray"),
            xref="paper", yref="paper"
        )
        
        fig.update_layout(
            template=self.chart_templates["default"],
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        return fig
    
    def _get_score_color(self, score: float) -> str:
        """Get color based on score value"""
        if score >= 80:
            return self.philippine_colors["success"]
        elif score >= 60:
            return self.philippine_colors["warning"]
        else:
            return self.philippine_colors["danger"]
    
    def _auto_detect_climate_chart(self, df: pd.DataFrame) -> str:
        """Auto-detect best climate chart type"""
        temp_cols = [col for col in df.columns if 'temp' in col.lower()]
        rain_cols = [col for col in df.columns if 'rain' in col.lower()]
        
        if temp_cols:
            return "temperature_trends"
        elif rain_cols:
            return "rainfall_patterns"
        else:
            return "general_climate"
    
    def _auto_detect_economic_chart(self, df: pd.DataFrame) -> str:
        """Auto-detect best economic chart type"""
        gdp_cols = [col for col in df.columns if 'gdp' in col.lower()]
        
        if gdp_cols:
            return "gdp_trends"
        else:
            return "general_economic"
    
    def _auto_detect_sustainability_chart(self, df: pd.DataFrame) -> str:
        """Auto-detect best sustainability chart type"""
        sdg_cols = [col for col in df.columns if 'sdg' in col.lower()]
        
        if sdg_cols:
            return "sdg_scorecard"
        else:
            return "general_sustainability"
    
    # Dashboard component methods
    def _add_regional_comparison(self, fig: go.Figure, data: Dict[str, Any], row: int, col: int):
        """Add regional comparison to dashboard"""
        regions = ["NCR", "CALABARZON", "Central Luzon", "Western Visayas", "Central Visayas"]
        values = [85, 78, 72, 68, 74]  # Sample data
        
        fig.add_trace(
            go.Bar(x=regions, y=values, marker_color=self.philippine_colors["regions"]),
            row=row, col=col
        )
    
    def _add_climate_trends(self, fig: go.Figure, data: Dict[str, Any], row: int, col: int):
        """Add climate trends to dashboard"""
        years = list(range(2020, 2025))
        temps = [28.5, 29.1, 28.8, 29.3, 29.0]  # Sample data
        
        fig.add_trace(
            go.Scatter(x=years, y=temps, mode='lines+markers', name='Temperature'),
            row=row, col=col
        )
    
    def _add_economic_indicators(self, fig: go.Figure, data: Dict[str, Any], row: int, col: int):
        """Add economic indicators to dashboard"""
        indicators = ["GDP Growth", "Employment", "Inflation", "Poverty Rate"]
        values = [6.2, 92.1, 3.8, 18.1]  # Sample data
        
        fig.add_trace(
            go.Bar(x=indicators, y=values, marker_color=self.philippine_colors["secondary"]),
            row=row, col=col
        )
    
    def _add_sdg_progress(self, fig: go.Figure, data: Dict[str, Any], row: int, col: int):
        """Add SDG progress to dashboard"""
        sdg_data = {"On Track": 8, "Moderate": 6, "Lagging": 3}  # Sample data
        
        fig.add_trace(
            go.Sunburst(
                labels=list(sdg_data.keys()),
                values=list(sdg_data.values()),
                parents=[""] * len(sdg_data)
            ),
            row=row, col=col
        )
    
    def _add_priority_areas(self, fig: go.Figure, data: Dict[str, Any], row: int, col: int):
        """Add priority areas to dashboard"""
        priorities = ["Climate Action", "Clean Energy", "Economic Growth", "Education", "Health"]
        scores = [65, 58, 72, 68, 74]  # Sample data
        
        fig.add_trace(
            go.Bar(x=priorities, y=scores, marker_color=self.philippine_colors["warning"]),
            row=row, col=col
        )
    
    def _add_risk_heatmap(self, fig: go.Figure, data: Dict[str, Any], row: int, col: int):
        """Add risk assessment heatmap to dashboard"""
        risks = ["Climate", "Economic", "Social", "Political"]
        impact = ["High", "Medium", "Low"]
        risk_matrix = [[3, 2, 1], [2, 3, 1], [1, 2, 2], [1, 1, 3]]  # Sample data
        
        fig.add_trace(
            go.Heatmap(
                z=risk_matrix,
                x=impact,
                y=risks,
                colorscale="Reds",
                showscale=False
            ),
            row=row, col=col
        )
