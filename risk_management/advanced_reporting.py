"""
Advanced Reporting System with Custom Report Builder
Provides comprehensive reporting capabilities with drag-and-drop report builder, 
scheduled reports, interactive dashboards, and multi-format export
"""

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import io
import base64

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

from jinja2 import Template, Environment, FileSystemLoader
import weasyprint
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.chart import LineChart, BarChart, Reference
from openpyxl.drawing.image import Image as OpenpyxlImage

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import redis
from celery import Celery

Base = declarative_base()

class ReportType(str, Enum):
    RISK_SUMMARY = "risk_summary"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    PORTFOLIO_OVERVIEW = "portfolio_overview"
    COMPLIANCE_REPORT = "compliance_report"
    CUSTOM_DASHBOARD = "custom_dashboard"
    REGULATORY_FILING = "regulatory_filing"
    EXECUTIVE_SUMMARY = "executive_summary"

class OutputFormat(str, Enum):
    PDF = "pdf"
    EXCEL = "excel"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    POWERPOINT = "powerpoint"
    INTERACTIVE_HTML = "interactive_html"

class ScheduleFrequency(str, Enum):
    REAL_TIME = "real_time"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ON_DEMAND = "on_demand"

class VisualizationType(str, Enum):
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    HEATMAP = "heatmap"
    SCATTER_PLOT = "scatter_plot"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    CANDLESTICK = "candlestick"
    GAUGE = "gauge"
    TABLE = "table"
    KPI_CARD = "kpi_card"

@dataclass
class ReportWidget:
    id: str
    title: str
    visualization_type: VisualizationType
    data_source: str
    filters: Dict[str, Any]
    styling: Dict[str, Any]
    position: Dict[str, int]  # x, y, width, height
    refresh_interval: int = 300  # seconds

@dataclass
class ReportTemplate:
    id: str
    name: str
    description: str
    report_type: ReportType
    widgets: List[ReportWidget]
    layout: Dict[str, Any]
    styling: Dict[str, Any]
    created_by: str
    created_at: datetime
    is_public: bool = False

class ReportJob(Base):
    __tablename__ = "report_jobs"
    
    id = Column(Integer, primary_key=True)
    job_id = Column(String(64), unique=True)
    template_id = Column(String(64))
    user_id = Column(String(64))
    schedule_frequency = Column(String(20))
    output_format = Column(String(20))
    parameters = Column(JSON)
    next_run = Column(DateTime)
    last_run = Column(DateTime)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
class ReportOutput(Base):
    __tablename__ = "report_outputs"
    
    id = Column(Integer, primary_key=True)
    report_id = Column(String(64), unique=True)
    job_id = Column(String(64))
    file_path = Column(String(500))
    file_size = Column(Integer)
    output_format = Column(String(20))
    generation_time = Column(Float)
    status = Column(String(20))  # generating, completed, failed
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class DataSource:
    def __init__(self, name: str, connection_string: str = None):
        self.name = name
        self.connection_string = connection_string
    
    async def get_data(self, query: str, params: Dict = None) -> pd.DataFrame:
        # Mock data generation for demonstration
        if "portfolio_performance" in query.lower():
            dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
            data = {
                'date': dates,
                'portfolio_value': np.cumsum(np.random.randn(len(dates)) * 1000) + 1000000,
                'daily_return': np.random.randn(len(dates)) * 0.02,
                'benchmark_return': np.random.randn(len(dates)) * 0.015
            }
            return pd.DataFrame(data)
        
        elif "risk_metrics" in query.lower():
            return pd.DataFrame({
                'metric': ['VaR 95%', 'VaR 99%', 'Expected Shortfall', 'Maximum Drawdown', 'Sharpe Ratio'],
                'value': [-50000, -85000, -120000, -0.15, 1.25],
                'benchmark': [-45000, -78000, -110000, -0.12, 1.18]
            })
        
        elif "position_breakdown" in query.lower():
            return pd.DataFrame({
                'asset_class': ['Equities', 'Fixed Income', 'Alternatives', 'Cash'],
                'allocation': [0.65, 0.25, 0.08, 0.02],
                'value': [650000, 250000, 80000, 20000],
                'risk_contribution': [0.75, 0.15, 0.08, 0.02]
            })
        
        return pd.DataFrame()

class VisualizationEngine:
    def __init__(self):
        self.style_config = {
            'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'font_family': 'Arial',
            'font_size': 12,
            'figure_size': (12, 8)
        }
    
    def create_line_chart(self, data: pd.DataFrame, config: Dict) -> go.Figure:
        fig = go.Figure()
        
        x_col = config.get('x_column', data.columns[0])
        y_cols = config.get('y_columns', [data.columns[1]])
        
        for i, y_col in enumerate(y_cols):
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='lines',
                name=y_col,
                line=dict(color=self.style_config['color_palette'][i % len(self.style_config['color_palette'])])
            ))
        
        fig.update_layout(
            title=config.get('title', 'Line Chart'),
            xaxis_title=config.get('x_title', x_col),
            yaxis_title=config.get('y_title', 'Value'),
            font=dict(family=self.style_config['font_family'], size=self.style_config['font_size'])
        )
        
        return fig
    
    def create_bar_chart(self, data: pd.DataFrame, config: Dict) -> go.Figure:
        fig = go.Figure()
        
        x_col = config.get('x_column', data.columns[0])
        y_col = config.get('y_column', data.columns[1])
        
        fig.add_trace(go.Bar(
            x=data[x_col],
            y=data[y_col],
            marker_color=self.style_config['color_palette'][0]
        ))
        
        fig.update_layout(
            title=config.get('title', 'Bar Chart'),
            xaxis_title=config.get('x_title', x_col),
            yaxis_title=config.get('y_title', y_col),
            font=dict(family=self.style_config['font_family'], size=self.style_config['font_size'])
        )
        
        return fig
    
    def create_pie_chart(self, data: pd.DataFrame, config: Dict) -> go.Figure:
        labels_col = config.get('labels_column', data.columns[0])
        values_col = config.get('values_column', data.columns[1])
        
        fig = go.Figure(data=[go.Pie(
            labels=data[labels_col],
            values=data[values_col],
            hole=config.get('donut_size', 0)
        )])
        
        fig.update_layout(
            title=config.get('title', 'Pie Chart'),
            font=dict(family=self.style_config['font_family'], size=self.style_config['font_size'])
        )
        
        return fig
    
    def create_heatmap(self, data: pd.DataFrame, config: Dict) -> go.Figure:
        # Assume data is correlation matrix or similar
        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale='RdYlBu_r'
        ))
        
        fig.update_layout(
            title=config.get('title', 'Heatmap'),
            font=dict(family=self.style_config['font_family'], size=self.style_config['font_size'])
        )
        
        return fig
    
    def create_gauge(self, value: float, config: Dict) -> go.Figure:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': config.get('title', 'Gauge')},
            delta = {'reference': config.get('reference_value', 0)},
            gauge = {
                'axis': {'range': [None, config.get('max_value', 100)]},
                'bar': {'color': self.style_config['color_palette'][0]},
                'steps': [
                    {'range': [0, config.get('threshold_1', 50)], 'color': "lightgray"},
                    {'range': [config.get('threshold_1', 50), config.get('threshold_2', 80)], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': config.get('threshold_value', 90)
                }
            }
        ))
        
        return fig
    
    def create_table(self, data: pd.DataFrame, config: Dict) -> go.Figure:
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(data.columns),
                fill_color='paleturquoise',
                align='left',
                font=dict(size=self.style_config['font_size'])
            ),
            cells=dict(
                values=[data[col] for col in data.columns],
                fill_color='lavender',
                align='left',
                font=dict(size=self.style_config['font_size'])
            )
        )])
        
        fig.update_layout(
            title=config.get('title', 'Data Table'),
            font=dict(family=self.style_config['font_family'])
        )
        
        return fig

class ReportBuilder:
    def __init__(self):
        self.data_sources = {}
        self.viz_engine = VisualizationEngine()
        self.templates = {}
        
    def add_data_source(self, name: str, data_source: DataSource):
        self.data_sources[name] = data_source
    
    def create_template(self, template: ReportTemplate):
        self.templates[template.id] = template
    
    async def generate_report(self, template_id: str, parameters: Dict = None, output_format: OutputFormat = OutputFormat.HTML) -> str:
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        # Generate widgets
        widget_outputs = {}
        for widget in template.widgets:
            try:
                data_source = self.data_sources.get(widget.data_source)
                if not data_source:
                    continue
                
                # Apply filters
                query = self._build_query(widget.data_source, widget.filters, parameters)
                data = await data_source.get_data(query, parameters)
                
                # Create visualization
                viz_output = await self._create_visualization(widget, data)
                widget_outputs[widget.id] = viz_output
                
            except Exception as e:
                logging.error(f"Error generating widget {widget.id}: {e}")
                widget_outputs[widget.id] = f"Error: {str(e)}"
        
        # Generate final report
        return await self._render_report(template, widget_outputs, output_format)
    
    def _build_query(self, data_source: str, filters: Dict, parameters: Dict = None) -> str:
        # Build SQL or data query based on data source and filters
        base_queries = {
            "portfolio_db": "SELECT * FROM portfolio_performance",
            "risk_db": "SELECT * FROM risk_metrics",
            "trading_db": "SELECT * FROM position_breakdown"
        }
        
        return base_queries.get(data_source, "SELECT * FROM data")
    
    async def _create_visualization(self, widget: ReportWidget, data: pd.DataFrame) -> Union[str, go.Figure]:
        config = widget.styling
        config['title'] = widget.title
        
        if widget.visualization_type == VisualizationType.LINE_CHART:
            return self.viz_engine.create_line_chart(data, config)
        elif widget.visualization_type == VisualizationType.BAR_CHART:
            return self.viz_engine.create_bar_chart(data, config)
        elif widget.visualization_type == VisualizationType.PIE_CHART:
            return self.viz_engine.create_pie_chart(data, config)
        elif widget.visualization_type == VisualizationType.HEATMAP:
            return self.viz_engine.create_heatmap(data, config)
        elif widget.visualization_type == VisualizationType.TABLE:
            return self.viz_engine.create_table(data, config)
        elif widget.visualization_type == VisualizationType.GAUGE:
            value = data.iloc[0, 1] if not data.empty else 0
            return self.viz_engine.create_gauge(value, config)
        elif widget.visualization_type == VisualizationType.KPI_CARD:
            return self._create_kpi_card(data, config)
        
        return "Visualization not implemented"
    
    def _create_kpi_card(self, data: pd.DataFrame, config: Dict) -> str:
        if data.empty:
            return "<div class='kpi-card'>No data available</div>"
        
        value = data.iloc[0, 1] if len(data.columns) > 1 else data.iloc[0, 0]
        title = config.get('title', 'KPI')
        
        return f"""
        <div class="kpi-card" style="background: {config.get('background_color', '#f8f9fa')}; 
                                   border: 1px solid #dee2e6; border-radius: 8px; 
                                   padding: 20px; margin: 10px; text-align: center;">
            <h3 style="margin: 0; color: {config.get('title_color', '#333')};">{title}</h3>
            <div style="font-size: 2em; font-weight: bold; color: {config.get('value_color', '#007bff')}; 
                       margin: 10px 0;">{value:,.2f}</div>
            <p style="margin: 0; color: {config.get('subtitle_color', '#666')};">
                {config.get('subtitle', '')}
            </p>
        </div>
        """
    
    async def _render_report(self, template: ReportTemplate, widget_outputs: Dict, output_format: OutputFormat) -> str:
        if output_format == OutputFormat.HTML:
            return self._render_html_report(template, widget_outputs)
        elif output_format == OutputFormat.PDF:
            return await self._render_pdf_report(template, widget_outputs)
        elif output_format == OutputFormat.EXCEL:
            return await self._render_excel_report(template, widget_outputs)
        elif output_format == OutputFormat.INTERACTIVE_HTML:
            return self._render_interactive_html_report(template, widget_outputs)
        
        return "Output format not supported"
    
    def _render_html_report(self, template: ReportTemplate, widget_outputs: Dict) -> str:
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ template.name }}</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }
                .widget { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .kpi-card { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; text-align: center; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ template.name }}</h1>
                <p>{{ template.description }}</p>
                <p>Generated on: {{ timestamp }}</p>
            </div>
            
            <div class="grid">
                {% for widget_id, output in widget_outputs.items() %}
                <div class="widget">
                    {{ output | safe }}
                </div>
                {% endfor %}
            </div>
        </body>
        </html>
        """
        
        env = Environment()
        tmpl = env.from_string(html_template)
        
        # Convert Plotly figures to HTML
        processed_outputs = {}
        for widget_id, output in widget_outputs.items():
            if isinstance(output, go.Figure):
                processed_outputs[widget_id] = pyo.plot(output, output_type='div', include_plotlyjs=True)
            else:
                processed_outputs[widget_id] = str(output)
        
        return tmpl.render(
            template=template,
            widget_outputs=processed_outputs,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    async def _render_pdf_report(self, template: ReportTemplate, widget_outputs: Dict) -> str:
        # Generate HTML first
        html_content = self._render_html_report(template, widget_outputs)
        
        # Convert to PDF using weasyprint
        output_file = f"/tmp/report_{uuid.uuid4().hex}.pdf"
        
        try:
            weasyprint.HTML(string=html_content).write_pdf(output_file)
            return output_file
        except Exception as e:
            logging.error(f"PDF generation failed: {e}")
            return "PDF generation failed"
    
    async def _render_excel_report(self, template: ReportTemplate, widget_outputs: Dict) -> str:
        output_file = f"/tmp/report_{uuid.uuid4().hex}.xlsx"
        
        try:
            wb = Workbook()
            ws = wb.active
            ws.title = "Report Summary"
            
            # Add header
            ws['A1'] = template.name
            ws['A1'].font = Font(size=16, bold=True)
            ws['A2'] = template.description
            ws['A3'] = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Add widgets (simplified for Excel)
            row = 5
            for widget_id, output in widget_outputs.items():
                ws[f'A{row}'] = f"Widget: {widget_id}"
                ws[f'A{row}'].font = Font(bold=True)
                row += 1
                
                # Add simple data representation
                if isinstance(output, str) and "kpi-card" in output:
                    ws[f'A{row}'] = "KPI Data"
                elif isinstance(output, go.Figure):
                    ws[f'A{row}'] = "Chart Data (see visualization in HTML report)"
                
                row += 3
            
            wb.save(output_file)
            return output_file
            
        except Exception as e:
            logging.error(f"Excel generation failed: {e}")
            return "Excel generation failed"
    
    def _render_interactive_html_report(self, template: ReportTemplate, widget_outputs: Dict) -> str:
        # Enhanced HTML with interactive features
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ template.name }} - Interactive</title>
            <meta charset="utf-8">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
                .header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .widget { background: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
                .controls { background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
                .refresh-btn { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
                .refresh-btn:hover { background: #0056b3; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ template.name }}</h1>
                <p>{{ template.description }}</p>
                <p>Generated on: {{ timestamp }}</p>
            </div>
            
            <div class="controls">
                <button class="refresh-btn" onclick="refreshReport()">Refresh Data</button>
                <button class="refresh-btn" onclick="exportReport()">Export PDF</button>
            </div>
            
            <div class="grid">
                {% for widget_id, output in widget_outputs.items() %}
                <div class="widget" id="widget-{{ widget_id }}">
                    {{ output | safe }}
                </div>
                {% endfor %}
            </div>
            
            <script>
                function refreshReport() {
                    location.reload();
                }
                
                function exportReport() {
                    window.print();
                }
                
                // Auto-refresh every 5 minutes
                setTimeout(function() {
                    refreshReport();
                }, 300000);
            </script>
        </body>
        </html>
        """
        
        env = Environment()
        tmpl = env.from_string(html_template)
        
        # Convert Plotly figures to HTML with full interactivity
        processed_outputs = {}
        for widget_id, output in widget_outputs.items():
            if isinstance(output, go.Figure):
                processed_outputs[widget_id] = pyo.plot(
                    output, 
                    output_type='div', 
                    include_plotlyjs=False,
                    config={'displayModeBar': True, 'responsive': True}
                )
            else:
                processed_outputs[widget_id] = str(output)
        
        return tmpl.render(
            template=template,
            widget_outputs=processed_outputs,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

class ReportScheduler:
    def __init__(self, celery_app: Celery, report_builder: ReportBuilder):
        self.celery = celery_app
        self.report_builder = report_builder
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    def schedule_report(self, job: ReportJob) -> str:
        task_id = str(uuid.uuid4())
        
        # Schedule with Celery
        if job.schedule_frequency == ScheduleFrequency.DAILY:
            from celery.schedules import crontab
            self.celery.conf.beat_schedule[f'report-{job.job_id}'] = {
                'task': 'generate_scheduled_report',
                'schedule': crontab(hour=9, minute=0),  # 9 AM daily
                'args': (job.job_id, task_id)
            }
        
        return task_id
    
    async def generate_scheduled_report(self, job_id: str, task_id: str):
        try:
            # Get job details from database
            # Generate report
            # Save to storage
            # Send notifications
            pass
        except Exception as e:
            logging.error(f"Scheduled report generation failed: {e}")

class ReportingSystem:
    def __init__(self):
        self.builder = ReportBuilder()
        self.templates = {}
        self.jobs = {}
        
        # Setup data sources
        self._setup_data_sources()
        self._create_default_templates()
    
    def _setup_data_sources(self):
        self.builder.add_data_source("portfolio_db", DataSource("portfolio_db"))
        self.builder.add_data_source("risk_db", DataSource("risk_db"))
        self.builder.add_data_source("trading_db", DataSource("trading_db"))
    
    def _create_default_templates(self):
        # Risk Summary Template
        risk_widgets = [
            ReportWidget(
                id="risk_overview",
                title="Risk Metrics Overview",
                visualization_type=VisualizationType.TABLE,
                data_source="risk_db",
                filters={},
                styling={"title": "Current Risk Metrics"},
                position={"x": 0, "y": 0, "width": 6, "height": 4}
            ),
            ReportWidget(
                id="var_gauge",
                title="Value at Risk (95%)",
                visualization_type=VisualizationType.GAUGE,
                data_source="risk_db",
                filters={"metric": "VaR 95%"},
                styling={"max_value": 100000, "threshold_value": 75000},
                position={"x": 6, "y": 0, "width": 6, "height": 4}
            ),
            ReportWidget(
                id="portfolio_allocation",
                title="Portfolio Allocation",
                visualization_type=VisualizationType.PIE_CHART,
                data_source="trading_db",
                filters={},
                styling={"donut_size": 0.3},
                position={"x": 0, "y": 4, "width": 12, "height": 6}
            )
        ]
        
        risk_template = ReportTemplate(
            id="risk_summary_template",
            name="Daily Risk Summary",
            description="Comprehensive daily risk analysis and portfolio overview",
            report_type=ReportType.RISK_SUMMARY,
            widgets=risk_widgets,
            layout={"columns": 12, "rows": 10},
            styling={"theme": "professional"},
            created_by="system",
            created_at=datetime.now()
        )
        
        self.builder.create_template(risk_template)
        self.templates[risk_template.id] = risk_template
    
    async def generate_report(self, template_id: str, output_format: OutputFormat = OutputFormat.HTML) -> str:
        return await self.builder.generate_report(template_id, output_format=output_format)
    
    def get_available_templates(self) -> List[Dict]:
        return [
            {
                "id": template.id,
                "name": template.name,
                "description": template.description,
                "type": template.report_type.value,
                "widgets_count": len(template.widgets)
            }
            for template in self.templates.values()
        ]

# Example usage and testing
async def main():
    # Initialize reporting system
    reporting = ReportingSystem()
    
    print("Advanced Reporting System initialized")
    print(f"Available templates: {len(reporting.templates)}")
    
    # List available templates
    templates = reporting.get_available_templates()
    for template in templates:
        print(f"- {template['name']}: {template['description']}")
    
    # Generate sample reports
    print("\nGenerating sample reports...")
    
    for output_format in [OutputFormat.HTML, OutputFormat.INTERACTIVE_HTML]:
        try:
            output_file = await reporting.generate_report(
                "risk_summary_template", 
                output_format=output_format
            )
            
            if output_format == OutputFormat.HTML:
                print(f"✓ HTML report generated successfully")
                print(f"  Sample content length: {len(output_file)} characters")
            else:
                print(f"✓ Interactive HTML report generated successfully")
                print(f"  Sample content length: {len(output_file)} characters")
                
        except Exception as e:
            print(f"✗ Error generating {output_format.value} report: {e}")
    
    # Test PDF and Excel generation
    try:
        pdf_file = await reporting.generate_report(
            "risk_summary_template", 
            output_format=OutputFormat.PDF
        )
        print(f"✓ PDF report generated: {pdf_file}")
    except Exception as e:
        print(f"✗ PDF generation failed: {e}")
    
    try:
        excel_file = await reporting.generate_report(
            "risk_summary_template", 
            output_format=OutputFormat.EXCEL
        )
        print(f"✓ Excel report generated: {excel_file}")
    except Exception as e:
        print(f"✗ Excel generation failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())