import sys
import os
import json
import requests
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QFileDialog, QScrollArea, QFrame, QGridLayout,
                             QProgressBar, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
                             QMessageBox, QSplitter, QGroupBox, QTextEdit, QComboBox, QLineEdit)
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont, QColor, QPalette, QIcon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

# Configuration - Replace with your actual Poe API details
POE_API_CONFIG = {
    'base_url': 'https://api.poe.com/bot/',
    'api_key': 'your-poe-api-key-here',
    'bots': {
        'plant_health_analyzer': 'your-plant-health-bot-name',
        'sensor_data_analyzer': 'your-sensor-bot-name'
    }
}

# Health thresholds with detailed ranges
HEALTH_THRESHOLDS = {
    'soil_moisture': {'min': 40, 'max': 70, 'unit': '%', 'description': 'Soil water content percentage'},
    'ph_level': {'min': 6.0, 'max': 7.5, 'unit': 'pH', 'description': 'Soil acidity/alkalinity level'},
    'nitrogen': {'min': 20, 'max': 40, 'unit': 'ppm', 'description': 'Nitrogen content in soil'},
    'phosphorus': {'min': 5, 'max': 15, 'unit': 'ppm', 'description': 'Phosphorus content in soil'},
    'potassium': {'min': 30, 'max': 60, 'unit': 'ppm', 'description': 'Potassium content in soil'},
    'temperature': {'min': 18, 'max': 26, 'unit': '°C', 'description': 'Ambient temperature'},
    'humidity': {'min': 50, 'max': 80, 'unit': '%', 'description': 'Relative humidity level'},
    'light_intensity': {'min': 200, 'max': 800, 'unit': 'µmol/m²/s', 'description': 'Light intensity for photosynthesis'}
}

class MetricIndicator(QWidget):
    """Custom widget to display a metric with visual indicator"""
    def __init__(self, name, value, threshold, parent=None):
        super().__init__(parent)
        self.name = name
        self.value = value
        self.threshold = threshold
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Metric name
        self.name_label = QLabel(f"{name.replace('_', ' ').title()}: {value} {threshold['unit']}")
        self.name_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.layout.addWidget(self.name_label)
        
        # Progress bar showing value relative to threshold
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        
        # Calculate percentage for display (clamped to 0-100)
        min_val = threshold['min']
        max_val = threshold['max']
        range_val = max_val - min_val
        
        if range_val > 0:
            percentage = ((value - min_val) / range_val) * 100
            percentage = max(0, min(100, percentage))  # Clamp between 0-100
        else:
            percentage = 50  # Default if min == max
            
        self.progress_bar.setValue(int(percentage))
        
        # Set color based on value relative to threshold
        if value < min_val * 0.8 or value > max_val * 1.2:
            # Critical - red
            self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #e74c3c; }")
            self.status = "critical"
        elif value >= min_val and value <= max_val:
            # Normal - green
            self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #2ecc71; }")
            self.status = "normal"
        else:
            # Warning - yellow
            self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #f39c12; }")
            self.status = "warning"
            
        self.layout.addWidget(self.progress_bar)
        
        # Threshold range label
        range_label = QLabel(f"Normal range: {min_val} - {max_val} {threshold['unit']}")
        range_label.setFont(QFont("Arial", 8))
        self.layout.addWidget(range_label)

class AnalysisReport(QWidget):
    """Widget to display the detailed analysis report"""
    def __init__(self, analysis_data, parent=None):
        super().__init__(parent)
        self.analysis_data = analysis_data
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Report title
        title = QLabel("Plant Health Analysis Report")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Timestamp
        timestamp = QLabel(f"Analysis performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        timestamp.setFont(QFont("Arial", 10))
        timestamp.setAlignment(Qt.AlignCenter)
        layout.addWidget(timestamp)
        
        # Health score
        health_score = self.analysis_data.get('health_score', 0)
        score_label = QLabel(f"Overall Health Score: {health_score}/100")
        score_label.setFont(QFont("Arial", 12, QFont.Bold))
        
        # Color code based on health score
        if health_score >= 70:
            score_label.setStyleSheet("color: #2ecc71;")  # Green
        elif health_score >= 50:
            score_label.setStyleSheet("color: #f39c12;")  # Yellow
        else:
            score_label.setStyleSheet("color: #e74c3c;")  # Red
            
        layout.addWidget(score_label)
        
        # Metrics grid
        metrics_group = QGroupBox("Sensor Metrics Analysis")
        metrics_layout = QGridLayout()
        metrics_group.setLayout(metrics_layout)
        
        sensor_data = self.analysis_data.get('sensor_data', {})
        row, col = 0, 0
        for metric, value in sensor_data.items():
            if metric in HEALTH_THRESHOLDS:
                metric_widget = MetricIndicator(metric, value, HEALTH_THRESHOLDS[metric])
                metrics_layout.addWidget(metric_widget, row, col)
                col += 1
                if col > 1:  # 2 columns
                    col = 0
                    row += 1
                    
        layout.addWidget(metrics_group)
        
        # Visual analysis summary
        visual_group = QGroupBox("Visual Analysis")
        visual_layout = QVBoxLayout()
        visual_group.setLayout(visual_layout)
        
        visual_analysis = self.analysis_data.get('plant_analysis', {}).get('analysis', 'No visual analysis available.')
        analysis_text = QTextEdit()
        analysis_text.setPlainText(visual_analysis)
        analysis_text.setReadOnly(True)
        visual_layout.addWidget(analysis_text)
        
        layout.addWidget(visual_group)
        
        # Recommendations
        recommendations = self.analysis_data.get('plant_analysis', {}).get('recommendations', [])
        if recommendations:
            rec_group = QGroupBox("Recommendations")
            rec_layout = QVBoxLayout()
            rec_group.setLayout(rec_layout)
            
            for rec in recommendations:
                rec_label = QLabel(f"• {rec}")
                rec_label.setWordWrap(True)
                rec_layout.addWidget(rec_label)
                
            layout.addWidget(rec_group)
            
        # Issues detected
        issues = self.analysis_data.get('plant_analysis', {}).get('issues_detected', [])
        if issues:
            issues_group = QGroupBox("Issues Detected")
            issues_layout = QVBoxLayout()
            issues_group.setLayout(issues_layout)
            issues_group.setStyleSheet("QGroupBox { color: #e74c3c; font-weight: bold; }")
            
            for issue in issues:
                issue_label = QLabel(f"⚠️ {issue}")
                issue_label.setWordWrap(True)
                issue_label.setStyleSheet("color: #e74c3c;")
                issues_layout.addWidget(issue_label)
                
            layout.addWidget(issues_group)

class ImagePreview(QLabel):
    """Widget to display the plant image with drag and drop support"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Drag & drop plant image here\nor click to browse")
        self.setStyleSheet("border: 2px dashed #ccc; padding: 20px;")
        self.setMinimumSize(400, 300)
        
    def set_image(self, image_path):
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            # Scale the image to fit while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(self.width(), self.height(), 
                                         Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)
            self.image_path = image_path
        else:
            self.setText("Failed to load image")

class Dashboard(QMainWindow):
    """Main dashboard window"""
    def __init__(self):
        super().__init__()
        self.analysis_history = []
        self.current_analysis = None
        self.captures_folder = os.path.join(os.path.expanduser("~"), "captures")
        
        self.init_ui()
        self.load_settings()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Plant Health Dashboard")
        self.setGeometry(100, 100, 1400, 800)
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for image upload and controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)
        
        # Image preview with drag and drop
        self.image_preview = ImagePreview()
        self.image_preview.mousePressEvent = self.browse_image
        left_layout.addWidget(self.image_preview)
        
        # Controls section
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # Captures folder selection
        folder_layout = QHBoxLayout()
        folder_label = QLabel("Captures Folder:")
        self.folder_path = QLineEdit(self.captures_folder)
        browse_folder_btn = QPushButton("Browse")
        browse_folder_btn.clicked.connect(self.select_captures_folder)
        
        folder_layout.addWidget(folder_label)
        folder_layout.addWidget(self.folder_path)
        folder_layout.addWidget(browse_folder_btn)
        controls_layout.addLayout(folder_layout)
        
        # Load latest capture button
        self.load_latest_btn = QPushButton("Load Latest Capture")
        self.load_latest_btn.clicked.connect(self.load_latest_capture)
        controls_layout.addWidget(self.load_latest_btn)
        
        # Analyze button
        self.analyze_btn = QPushButton("Analyze Plant Health")
        self.analyze_btn.clicked.connect(self.analyze_plant)
        self.analyze_btn.setEnabled(False)
        controls_layout.addWidget(self.analyze_btn)
        
        left_layout.addWidget(controls_group)
        
        # History section
        history_group = QGroupBox("Recent Analyses")
        history_layout = QVBoxLayout(history_group)
        
        self.history_list = QTableWidget()
        self.history_list.setColumnCount(3)
        self.history_list.setHorizontalHeaderLabels(["Date", "Health Score", "Status"])
        self.history_list.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_list.setSelectionBehavior(QTableWidget.SelectRows)
        self.history_list.doubleClicked.connect(self.view_history_item)
        
        history_layout.addWidget(self.history_list)
        left_layout.addWidget(history_group)
        
        # Right panel for analysis results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Tab widget for different views
        self.tabs = QTabWidget()
        
        # Dashboard tab
        self.dashboard_tab = QWidget()
        self.setup_dashboard_tab()
        self.tabs.addTab(self.dashboard_tab, "Dashboard")
        
        # Detailed report tab
        self.report_tab = QWidget()
        self.report_layout = QVBoxLayout(self.report_tab)
        self.report_scroll = QScrollArea()
        self.report_scroll.setWidgetResizable(True)
        self.report_layout.addWidget(self.report_scroll)
        self.tabs.addTab(self.report_tab, "Detailed Report")
        
        right_layout.addWidget(self.tabs)
        
        # Add panels to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 1100])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def setup_dashboard_tab(self):
        """Set up the dashboard tab with overview widgets"""
        layout = QVBoxLayout(self.dashboard_tab)
        
        # Health score indicator
        health_group = QGroupBox("Overall Health")
        health_layout = QVBoxLayout(health_group)
        
        self.health_score = QLabel("No analysis yet")
        self.health_score.setFont(QFont("Arial", 24, QFont.Bold))
        self.health_score.setAlignment(Qt.AlignCenter)
        health_layout.addWidget(self.health_score)
        
        self.health_status = QLabel("Upload an image to begin analysis")
        self.health_status.setAlignment(Qt.AlignCenter)
        health_layout.addWidget(self.health_status)
        
        layout.addWidget(health_group)
        
        # Key metrics
        metrics_group = QGroupBox("Key Metrics")
        metrics_layout = QGridLayout(metrics_group)
        
        # Placeholder metrics - will be updated after analysis
        self.metric_widgets = {}
        important_metrics = ['soil_moisture', 'ph_level', 'temperature', 'humidity']
        
        for i, metric in enumerate(important_metrics):
            widget = QLabel("N/A")
            widget.setAlignment(Qt.AlignCenter)
            widget.setFrameStyle(QFrame.Box)
            metrics_layout.addWidget(QLabel(metric.replace('_', ' ').title()), i, 0)
            metrics_layout.addWidget(widget, i, 1)
            self.metric_widgets[metric] = widget
            
        layout.addWidget(metrics_group)
        
        # Alerts section
        alerts_group = QGroupBox("Alerts & Notifications")
        alerts_layout = QVBoxLayout(alerts_group)
        
        self.alerts_display = QTextEdit()
        self.alerts_display.setReadOnly(True)
        self.alerts_display.setPlainText("No alerts. Plant health appears normal.")
        alerts_layout.addWidget(self.alerts_display)
        
        layout.addWidget(alerts_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.pdf_btn = QPushButton("Generate PDF Report")
        self.pdf_btn.clicked.connect(self.generate_pdf_report)
        self.pdf_btn.setEnabled(False)
        button_layout.addWidget(self.pdf_btn)
        
        self.email_btn = QPushButton("Email Report")
        self.email_btn.clicked.connect(self.email_report)
        self.email_btn.setEnabled(False)
        button_layout.addWidget(self.email_btn)
        
        layout.addLayout(button_layout)
        
    def browse_image(self, event):
        """Open file dialog to select an image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Plant Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if file_path:
            self.image_preview.set_image(file_path)
            self.analyze_btn.setEnabled(True)
            
    def select_captures_folder(self):
        """Select the folder where plant images are captured"""
        folder = QFileDialog.getExistingDirectory(self, "Select Captures Folder", self.captures_folder)
        if folder:
            self.captures_folder = folder
            self.folder_path.setText(folder)
            
    def load_latest_capture(self):
        """Load the most recent image from the captures folder"""
        try:
            captures_path = Path(self.captures_folder)
            if not captures_path.exists():
                QMessageBox.warning(self, "Folder Not Found", 
                                  f"The captures folder does not exist: {self.captures_folder}")
                return
                
            # Find all image files
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(captures_path.glob(ext))
                
            if not image_files:
                QMessageBox.information(self, "No Images", 
                                      "No plant images found in the captures folder.")
                return
                
            # Get the most recent file
            latest_file = max(image_files, key=os.path.getctime)
            self.image_preview.set_image(str(latest_file))
            self.analyze_btn.setEnabled(True)
            self.statusBar().showMessage(f"Loaded latest capture: {latest_file.name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load latest capture: {str(e)}")
            
    def analyze_plant(self):
        """Send the image to AI bots for analysis"""
        if not hasattr(self.image_preview, 'image_path'):
            QMessageBox.warning(self, "No Image", "Please select an image first.")
            return
            
        self.statusBar().showMessage("Analyzing plant health...")
        self.analyze_btn.setEnabled(False)
        
        # Simulate API calls (replace with actual Poe API integration)
        # In a real implementation, you would make actual HTTP requests to the Poe API
        
        # For demonstration, we'll use mock data
        plant_analysis = {
            "analysis": "Plant exhibits healthy green foliage with good leaf structure. "
                       "Minor yellowing observed on lower leaves, possibly indicating "
                       "natural aging or slight nutrient deficiency.",
            "health_score": 82,
            "confidence": 0.89,
            "recommendations": [
                "Monitor nitrogen levels - slight deficiency indicated by lower leaf yellowing",
                "Maintain current watering schedule",
                "Consider light fertilization with balanced NPK",
                "Continue regular monitoring for pest activity"
            ],
            "issues_detected": [
                "Mild nitrogen deficiency (lower leaf yellowing)",
                "Slight leaf tip browning - possible over-fertilization"
            ],
            "disease_risk": "Low",
            "growth_stage": "Mature vegetative"
        }
        
        # Mock sensor data
        import random
        sensor_data = {
            "soil_moisture": round(random.uniform(45, 65), 1),
            "ph_level": round(random.uniform(6.2, 7.2), 1),
            "nitrogen": round(random.uniform(18, 35), 1),
            "phosphorus": round(random.uniform(6, 14), 1),
            "potassium": round(random.uniform(35, 55), 1),
            "temperature": round(random.uniform(20, 24), 1),
            "humidity": round(random.uniform(55, 75), 1),
            "light_intensity": round(random.uniform(300, 600), 0)
        }
        
        # Combine analysis results
        self.current_analysis = {
            "timestamp": datetime.now().isoformat(),
            "image_path": self.image_preview.image_path,
            "plant_analysis": plant_analysis,
            "sensor_data": sensor_data
        }
        
        # Add to history
        self.analysis_history.append(self.current_analysis)
        self.update_history_list()
        
        # Update UI with results
        self.update_dashboard()
        
        # Enable report buttons
        self.pdf_btn.setEnabled(True)
        self.email_btn.setEnabled(True)
        
        self.statusBar().showMessage("Analysis complete")
        
    def update_dashboard(self):
        """Update the dashboard with current analysis results"""
        if not self.current_analysis:
            return
            
        # Update health score
        health_score = self.current_analysis['plant_analysis']['health_score']
        self.health_score.setText(f"{health_score}/100")
        
        # Set color based on health score
        if health_score >= 70:
            self.health_score.setStyleSheet("color: #2ecc71;")  # Green
            self.health_status.setText("Healthy")
        elif health_score >= 50:
            self.health_score.setStyleSheet("color: #f39c12;")  # Yellow
            self.health_status.setText("Needs Attention")
        else:
            self.health_score.setStyleSheet("color: #e74c3c;")  # Red
            self.health_status.setText("Critical")
            
        # Update key metrics
        sensor_data = self.current_analysis['sensor_data']
        for metric, widget in self.metric_widgets.items():
            if metric in sensor_data:
                value = sensor_data[metric]
                unit = HEALTH_THRESHOLDS[metric]['unit']
                widget.setText(f"{value} {unit}")
                
                # Color code based on thresholds
                threshold = HEALTH_THRESHOLDS[metric]
                if value < threshold['min'] * 0.8 or value > threshold['max'] * 1.2:
                    widget.setStyleSheet("background-color: #e74c3c; color: white;")  # Red
                elif value >= threshold['min'] and value <= threshold['max']:
                    widget.setStyleSheet("background-color: #2ecc71; color: white;")  # Green
                else:
                    widget.setStyleSheet("background-color: #f39c12; color: white;")  # Yellow
                    
        # Update alerts
        alerts = []
        plant_analysis = self.current_analysis['plant_analysis']
        
        # Check sensor data for anomalies
        for metric, value in sensor_data.items():
            if metric in HEALTH_THRESHOLDS:
                threshold = HEALTH_THRESHOLDS[metric]
                if value < threshold['min']:
                    alerts.append(f"⚠️ {metric.replace('_', ' ').title()} is low: {value} {threshold['unit']} "
                                f"(normal: {threshold['min']}-{threshold['max']} {threshold['unit']})")
                elif value > threshold['max']:
                    alerts.append(f"⚠️ {metric.replace('_', ' ').title()} is high: {value} {threshold['unit']} "
                                f"(normal: {threshold['min']}-{threshold['max']} {threshold['unit']})")
        
        # Add issues from visual analysis
        for issue in plant_analysis.get('issues_detected', []):
            alerts.append(f"🔍 {issue}")
            
        # Add recommendations
        for rec in plant_analysis.get('recommendations', []):
            alerts.append(f"💡 {rec}")
            
        if alerts:
            self.alerts_display.setPlainText("\n\n".join(alerts))
        else:
            self.alerts_display.setPlainText("No alerts. Plant health appears normal.")
            
        # Update detailed report
        self.update_report()
        
    def update_report(self):
        """Update the detailed report tab"""
        if not self.current_analysis:
            return
            
        # Clear previous report
        if hasattr(self, 'report_widget'):
            self.report_scroll.takeWidget()
            
        # Create new report
        self.report_widget = AnalysisReport(self.current_analysis)
        self.report_scroll.setWidget(self.report_widget)
        
    def update_history_list(self):
        """Update the history list with all analyses"""
        self.history_list.setRowCount(len(self.analysis_history))
        
        for i, analysis in enumerate(reversed(self.analysis_history)):
            timestamp = datetime.fromisoformat(analysis['timestamp']).strftime("%Y-%m-%d %H:%M")
            health_score = analysis['plant_analysis']['health_score']
            
            # Create table items
            self.history_list.setItem(i, 0, QTableWidgetItem(timestamp))
            self.history_list.setItem(i, 1, QTableWidgetItem(str(health_score)))
            
            # Status item with color coding
            status_item = QTableWidgetItem()
            if health_score >= 70:
                status_item.setText("Healthy")
                status_item.setBackground(QColor(46, 204, 113))  # Green
            elif health_score >= 50:
                status_item.setText("Needs Attention")
                status_item.setBackground(QColor(243, 156, 18))  # Yellow
            else:
                status_item.setText("Critical")
                status_item.setBackground(QColor(231, 76, 60))   # Red
                
            status_item.setForeground(QColor(255, 255, 255))  # White text
            self.history_list.setItem(i, 2, status_item)
            
    def view_history_item(self, index):
        """View a historical analysis item"""
        # Reverse index since we display history in reverse chronological order
        idx = len(self.analysis_history) - index.row() - 1
        if 0 <= idx < len(self.analysis_history):
            self.current_analysis = self.analysis_history[idx]
            self.update_dashboard()
            self.tabs.setCurrentIndex(1)  # Switch to report tab
            
    def generate_pdf_report(self):
        """Generate a PDF report of the current analysis"""
        if not self.current_analysis:
            QMessageBox.warning(self, "No Analysis", "No analysis data available to generate report.")
            return
            
        # In a real implementation, you would use a library like ReportLab
        # to generate a professional PDF report
        
        # For demonstration, we'll just show a message
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF Report", "", "PDF Files (*.pdf)"
        )
        
        if save_path:
            # Here you would implement actual PDF generation
            # For now, we'll just create a simple text file as a placeholder
            try:
                with open(save_path, 'w') as f:
                    f.write("Plant Health Analysis Report\n")
                    f.write("============================\n\n")
                    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Health Score: {self.current_analysis['plant_analysis']['health_score']}/100\n\n")
                    
                    f.write("Sensor Data:\n")
                    for metric, value in self.current_analysis['sensor_data'].items():
                        f.write(f"  {metric.replace('_', ' ').title()}: {value} {HEALTH_THRESHOLDS[metric]['unit']}\n")
                    
                    f.write("\nVisual Analysis:\n")
                    f.write(f"  {self.current_analysis['plant_analysis']['analysis']}\n")
                    
                    f.write("\nRecommendations:\n")
                    for rec in self.current_analysis['plant_analysis']['recommendations']:
                        f.write(f"  • {rec}\n")
                        
                QMessageBox.information(self, "Report Generated", 
                                      f"Report saved to: {save_path}\n\nNote: This is a placeholder implementation. "
                                      "In production, you would use a proper PDF generation library.")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to generate report: {str(e)}")
                
    def email_report(self):
        """Email the current analysis report"""
        if not self.current_analysis:
            QMessageBox.warning(self, "No Analysis", "No analysis data available to email.")
            return
            
        # In a real implementation, you would integrate with an email service
        QMessageBox.information(self, "Email Report", 
                              "This would send the report via email in a production environment.\n\n"
                              "You would need to implement email integration with your preferred email service.")
        
    def load_settings(self):
        """Load application settings"""
        # In a real implementation, you would load from a config file
        pass
        
    def save_settings(self):
        """Save application settings"""
        # In a real implementation, you would save to a config file
        pass
        
    def closeEvent(self, event):
        """Handle application close event"""
        self.save_settings()
        event.accept()

def call_poe_bot(bot_name, message, image_path=None):
    """
    Function to call Poe bots via API
    Replace this with your actual Poe API implementation
    """
    # This is a placeholder implementation
    # In production, you would make actual API calls to Poe
    
    headers = {
        'Authorization': f'Bearer {POE_API_CONFIG["api_key"]}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'bot': POE_API_CONFIG['bots'].get(bot_name, bot_name),
        'message': message
    }
    
    # Add image if provided (would need to encode as base64)
    if image_path:
        # Implementation for image handling would go here
        pass
        
    try:
        # Actual API call would look something like this:
        # response = requests.post(POE_API_CONFIG['base_url'], headers=headers, json=payload)
        # return response.json()
        
        # For now, return mock data
        if bot_name == "plant_health_analyzer":
            return {
                "analysis": "Plant exhibits healthy green foliage with good leaf structure. "
                           "Minor yellowing observed on lower leaves, possibly indicating "
                           "natural aging or slight nutrient deficiency.",
                "health_score": 82,
                "confidence": 0.89,
                "recommendations": [
                    "Monitor nitrogen levels - slight deficiency indicated by lower leaf yellowing",
                    "Maintain current watering schedule",
                    "Consider light fertilization with balanced NPK",
                    "Continue regular monitoring for pest activity"
                ],
                "issues_detected": [
                    "Mild nitrogen deficiency (lower leaf yellowing)",
                    "Slight leaf tip browning - possible over-fertilization"
                ],
                "disease_risk": "Low",
                "growth_stage": "Mature vegetative"
            }
        elif bot_name == "sensor_data_analyzer":
            import random
            return {
                "soil_moisture": round(random.uniform(45, 65), 1),
                "ph_level": round(random.uniform(6.2, 7.2), 1),
                "nitrogen": round(random.uniform(18, 35), 1),
                "phosphorus": round(random.uniform(6, 14), 1),
                "potassium": round(random.uniform(35, 55), 1),
                "temperature": round(random.uniform(20, 24), 1),
                "humidity": round(random.uniform(55, 75), 1),
                "light_intensity": round(random.uniform(300, 600), 0)
            }
            
    except Exception as e:
        print(f"Error calling Poe bot {bot_name}: {str(e)}")
        return {"error": f"API call failed: {str(e)}"}

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    dashboard = Dashboard()
    dashboard.show()
    
    sys.exit(app.exec_())