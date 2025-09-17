import sys
import os
import json
import base64
from datetime import datetime
from pathlib import Path
import cv2
import time
import threading

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QScrollArea, QFrame, QGridLayout,
                             QProgressBar, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
                             QMessageBox, QSplitter, QGroupBox, QTextEdit, QSpinBox, QCheckBox)
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QPixmap, QFont, QColor, QPalette, QIcon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import openai

# ==================== CONFIGURATION ====================
# CAMERA SETTINGS - CHANGE CAPTURE INTERVAL HERE
CAPTURE_INTERVAL = 60  # seconds between captures (60 = 1 minute)
# For testing: CAPTURE_INTERVAL = 10  # 10 seconds for testing

# Poe API Configuration
POE_API_CONFIG = {
    'api_key': 'ayUVKcP-GPAdMwlb3pmsX6K0ykLMX5D97oCxtTiqOVw',  # Replace with your actual API key
    'base_url': 'https://api.poe.com/v1',
    'model': 'BotLPXCKK6G14'  # Your plant analysis bot
}

# Health thresholds
HEALTH_THRESHOLDS = {
    'soil_moisture': {'min': 40, 'max': 70, 'unit': '%', 'description': 'Soil water content percentage'},
    'ph_level': {'min': 6.0, 'max': 7.5, 'unit': 'pH', 'description': 'Soil acidity/alkalinity level'},
    'nitrogen': {'min': 20, 'max': 40, 'unit': 'ppm', 'description': 'Nitrogen content in soil'},
    'phosphorus': {'min': 5, 'max': 15, 'unit': 'ppm', 'description': 'Phosphorus content in soil'},
    'potassium': {'min': 30, 'max': 60, 'unit': 'ppm', 'description': 'Potassium content in soil'},
    'temperature': {'min': 18, 'max': 26, 'unit': '°C', 'description': 'Ambient temperature'},
    'humidity': {'min': 50, 'max': 80, 'unit': '%', 'description': 'Relative humidity level'},
    'light_intensity': {'min': 200, 'max': 800, 'unit': 'μmol/m²/s', 'description': 'Light intensity for photosynthesis'}
}

class CameraThread(QThread):
    """Thread to handle automatic camera capture"""
    image_captured = pyqtSignal(str)  # Signal emits the path of captured image
    capture_status = pyqtSignal(str)  # Signal for status updates
    
    def __init__(self, capture_interval=CAPTURE_INTERVAL):
        super().__init__()
        self.capture_interval = capture_interval
        self.running = True
        self.captures_folder = "captures"
        self.create_captures_folder()
        
    def create_captures_folder(self):
        """Create the captures folder if it doesn't exist"""
        if not os.path.exists(self.captures_folder):
            os.makedirs(self.captures_folder)
            
    def set_capture_interval(self, interval):
        """Update capture interval"""
        self.capture_interval = interval
        
    def capture_image(self):
        """Capture a single image from the webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            self.capture_status.emit("❌ Error: Could not open webcam")
            return None
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        ret, frame = cap.read()
        
        if ret:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"plant_{timestamp}.jpg"
            filepath = os.path.join(self.captures_folder, filename)
            
            cv2.imwrite(filepath, frame)
            cap.release()
            
            self.capture_status.emit(f"📸 Captured: {filename}")
            return filepath
        else:
            cap.release()
            self.capture_status.emit("❌ Error: Failed to capture image")
            return None
            
    def run(self):
        """Main thread loop"""
        while self.running:
            filepath = self.capture_image()
            if filepath:
                self.image_captured.emit(filepath)
            
            # Wait for the specified interval
            for _ in range(self.capture_interval):
                if not self.running:
                    break
                time.sleep(1)
                
    def stop(self):
        """Stop the camera thread"""
        self.running = False
        self.wait()

class ModernMetricCard(QWidget):
    """Modern metric display card"""
    def __init__(self, name, value, threshold, parent=None):
        super().__init__(parent)
        self.name = name
        self.value = value
        self.threshold = threshold
        
        self.setFixedHeight(120)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(8)
        
        # Metric name
        name_label = QLabel(self.name.replace('_', ' ').title())
        name_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        name_label.setStyleSheet("color: #2c3e50;")
        layout.addWidget(name_label)
        
        # Value
        value_label = QLabel(f"{self.value} {self.threshold['unit']}")
        value_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(value_label)
        
        # Range indicator
        range_label = QLabel(f"Normal: {self.threshold['min']}-{self.threshold['max']} {self.threshold['unit']}")
        range_label.setFont(QFont("Segoe UI", 9))
        range_label.setStyleSheet("color: #7f8c8d;")
        layout.addWidget(range_label)
        
        # Status indicator
        status_widget = QWidget()
        status_widget.setFixedHeight(6)
        
        # Determine status color
        if self.value < self.threshold['min'] * 0.8 or self.value > self.threshold['max'] * 1.2:
            color = "#e74c3c"  # Red - Critical
            value_label.setStyleSheet("color: #e74c3c;")
        elif self.value >= self.threshold['min'] and self.value <= self.threshold['max']:
            color = "#27ae60"  # Green - Normal
            value_label.setStyleSheet("color: #27ae60;")
        else:
            color = "#f39c12"  # Orange - Warning
            value_label.setStyleSheet("color: #f39c12;")
            
        status_widget.setStyleSheet(f"background-color: {color}; border-radius: 3px;")
        layout.addWidget(status_widget)
        
        self.setLayout(layout)
        self.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #e1e5e9;
            }
            QWidget:hover {
                border: 1px solid #3498db;
                box-shadow: 0 4px 12px rgba(52, 152, 219, 0.15);
            }
        """)

class ModernImageDisplay(QLabel):
    """Modern image display widget"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText("📷 Waiting for first capture...")
        self.setMinimumSize(300, 200)
        self.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 2px dashed #dee2e6;
                border-radius: 12px;
                color: #6c757d;
                font-size: 14px;
                font-weight: 500;
            }
        """)
        
    def set_image(self, image_path):
        """Set and display the image"""
        try:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.width() - 20, self.height() - 20, 
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.setPixmap(scaled_pixmap)
                self.setStyleSheet("""
                    QLabel {
                        background-color: white;
                        border: 1px solid #e1e5e9;
                        border-radius: 12px;
                    }
                """)
            else:
                self.setText("❌ Failed to load image")
        except Exception as e:
            self.setText(f"❌ Error loading image: {str(e)}")

class ModernDashboard(QMainWindow):
    """Modern Plant Health Dashboard"""
    
    def __init__(self):
        super().__init__()
        self.analysis_history = []
        self.current_analysis = None
        self.camera_thread = None
        
        # Initialize OpenAI client for Poe API
        self.poe_client = openai.OpenAI(
            api_key=POE_API_CONFIG['api_key'],
            base_url=POE_API_CONFIG['base_url'],
        )
        
        self.init_ui()
        self.apply_modern_theme()
        self.start_camera_capture()
        
    def init_ui(self):
        """Initialize the modern UI"""
        self.setWindowTitle("🌱 Plant Health Dashboard")
        self.setGeometry(100, 100, 1600, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Left panel - Camera and Controls
        left_panel = self.create_left_panel()
        left_panel.setMaximumWidth(400)
        
        # Right panel - Analysis Results
        right_panel = self.create_right_panel()
        
        # Add to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, stretch=2)
        
        # Status bar
        self.statusBar().showMessage("🚀 Dashboard initialized - Starting camera...")
        
    def create_left_panel(self):
        """Create the left control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(20)
        
        # Camera preview
        camera_group = QGroupBox("📷 Live Plant Monitor")
        camera_layout = QVBoxLayout(camera_group)
        
        self.image_display = ModernImageDisplay()
        camera_layout.addWidget(self.image_display)
        
        layout.addWidget(camera_group)
        
        # Camera controls
        controls_group = QGroupBox("⚙️ Camera Settings")
        controls_layout = QVBoxLayout(controls_group)
        
        # Capture interval setting
        interval_layout = QHBoxLayout()
        interval_label = QLabel("Capture Interval (seconds):")
        self.interval_spinbox = QSpinBox()
        self.interval_spinbox.setRange(5, 3600)  # 5 seconds to 1 hour
        self.interval_spinbox.setValue(CAPTURE_INTERVAL)
        self.interval_spinbox.valueChanged.connect(self.update_capture_interval)
        
        interval_layout.addWidget(interval_label)
        interval_layout.addWidget(self.interval_spinbox)
        controls_layout.addLayout(interval_layout)
        
        # Auto-analysis toggle
        self.auto_analyze_checkbox = QCheckBox("Auto-analyze new captures")
        self.auto_analyze_checkbox.setChecked(True)
        controls_layout.addWidget(self.auto_analyze_checkbox)
        
        # Manual analyze button
        self.analyze_btn = QPushButton("🔍 Analyze Current Image")
        self.analyze_btn.clicked.connect(self.analyze_current_image)
        self.analyze_btn.setEnabled(False)
        controls_layout.addWidget(self.analyze_btn)
        
        layout.addWidget(controls_group)
        
        # Analysis history
        history_group = QGroupBox("📊 Analysis History")
        history_layout = QVBoxLayout(history_group)
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(3)
        self.history_table.setHorizontalHeaderLabels(["Time", "Score", "Status"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.setMaximumHeight(200)
        self.history_table.doubleClicked.connect(self.view_history_item)
        
        history_layout.addWidget(self.history_table)
        layout.addWidget(history_group)
        
        layout.addStretch()
        return panel
        
    def create_right_panel(self):
        """Create the right analysis panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Header
        header = QLabel("🌿 Plant Health Analysis")
        header.setFont(QFont("Segoe UI", 20, QFont.Bold))
        header.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(header)
        
        # Health score display
        self.health_score_widget = self.create_health_score_widget()
        layout.addWidget(self.health_score_widget)
        
        # Metrics grid
        self.metrics_container = QWidget()
        self.metrics_layout = QGridLayout(self.metrics_container)
        self.metrics_layout.setSpacing(15)
        layout.addWidget(self.metrics_container)
        
        # Analysis report
        report_group = QGroupBox("📋 Detailed Analysis")
        report_layout = QVBoxLayout(report_group)
        
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setMaximumHeight(150)
        self.analysis_text.setPlainText("Waiting for first analysis...")
        report_layout.addWidget(self.analysis_text)
        
        layout.addWidget(report_group)
        
        # Recommendations
        rec_group = QGroupBox("💡 Recommendations")
        rec_layout = QVBoxLayout(rec_group)
        
        self.recommendations_text = QTextEdit()
        self.recommendations_text.setReadOnly(True)
        self.recommendations_text.setMaximumHeight(120)
        self.recommendations_text.setPlainText("Analysis recommendations will appear here...")
        rec_layout.addWidget(self.recommendations_text)
        
        layout.addWidget(rec_group)
        
        return panel
        
    def create_health_score_widget(self):
        """Create the health score display widget"""
        widget = QWidget()
        widget.setFixedHeight(100)
        layout = QHBoxLayout(widget)
        
        # Score circle (simplified as text for now)
        score_container = QWidget()
        score_container.setFixedSize(80, 80)
        score_layout = QVBoxLayout(score_container)
        score_layout.setContentsMargins(0, 0, 0, 0)
        
        self.health_score_label = QLabel("--")
        self.health_score_label.setFont(QFont("Segoe UI", 24, QFont.Bold))
        self.health_score_label.setAlignment(Qt.AlignCenter)
        self.health_score_label.setStyleSheet("color: #7f8c8d;")
        score_layout.addWidget(self.health_score_label)
        
        score_text = QLabel("Health Score")
        score_text.setFont(QFont("Segoe UI", 10))
        score_text.setAlignment(Qt.AlignCenter)
        score_text.setStyleSheet("color: #7f8c8d;")
        score_layout.addWidget(score_text)
        
        # Status text
        self.status_label = QLabel("Waiting for analysis...")
        self.status_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.status_label.setStyleSheet("color: #7f8c8d;")
        
        layout.addWidget(score_container)
        layout.addWidget(self.status_label)
        layout.addStretch()
        
        widget.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #e1e5e9;
                margin-bottom: 20px;
            }
        """)
        
        return widget
        
    def apply_modern_theme(self):
        """Apply modern theme to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                color: #2c3e50;
                border: 1px solid #e1e5e9;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                background-color: white;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
            QTextEdit {
                border: 1px solid #e1e5e9;
                border-radius: 8px;
                background-color: #f8f9fa;
                padding: 10px;
            }
            QSpinBox {
                border: 1px solid #e1e5e9;
                border-radius: 6px;
                padding: 8px;
                background-color: white;
            }
            QCheckBox {
                font-weight: 500;
                color: #2c3e50;
            }
            QTableWidget {
                border: 1px solid #e1e5e9;
                border-radius: 8px;
                background-color: white;
                gridline-color: #f1f3f4;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                padding: 8px;
                border: 1px solid #e1e5e9;
                font-weight: bold;
            }
        """)
        
    def start_camera_capture(self):
        """Start the automatic camera capture"""
        self.camera_thread = CameraThread(CAPTURE_INTERVAL)
        self.camera_thread.image_captured.connect(self.on_image_captured)
        self.camera_thread.capture_status.connect(self.on_capture_status)
        self.camera_thread.start()
        
    @pyqtSlot(str)
    def on_image_captured(self, image_path):
        """Handle new image capture"""
        self.image_display.set_image(image_path)
        self.current_image_path = image_path
        self.analyze_btn.setEnabled(True)
        
        # Auto-analyze if enabled
        if self.auto_analyze_checkbox.isChecked():
            self.analyze_current_image()
            
    @pyqtSlot(str)  
    def on_capture_status(self, status):
        """Handle capture status updates"""
        self.statusBar().showMessage(status)
        
    def update_capture_interval(self, interval):
        """Update the camera capture interval"""
        if self.camera_thread:
            self.camera_thread.set_capture_interval(interval)
            self.statusBar().showMessage(f"📷 Capture interval updated to {interval} seconds")
            
    def analyze_current_image(self):
        """Analyze the current image using Poe API"""
        if not hasattr(self, 'current_image_path'):
            QMessageBox.warning(self, "No Image", "No image available for analysis.")
            return
            
        self.statusBar().showMessage("🔍 Analyzing plant health...")
        self.analyze_btn.setEnabled(False)
        
        try:
            # Convert image to base64 for API
            with open(self.current_image_path, 'rb') as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Call Poe API
            response = self.poe_client.chat.completions.create(
                model=POE_API_CONFIG['model'],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this plant image for health assessment. Provide: 1) Overall health score (0-100), 2) Detailed analysis of visible conditions, 3) Specific recommendations for improvement, 4) Any diseases or issues detected. Format as JSON with keys: health_score, analysis, recommendations (array), issues_detected (array)."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ]
            )
            
            # Parse response
            analysis_text = response.choices[0].message.content
            
            # Try to parse JSON, fallback to text parsing
            try:
                import json
                analysis_data = json.loads(analysis_text)
            except:
                # Fallback parsing
                analysis_data = {
                    "health_score": 75,  # Default score
                    "analysis": analysis_text,
                    "recommendations": ["Monitor plant regularly", "Ensure proper watering"],
                    "issues_detected": []
                }
            
            # Generate mock sensor data (replace with actual sensor integration)
            sensor_data = self.generate_mock_sensor_data()
            
            # Create complete analysis
            complete_analysis = {
                "timestamp": datetime.now().isoformat(),
                "image_path": self.current_image_path,
                "plant_analysis": analysis_data,
                "sensor_data": sensor_data
            }
            
            self.current_analysis = complete_analysis
            self.analysis_history.append(complete_analysis)
            
            # Update UI
            self.update_dashboard()
            self.update_history_table()
            
            self.statusBar().showMessage("✅ Analysis completed successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Failed to analyze image: {str(e)}")
            self.statusBar().showMessage("❌ Analysis failed")
            
        finally:
            self.analyze_btn.setEnabled(True)
            
    def generate_mock_sensor_data(self):
        """Generate mock sensor data (replace with actual sensor integration)"""
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
        
    def update_dashboard(self):
        """Update dashboard with current analysis"""
        if not self.current_analysis:
            return
            
        analysis = self.current_analysis['plant_analysis']
        sensor_data = self.current_analysis['sensor_data']
        
        # Update health score
        health_score = analysis.get('health_score', 0)
        self.health_score_label.setText(str(health_score))
        
        # Update status and colors
        if health_score >= 80:
            self.health_score_label.setStyleSheet("color: #27ae60;")
            self.status_label.setText("Excellent Health")
            self.status_label.setStyleSheet("color: #27ae60;")
        elif health_score >= 60:
            self.health_score_label.setStyleSheet("color: #f39c12;")
            self.status_label.setText("Good Health")
            self.status_label.setStyleSheet("color: #f39c12;")
        else:
            self.health_score_label.setStyleSheet("color: #e74c3c;")
            self.status_label.setText("Needs Attention")
            self.status_label.setStyleSheet("color: #e74c3c;")
            
        # Update metrics grid
        self.clear_metrics()
        row, col = 0, 0
        for metric, value in sensor_data.items():
            if metric in HEALTH_THRESHOLDS:
                metric_card = ModernMetricCard(metric, value, HEALTH_THRESHOLDS[metric])
                self.metrics_layout.addWidget(metric_card, row, col)
                col += 1
                if col > 2:  # 3 columns
                    col = 0
                    row += 1
                    
        # Update analysis text
        self.analysis_text.setPlainText(analysis.get('analysis', 'No analysis available'))
        
        # Update recommendations
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            rec_text = "\n".join([f"• {rec}" for rec in recommendations])
        else:
            rec_text = "No specific recommendations at this time."
        self.recommendations_text.setPlainText(rec_text)
        
    def clear_metrics(self):
        """Clear existing metric widgets"""
        while self.metrics_layout.count():
            child = self.metrics_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
    def update_history_table(self):
        """Update the history table"""
        self.history_table.setRowCount(min(len(self.analysis_history), 10))  # Show last 10
        
        for i, analysis in enumerate(reversed(self.analysis_history[-10:])):
            timestamp = datetime.fromisoformat(analysis['timestamp']).strftime("%H:%M:%S")
            health_score = analysis['plant_analysis'].get('health_score', 0)
            
            self.history_table.setItem(i, 0, QTableWidgetItem(timestamp))
            self.history_table.setItem(i, 1, QTableWidgetItem(str(health_score)))
            
            # Status with color
            status_item = QTableWidgetItem()
            if health_score >= 80:
                status_item.setText("Excellent")
                status_item.setForeground(QColor(39, 174, 96))
            elif health_score >= 60:
                status_item.setText("Good")  
                status_item.setForeground(QColor(243, 156, 18))
            else:
                status_item.setText("Poor")
                status_item.setForeground(QColor(231, 76, 60))
                
            self.history_table.setItem(i, 2, status_item)
            
    def view_history_item(self, index):
        """View a historical analysis"""
        row = index.row()
        if row < len(self.analysis_history):
            # Get the analysis (accounting for reversed display)
            analysis_index = len(self.analysis_history) - 1 - row
            self.current_analysis = self.analysis_history[analysis_index]
            
            # Update image display
            if os.path.exists(self.current_analysis['image_path']):
                self.image_display.set_image(self.current_analysis['image_path'])
            
            # Update dashboard
            self.update_dashboard()
            
    def closeEvent(self, event):
        """Handle application close"""
        if self.camera_thread:
            self.camera_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set modern application properties
    app.setApplicationName("Plant Health Dashboard")
    app.setApplicationVersion("2.0")
    app.setStyle("Fusion")  # Modern fusion style
    
    # Apply modern palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(248, 249, 250))
    palette.setColor(QPalette.WindowText, QColor(44, 62, 80))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(241, 243, 244))
    app.setPalette(palette)
    
    dashboard = ModernDashboard()
    dashboard.show()
    
    sys.exit(app.exec_())