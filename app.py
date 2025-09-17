import os
import json
import base64
import asyncio
import aiohttp
import cv2
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import concurrent.futures
import logging
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()  # Load variables from .env file

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'fallback-key-for-development')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CAPTURES_FOLDER'] = 'captures'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
CORS(app)

# Configuration
POE_API_CONFIG = {
    'api_key': os.getenv('POE_API_KEY'),
    'base_url': 'https://api.poe.com/v1',
    'model': 'BotLPXCKK6G14'
}

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

# Thread pool for CPU-intensive tasks
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Global state (in production, use Redis or database)
analysis_history = []
current_analysis = None
auto_capture_enabled = False
auto_capture_thread = None
auto_capture_interval = 300  # 5 minutes default

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CAPTURES_FOLDER'], exist_ok=True)

# Camera functions integrated from camera.py
def create_captures_folder():
    """Create the captures folder if it doesn't exist."""
    if not os.path.exists(app.config['CAPTURES_FOLDER']):
        os.makedirs(app.config['CAPTURES_FOLDER'])
        logger.info(f"Created '{app.config['CAPTURES_FOLDER']}' folder")

def capture_image():
    """Capture a single image from the webcam and return the filepath."""
    cap = None
    try:
        # Try different camera indices
        for camera_index in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                logger.info(f"Camera found at index {camera_index}")
                break
            cap.release()
        
        if not cap or not cap.isOpened():
            logger.error("Could not open any webcam")
            return None
        
        # Allow camera to warm up
        time.sleep(0.5)
        
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        if ret:
            # Generate timestamp filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"plant_{timestamp}.jpg"
            filepath = os.path.join(app.config['CAPTURES_FOLDER'], filename)
            
            # Save the image
            cv2.imwrite(filepath, frame)
            logger.info(f"Captured image: {filename}")
            
            return filepath
        else:
            logger.error("Failed to capture image - no frame received")
            return None
            
    except Exception as e:
        logger.error(f"Error in capture_image: {str(e)}")
        return None
    finally:
        if cap:
            cap.release()

def auto_capture_loop():
    """Auto-capture loop running in background thread."""
    global auto_capture_enabled, auto_capture_interval
    
    logger.info(f"Auto-capture started with interval: {auto_capture_interval} seconds")
    
    while auto_capture_enabled:
        try:
            # Capture an image
            image_path = capture_image()
            
            if image_path:
                # Process the captured image
                process_image(image_path)
                
                # Emit event for new capture
                socketio.emit('auto_capture', {
                    'image_path': image_path,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Wait for the specified interval
            for _ in range(auto_capture_interval):
                if not auto_capture_enabled:
                    break
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in auto-capture loop: {str(e)}")
            time.sleep(30)  # Wait before retrying
    
    logger.info("Auto-capture stopped")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/camera/capture', methods=['POST'])
def capture_from_camera():
    """Capture image from webcam"""
    try:
        image_path = capture_image()
        
        if not image_path:
            return jsonify({'error': 'Failed to capture image from camera. Check if camera is connected and accessible.'}), 500
        
        # Process image in background
        executor.submit(process_image, image_path)
        
        return jsonify({
            'success': True, 
            'image_path': image_path,
            'message': 'Image captured successfully'
        })
    
    except Exception as e:
        logger.error(f"Error capturing from camera: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera/auto-capture', methods=['POST'])
def toggle_auto_capture():
    """Enable or disable auto-capture"""
    global auto_capture_enabled, auto_capture_thread, auto_capture_interval
    
    try:
        data = request.json
        enabled = data.get('enabled', False)
        interval = data.get('interval', 300)
        
        # Update interval if provided
        if interval and interval >= 10:  # Minimum 10 seconds
            auto_capture_interval = interval
        
        # Start or stop auto-capture
        if enabled and not auto_capture_enabled:
            auto_capture_enabled = True
            auto_capture_thread = threading.Thread(target=auto_capture_loop, daemon=True)
            auto_capture_thread.start()
            message = f"Auto-capture enabled with {auto_capture_interval} second interval"
        elif not enabled and auto_capture_enabled:
            auto_capture_enabled = False
            if auto_capture_thread and auto_capture_thread.is_alive():
                auto_capture_thread.join(timeout=5.0)
            message = "Auto-capture disabled"
        else:
            message = f"Auto-capture is already {'enabled' if enabled else 'disabled'}"
        
        return jsonify({
            'success': True, 
            'enabled': auto_capture_enabled,
            'interval': auto_capture_interval,
            'message': message
        })
    
    except Exception as e:
        logger.error(f"Error toggling auto-capture: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera/status', methods=['GET'])
def get_camera_status():
    """Get camera and auto-capture status"""
    return jsonify({
        'auto_capture_enabled': auto_capture_enabled,
        'auto_capture_interval': auto_capture_interval,
        'camera_available': is_camera_available()
    })

def is_camera_available():
    """Check if camera is available"""
    cap = cv2.VideoCapture(0)
    available = cap.isOpened()
    cap.release()
    return available

@app.route('/api/capture', methods=['POST'])
async def capture_image_endpoint():
    """Capture image from uploaded file"""
    try:
        # For web app, we'll receive images from the client
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Save the uploaded image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plant_{timestamp}_{secure_filename(image_file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(filepath)
        
        # Process image in background
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, process_image, filepath)
        
        return jsonify({'success': True, 'image_path': filepath})
    
    except Exception as e:
        logger.error(f"Error capturing image: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_image(filepath):
    """Process image and trigger analysis"""
    try:
        # Optimize image for web (resize, compress)
        img = cv2.imread(filepath)
        if img is None:
            logger.error(f"Failed to read image: {filepath}")
            return
        
        # Resize if too large
        height, width = img.shape[:2]
        if width > 1200:
            scale = 1200 / width
            new_width = 1200
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Save optimized version
            cv2.imwrite(filepath, img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        # Emit event for new image
        socketio.emit('image_captured', {
            'image_path': filepath,
            'timestamp': datetime.now().isoformat()
        })
        
        # Auto-analyze if enabled
        socketio.emit('analysis_started', {'message': 'Starting analysis...'})
        analyze_image(filepath)
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        socketio.emit('analysis_error', {'error': str(e)})

@app.route('/api/analyze', methods=['POST'])
def analyze_image_endpoint():
    """API endpoint to trigger image analysis"""
    try:
        data = request.json
        image_path = data.get('image_path')
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({'error': 'Invalid image path'}), 400
        
        # Run analysis in background
        executor.submit(analyze_image, image_path)
        
        return jsonify({'success': True, 'message': 'Analysis started'})
    
    except Exception as e:
        logger.error(f"Error starting analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

def analyze_image(image_path):
    """Analyze plant image using Poe API"""
    try:
        # Convert image to base64
        with open(image_path, 'rb') as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Call Poe API asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        analysis_data = loop.run_until_complete(call_poe_api(img_base64))
        
        # Generate mock sensor data
        sensor_data = generate_mock_sensor_data()
        
        # Create complete analysis
        complete_analysis = {
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "plant_analysis": analysis_data,
            "sensor_data": sensor_data
        }
        
        # Update global state
        global current_analysis, analysis_history
        current_analysis = complete_analysis
        analysis_history.append(complete_analysis)
        
        # Keep only last 50 analyses
        if len(analysis_history) > 50:
            analysis_history = analysis_history[-50:]
        
        # Emit analysis results
        socketio.emit('analysis_complete', complete_analysis)
        
        # Check for alerts
        check_for_alerts(complete_analysis)
        
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        socketio.emit('analysis_error', {'error': str(e)})

async def call_poe_api(image_base64):
    """Call Poe API asynchronously"""
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                'Authorization': f'Bearer {POE_API_CONFIG["api_key"]}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                "model": POE_API_CONFIG['model'],
                "messages": [
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
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ]
            }
            
            async with session.post(
                f"{POE_API_CONFIG['base_url']}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    analysis_text = result['choices'][0]['message']['content']
                    
                    # Try to parse JSON
                    try:
                        return json.loads(analysis_text)
                    except json.JSONDecodeError:
                        # Fallback parsing
                        return {
                            "health_score": 75,
                            "analysis": analysis_text,
                            "recommendations": ["Monitor plant regularly", "Ensure proper watering"],
                            "issues_detected": []
                        }
                else:
                    error_text = await response.text()
                    logger.error(f"Poe API error: {response.status} - {error_text}")
                    raise Exception(f"API error: {response.status}")
                    
    except Exception as e:
        logger.error(f"Error calling Poe API: {str(e)}")
        # Return fallback data
        return {
            "health_score": 75,
            "analysis": f"Analysis failed: {str(e)}",
            "recommendations": ["Check connection and try again"],
            "issues_detected": ["Analysis error"]
        }

def generate_mock_sensor_data():
    """Generate mock sensor data"""
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

def check_for_alerts(analysis):
    """Check analysis results for alerts"""
    try:
        health_score = analysis['plant_analysis'].get('health_score', 0)
        issues = analysis['plant_analysis'].get('issues_detected', [])
        sensor_data = analysis['sensor_data']
        
        alerts = []
        
        # Health score alert
        if health_score < 60:
            alerts.append({
                'type': 'critical',
                'message': f'Low health score: {health_score}/100',
                'timestamp': datetime.now().isoformat()
            })
        
        # Sensor data alerts
        for metric, value in sensor_data.items():
            if metric in HEALTH_THRESHOLDS:
                threshold = HEALTH_THRESHOLDS[metric]
                if value < threshold['min'] or value > threshold['max']:
                    alerts.append({
                        'type': 'warning' if 0.8 * threshold['min'] <= value <= 1.2 * threshold['max'] else 'critical',
                        'message': f'{metric.replace("_", " ").title()} is out of range: {value} {threshold["unit"]} (normal: {threshold["min"]}-{threshold["max"]})',
                        'timestamp': datetime.now().isoformat()
                    })
        
        # Issue alerts
        for issue in issues:
            alerts.append({
                'type': 'critical',
                'message': f'Detected issue: {issue}',
                'timestamp': datetime.now().isoformat()
            })
        
        # Send alerts if any
        if alerts:
            socketio.emit('alerts', alerts)
            
    except Exception as e:
        logger.error(f"Error checking alerts: {str(e)}")

@app.route('/api/report/pdf', methods=['POST'])
def generate_pdf_report():
    """Generate PDF report"""
    try:
        data = request.json
        analysis_id = data.get('analysis_id')
        
        # Find analysis (in production, use database)
        global analysis_history
        analysis = None
        
        if analysis_id == 'current' and current_analysis:
            analysis = current_analysis
        else:
            for a in analysis_history:
                if a.get('id') == analysis_id:
                    analysis = a
                    break
        
        if not analysis:
            return jsonify({'error': 'Analysis not found'}), 404
        
        # Generate PDF in background thread
        pdf_path = executor.submit(create_pdf_report, analysis).result()
        
        return send_file(pdf_path, as_attachment=True, download_name='plant_health_report.pdf')
    
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        return jsonify({'error': str(e)}), 500

def create_pdf_report(analysis):
    """Create PDF report using ReportLab"""
    try:
        # Create PDF document
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/plant_report_{timestamp}.pdf"
        os.makedirs('reports', exist_ok=True)
        
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Title
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=20,
            spaceAfter=30,
            alignment=1  # Center
        )
        elements.append(Paragraph("Plant Health Analysis Report", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Timestamp
        timestamp_str = datetime.fromisoformat(analysis['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Paragraph(f"Analysis Date: {timestamp_str}", styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Health Score
        health_score = analysis['plant_analysis'].get('health_score', 0)
        score_style = ParagraphStyle(
            'Score',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.green if health_score >= 80 else 
                     colors.orange if health_score >= 60 else colors.red
        )
        elements.append(Paragraph(f"Health Score: {health_score}/100", score_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Sensor Data Table
        sensor_data = analysis['sensor_data']
        sensor_table_data = [['Parameter', 'Value', 'Normal Range', 'Status']]
        
        for metric, value in sensor_data.items():
            if metric in HEALTH_THRESHOLDS:
                threshold = HEALTH_THRESHOLDS[metric]
                normal_range = f"{threshold['min']}-{threshold['max']} {threshold['unit']}"
                
                if value < threshold['min'] * 0.8 or value > threshold['max'] * 1.2:
                    status = "Critical"
                    color = colors.red
                elif value >= threshold['min'] and value <= threshold['max']:
                    status = "Normal"
                    color = colors.green
                else:
                    status = "Warning"
                    color = colors.orange
                
                sensor_table_data.append([
                    metric.replace('_', ' ').title(),
                    f"{value} {threshold['unit']}",
                    normal_range,
                    status
                ])
        
        sensor_table = Table(sensor_table_data, colWidths=[1.5*inch, 1.2*inch, 1.8*inch, 1*inch])
        sensor_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(sensor_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Analysis
        elements.append(Paragraph("Analysis", styles['Heading2']))
        analysis_text = analysis['plant_analysis'].get('analysis', 'No analysis available.')
        elements.append(Paragraph(analysis_text, styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Recommendations
        elements.append(Paragraph("Recommendations", styles['Heading2']))
        recommendations = analysis['plant_analysis'].get('recommendations', [])
        if recommendations:
            for rec in recommendations:
                elements.append(Paragraph(f"• {rec}", styles['Normal']))
        else:
            elements.append(Paragraph("No specific recommendations at this time.", styles['Normal']))
        
        # Build PDF
        doc.build(elements)
        return filename
        
    except Exception as e:
        logger.error(f"Error creating PDF: {str(e)}")
        raise

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get analysis history"""
    try:
        # Return last 10 analyses
        return jsonify({'history': analysis_history[-10:]})
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    emit('connected', {'message': 'Connected to Plant Health Dashboard'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

if __name__ == '__main__':
    # Create necessary directories
    create_captures_folder()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)