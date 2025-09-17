<template>
  <div id="app" class="app-container">
    <div class="notification-container">
      <div v-for="(alert, index) in alerts" :key="index" 
           :class="['alert', alert.type]">
        <span class="alert-message">{{ alert.message }}</span>
        <button @click="dismissAlert(index)" class="alert-dismiss">×</button>
      </div>
    </div>

    <header class="app-header">
      <h1>🌱 Plant Health Dashboard</h1>
      <div class="header-controls">
        <button @click="captureImage" class="btn btn-primary">
          📸 Capture Image
        </button>
        <button @click="downloadReport" class="btn btn-secondary" :disabled="!currentAnalysis">
          📄 Download Report
        </button>
      </div>
    </header>

    <div class="dashboard-container">
      <!-- Left Panel -->
      <div class="left-panel">
        <div class="card">
          <h2>📷 Live Plant Monitor</h2>
          <div class="image-container">
            <img v-if="currentImage" :src="currentImage" alt="Plant Image" class="plant-image" />
            <div v-else class="image-placeholder">
              <span>📷 Waiting for first capture...</span>
            </div>
          </div>
        </div>

        <div class="card">
          <h2>⚙️ Camera Settings</h2>
          <div class="settings-controls">
            <label>
              Auto-analyze new captures
              <input type="checkbox" v-model="autoAnalyze" />
            </label>
            <button @click="analyzeCurrent" class="btn btn-primary" :disabled="!currentImage">
              🔍 Analyze Current Image
            </button>
          </div>
        </div>

        <div class="card">
          <h2>📊 Analysis History</h2>
          <div class="history-table">
            <div v-for="(item, index) in history" :key="index" 
                 class="history-item" @click="viewHistoryItem(item)">
              <span class="history-time">{{ formatTime(item.timestamp) }}</span>
              <span class="history-score" :class="getScoreClass(item.plant_analysis.health_score)">
                {{ item.plant_analysis.health_score }}
              </span>
              <span class="history-status">{{ getStatusText(item.plant_analysis.health_score) }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Right Panel -->
      <div class="right-panel">
        <div class="card health-score-card">
          <h2>🌿 Plant Health Analysis</h2>
          <div class="score-display">
            <div class="score-circle" :class="getScoreClass(currentAnalysis?.plant_analysis.health_score)">
              <span class="score-value">{{ currentAnalysis?.plant_analysis.health_score || '--' }}</span>
              <span class="score-label">Health Score</span>
            </div>
            <div class="status-text">
              {{ getStatusText(currentAnalysis?.plant_analysis.health_score) || 'Waiting for analysis...' }}
            </div>
          </div>
        </div>

        <div class="card">
          <h2>📈 Health Metrics</h2>
          <div class="metrics-grid">
            <div v-for="(value, metric) in currentAnalysis?.sensor_data || {}" :key="metric" 
                 class="metric-card" :class="getMetricStatus(metric, value)">
              <h3>{{ formatMetricName(metric) }}</h3>
              <div class="metric-value">{{ value }} {{ HEALTH_THRESHOLDS[metric]?.unit || '' }}</div>
              <div class="metric-range">
                Normal: {{ HEALTH_THRESHOLDS[metric]?.min || 0 }}-{{ HEALTH_THRESHOLDS[metric]?.max || 0 }} 
                {{ HEALTH_THRESHOLDS[metric]?.unit || '' }}
              </div>
              <div class="metric-status-bar"></div>
            </div>
          </div>
        </div>

        <div class="card">
          <h2>📋 Detailed Analysis</h2>
          <div class="analysis-text">
            {{ currentAnalysis?.plant_analysis.analysis || 'Waiting for first analysis...' }}
          </div>
        </div>

        <div class="card">
          <h2>💡 Recommendations</h2>
          <ul class="recommendations-list">
            <li v-for="(rec, index) in currentAnalysis?.plant_analysis.recommendations || []" :key="index">
              {{ rec }}
            </li>
            <li v-if="!currentAnalysis?.plant_analysis.recommendations?.length">
              No specific recommendations at this time.
            </li>
          </ul>
        </div>
      </div>
    </div>

    <input type="file" ref="fileInput" @change="handleFileUpload" accept="image/*" capture="environment" style="display: none;" />
  </div>
</template>

<script>
import io from 'socket.io-client';

export default {
  name: 'App',
  data() {
    return {
      socket: null,
      currentImage: null,
      currentAnalysis: null,
      history: [],
      alerts: [],
      autoAnalyze: true,
      HEALTH_THRESHOLDS: {
        'soil_moisture': {'min': 40, 'max': 70, 'unit': '%'},
        'ph_level': {'min': 6.0, 'max': 7.5, 'unit': 'pH'},
        'nitrogen': {'min': 20, 'max': 40, 'unit': 'ppm'},
        'phosphorus': {'min': 5, 'max': 15, 'unit': 'ppm'},
        'potassium': {'min': 30, 'max': 60, 'unit': 'ppm'},
        'temperature': {'min': 18, 'max': 26, 'unit': '°C'},
        'humidity': {'min': 50, 'max': 80, 'unit': '%'},
        'light_intensity': {'min': 200, 'max': 800, 'unit': 'μmol/m²/s'}
      }
    };
  },
  mounted() {
    this.initSocket();
    this.loadHistory();
  },
  methods: {
    initSocket() {
      this.socket = io.connect('http://localhost:5000');
      
      this.socket.on('connected', (data) => {
        console.log('Connected to server:', data.message);
      });
      
      this.socket.on('image_captured', (data) => {
        this.currentImage = this.getImageUrl(data.image_path);
        if (this.autoAnalyze) {
          this.analyzeImage(data.image_path);
        }
      });
      
      this.socket.on('analysis_started', (data) => {
        this.showNotification('Analysis started...', 'info');
      });
      
      this.socket.on('analysis_complete', (data) => {
        this.currentAnalysis = data;
        this.loadHistory();
        this.showNotification('Analysis completed!', 'success');
      });
      
      this.socket.on('analysis_error', (data) => {
        this.showNotification(`Analysis error: ${data.error}`, 'error');
      });
      
      this.socket.on('alerts', (alerts) => {
        this.alerts = [...this.alerts, ...alerts];
        alerts.forEach(alert => {
          this.showNotification(alert.message, alert.type);
        });
      });
    },
    
    getImageUrl(path) {
      // In a real app, you'd serve images through a proper endpoint
      return path ? `http://localhost:5000/${path}` : null;
    },
    
   async captureImage() {
        try {
          // Use the camera capture endpoint instead of file upload
          const response = await fetch('/api/camera/capture', {
            method: 'POST'
          });
          
          if (!response.ok) {
            throw new Error('Failed to capture image from camera');
          }
          
          const data = await response.json();
          this.showNotification('Image captured successfully!', 'success');
          
          // The backend will emit socket events for image processing
          
        } catch (error) {
          this.showNotification(`Error: ${error.message}`, 'error');
          console.error('Capture error:', error);
          
          // Fallback to file input if camera capture fails
          this.$refs.fileInput.click();
        }
      },
    async handleFileUpload(event) {
       const file = event.target.files[0];
        if (!file) return;
        
        const formData = new FormData();
        formData.append('image', file);
        
        try {
          const response = await fetch('/api/capture', {
            method: 'POST',
            body: formData
          });
          
          if (!response.ok) {
            throw new Error('Failed to upload image');
          }
          
          this.showNotification('Image uploaded successfully!', 'success');
        } catch (error) {
          this.showNotification(`Error: ${error.message}`, 'error');
        }
        
        // Reset file input
        event.target.value = '';
        },
        
        async analyzeImage(imagePath) {
          try {
            const response = await fetch('/api/analyze', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json'
              },
              body: JSON.stringify({ image_path: imagePath })
            });
            
            if (!response.ok) {
              throw new Error('Failed to start analysis');
            }
            
            this.showNotification('Analysis started...', 'info');
          } catch (error) {
            this.showNotification(`Error: ${error.message}`, 'error');
          }
        },
    
    analyzeCurrent() {
      if (this.currentImage) {
        // Extract path from URL
        const path = this.currentImage.replace('http://localhost:5000/', '');
        this.analyzeImage(path);
      }
    },
    
    async loadHistory() {
      try {
        const response = await fetch('/api/history');
        if (response.ok) {
          const data = await response.json();
          this.history = data.history || [];
        }
      } catch (error) {
        console.error('Error loading history:', error);
      }
    },
    
    viewHistoryItem(item) {
      this.currentAnalysis = item;
      this.currentImage = this.getImageUrl(item.image_path);
    },
    
    async downloadReport() {
      if (!this.currentAnalysis) return;
      
      try {
        // In a real app, you'd use the analysis ID
        const response = await fetch('/api/report/pdf', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ analysis_id: 'current' })
        });
        
        if (response.ok) {
          const blob = await response.blob();
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = 'plant_health_report.pdf';
          document.body.appendChild(a);
          a.click();
          window.URL.revokeObjectURL(url);
          document.body.removeChild(a);
        } else {
          throw new Error('Failed to generate report');
        }
      } catch (error) {
        this.showNotification(`Error: ${error.message}`, 'error');
      }
    },
    
    getScoreClass(score) {
      if (score === undefined || score === null) return 'unknown';
      if (score >= 80) return 'excellent';
      if (score >= 60) return 'good';
      return 'poor';
    },
    
    getStatusText(score) {
      if (score === undefined || score === null) return '';
      if (score >= 80) return 'Excellent Health';
      if (score >= 60) return 'Good Health';
      return 'Needs Attention';
    },
    
    getMetricStatus(metric, value) {
      if (!this.HEALTH_THRESHOLDS[metric]) return 'unknown';
      
      const threshold = this.HEALTH_THRESHOLDS[metric];
      if (value < threshold.min * 0.8 || value > threshold.max * 1.2) return 'critical';
      if (value >= threshold.min && value <= threshold.max) return 'normal';
      return 'warning';
    },
    
    formatMetricName(metric) {
      return metric.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    },
    
    formatTime(timestamp) {
      return new Date(timestamp).toLocaleTimeString();
    },
    
    showNotification(message, type = 'info') {
      // Browser notification if supported
      if ('Notification' in window && Notification.permission === 'granted') {
        new Notification('Plant Health Dashboard', {
          body: message,
          icon: '/favicon.ico'
        });
      }
      
      // Console log for debugging
      console.log(`${type}: ${message}`);
    },
    
    dismissAlert(index) {
      this.alerts.splice(index, 1);
    }
  },
  
  beforeUnmount() {
    if (this.socket) {
      this.socket.disconnect();
    }
  }
};
</script>

<style>
/* Modern CSS styles will go here */
:root {
  --primary-color: #3498db;
  --secondary-color: #2ecc71;
  --warning-color: #f39c12;
  --danger-color: #e74c3c;
  --dark-color: #2c3e50;
  --light-color: #ecf0f1;
  --gray-color: #95a5a6;
  --card-bg: white;
  --border-radius: 12px;
  --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  --transition: all 0.3s ease;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background-color: #f8f9fa;
  color: var(--dark-color);
  line-height: 1.6;
}

.app-container {
  max-width: 1600px;
  margin: 0 auto;
  padding: 20px;
}

.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding: 15px 20px;
  background: white;
  border-radius: var(--border-radius);
  box-shadow: var(--shadow);
}

.app-header h1 {
  font-size: 24px;
  color: var(--dark-color);
}

.header-controls {
  display: flex;
  gap: 10px;
}

.btn {
  padding: 10px 15px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
  transition: var(--transition);
}

.btn-primary {
  background-color: var(--primary-color);
  color: white;
}

.btn-primary:hover {
  background-color: #2980b9;
}

.btn-secondary {
  background-color: var(--secondary-color);
  color: white;
}

.btn-secondary:hover {
  background-color: #27ae60;
}

.btn:disabled {
  background-color: var(--gray-color);
  cursor: not-allowed;
}

.dashboard-container {
  display: grid;
  grid-template-columns: 400px 1fr;
  gap: 20px;
}

.card {
  background: var(--card-bg);
  border-radius: var(--border-radius);
  padding: 20px;
  margin-bottom: 20px;
  box-shadow: var(--shadow);
  transition: var(--transition);
}

.card:hover {
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
}

.card h2 {
  margin-bottom: 15px;
  font-size: 18px;
  color: var(--dark-color);
}

/* Image container */
.image-container {
  width: 100%;
  height: 250px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #f8f9fa;
  border: 2px dashed #dee2e6;
  border-radius: 8px;
  overflow: hidden;
}

.plant-image {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.image-placeholder {
  color: #6c757d;
  font-weight: 500;
}

/* Settings controls */
.settings-controls {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.settings-controls label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 500;
}

/* History table */
.history-table {
  max-height: 200px;
  overflow-y: auto;
}

.history-item {
  display: grid;
  grid-template-columns: 1fr auto auto;
  gap: 10px;
  padding: 10px;
  border-bottom: 1px solid #eee;
  cursor: pointer;
  transition: var(--transition);
}

.history-item:hover {
  background-color: #f8f9fa;
}

.history-time {
  font-size: 14px;
  color: #6c757d;
}

.history-score {
  font-weight: bold;
  padding: 2px 8px;
  border-radius: 20px;
  font-size: 14px;
}

.history-score.excellent {
  background-color: #d4edda;
  color: #155724;
}

.history-score.good {
  background-color: #fff3cd;
  color: #856404;
}

.history-score.poor {
  background-color: #f8d7da;
  color: #721c24;
}

.history-status {
  font-size: 14px;
  font-weight: 500;
}

/* Health score display */
.health-score-card {
  text-align: center;
}

.score-display {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 15px;
}

.score-circle {
  width: 120px;
  height: 120px;
  border-radius: 50%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
}

.score-circle.excellent {
  background-color: #d4edda;
  color: #155724;
  border: 4px solid #28a745;
}

.score-circle.good {
  background-color: #fff3cd;
  color: #856404;
  border: 4px solid #ffc107;
}

.score-circle.poor {
  background-color: #f8d7da;
  color: #721c24;
  border: 4px solid #dc3545;
}

.score-circle.unknown {
  background-color: #e2e3e5;
  color: #383d41;
  border: 4px solid #6c757d;
}

.score-value {
  font-size: 32px;
  line-height: 1;
}

.score-label {
  font-size: 14px;
  margin-top: 5px;
}

.status-text {
  font-size: 18px;
  font-weight: 600;
}

/* Metrics grid */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 15px;
}

.metric-card {
  padding: 15px;
  border-radius: 8px;
  background-color: #f8f9fa;
  position: relative;
  overflow: hidden;
}

.metric-card h3 {
  font-size: 14px;
  margin-bottom: 8px;
  color: #6c757d;
}

.metric-value {
  font-size: 20px;
  font-weight: bold;
  margin-bottom: 5px;
}

.metric-range {
  font-size: 12px;
  color: #6c757d;
}

.metric-card.normal {
  border-left: 4px solid #28a745;
}

.metric-card.warning {
  border-left: 4px solid #ffc107;
}

.metric-card.critical {
  border-left: 4px solid #dc3545;
}

.metric-card.unknown {
  border-left: 4px solid #6c757d;
}

.metric-status-bar {
  position: absolute;
  bottom: 0;
  left: 0;
  height: 3px;
  width: 100%;
}

.metric-card.normal .metric-status-bar {
  background-color: #28a745;
}

.metric-card.warning .metric-status-bar {
  background-color: #ffc107;
}

.metric-card.critical .metric-status-bar {
  background-color: #dc3545;
}

.metric-card.unknown .metric-status-bar {
  background-color: #6c757d;
}

/* Analysis text */
.analysis-text {
  line-height: 1.6;
  padding: 10px;
  background-color: #f8f9fa;
  border-radius: 8px;
  max-height: 200px;
  overflow-y: auto;
}

/* Recommendations */
.recommendations-list {
  padding-left: 20px;
}

.recommendations-list li {
  margin-bottom: 8px;
  line-height: 1.5;
}

/* Notifications */
.notification-container {
  position: fixed;
  top: 20px;
  right: 20px;
  z-index: 1000;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.alert {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-radius: 8px;
  color: white;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  max-width: 350px;
  animation: slideIn 0.3s ease;
}

.alert.critical {
  background-color: var(--danger-color);
}

.alert.warning {
  background-color: var(--warning-color);
}

.alert.info {
  background-color: var(--primary-color);
}

.alert.success {
  background-color: var(--secondary-color);
}

.alert-dismiss {
  background: none;
  border: none;
  color: white;
  font-size: 18px;
  cursor: pointer;
  margin-left: 10px;
}

@keyframes slideIn {
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

/* Responsive design */
@media (max-width: 1200px) {
  .dashboard-container {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .app-header {
    flex-direction: column;
    gap: 15px;
    text-align: center;
  }
  
  .header-controls {
    flex-direction: column;
    width: 100%;
  }
  
  .btn {
    width: 100%;
  }
  
  .metrics-grid {
    grid-template-columns: 1fr;
  }
}
</style>
