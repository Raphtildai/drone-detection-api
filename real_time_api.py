# real_time_api.py
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
import threading
import json

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global real-time detector
real_time_detector = None

@app.route('/api/start-monitoring', methods=['POST'])
def start_monitoring():
    """Start real-time monitoring"""
    global real_time_detector
    
    try:
        if real_time_detector and real_time_detector.is_monitoring:
            return jsonify({'status': 'error', 'message': 'Monitoring already active'})
        
        # Initialize real-time detector
        from real_time_audio import RealTimeDroneDetector
        real_time_detector = RealTimeDroneDetector(config)
        
        def detection_callback(result):
            """Callback when drone is detected"""
            # Send real-time update via WebSocket
            socketio.emit('drone_detected', {
                'timestamp': time.time(),
                'detected': True,
                'confidence': result.get('confidence', 0),
                'position': result.get('position'),
                'localized': result.get('localized', False)
            })
        
        # Start monitoring
        real_time_detector.start_monitoring(detection_callback)
        
        return jsonify({
            'status': 'success',
            'message': 'Real-time monitoring started',
            'channels': 3
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/stop-monitoring', methods=['POST'])
def stop_monitoring():
    """Stop real-time monitoring"""
    global real_time_detector
    
    try:
        if real_time_detector:
            real_time_detector.stop_monitoring()
            real_time_detector = None
        
        return jsonify({
            'status': 'success',
            'message': 'Real-time monitoring stopped'
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/monitoring-status', methods=['GET'])
def monitoring_status():
    """Get current monitoring status"""
    global real_time_detector
    
    status = {
        'active': real_time_detector and real_time_detector.is_monitoring,
        'channels': 3 if real_time_detector else 0,
        'timestamp': time.time()
    }
    
    return jsonify(status)

# WebSocket events
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    socketio.emit('status', {'message': 'Connected to drone monitoring'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')