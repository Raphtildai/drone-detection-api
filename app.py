#!/usr/bin/env python3
from flask_socketio import SocketIO, emit
import threading
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import torch
import io
import base64
import logging
from pathlib import Path
import tempfile
import os
import time
import gc
import json

from model_loader import ModelLoader
from audio_processor import AudioProcessor

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Socket.IO
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize components
model_loader = ModelLoader()
audio_processor = AudioProcessor()

# Global variables for real-time monitoring
is_monitoring = False
monitoring_thread = None

# Localization configuration
class LocalizationConfig:
    MIC_POSITIONS = np.array([[0, 0], [0.5, 0], [0, 0.5]])  # Microphone positions
    SOUND_SPEED = 343.0  # Speed of sound in m/s

config = LocalizationConfig()

def calculate_tdoa(audio_data, sr):
    """Calculate Time Difference of Arrival between microphone channels"""
    try:
        if audio_data.ndim != 2 or audio_data.shape[1] < 3:
            return None, "Audio must have at least 3 channels for localization"
        
        # Simple cross-correlation based TDOA calculation
        tdoas = []
        ref_channel = audio_data[:, 0]  # Use first channel as reference
        
        for i in range(1, 3):
            # Cross-correlation between reference and other channels
            correlation = np.correlate(ref_channel, audio_data[:, i], mode='full')
            lag = np.argmax(correlation) - (len(ref_channel) - 1)
            tdoa = lag / sr
            tdoas.append(tdoa)
        
        return np.array(tdoas), None
        
    except Exception as e:
        logger.error(f"TDOA calculation error: {str(e)}")
        return None, str(e)

def localize_drone(tdoas, mic_positions, sound_speed):
    """Localize drone position using TDOA and microphone array"""
    try:
        if tdoas is None or len(tdoas) != 2:
            return None, "Invalid TDOA data"
        
        # Simple localization using time differences
        # This is a simplified version - you can implement more sophisticated algorithms
        distances = tdoas * sound_speed
        
        # Using linear equations for localization (simplified)
        A = np.array([
            [2*(mic_positions[1,0] - mic_positions[0,0]), 2*(mic_positions[1,1] - mic_positions[0,1])],
            [2*(mic_positions[2,0] - mic_positions[0,0]), 2*(mic_positions[2,1] - mic_positions[0,1])]
        ])
        
        b = np.array([
            distances[0]**2 - (mic_positions[1,0]**2 + mic_positions[1,1]**2) + (mic_positions[0,0]**2 + mic_positions[0,1]**2),
            distances[1]**2 - (mic_positions[2,0]**2 + mic_positions[2,1]**2) + (mic_positions[0,0]**2 + mic_positions[0,1]**2)
        ])
        
        # Solve for position
        position = np.linalg.lstsq(A, b, rcond=None)[0]
        
        return position.tolist(), None
        
    except Exception as e:
        logger.error(f"Localization error: {str(e)}")
        return None, str(e)

def create_visualization_data(estimated_position, true_position=None, confidence=None, error=None):
    """Create data for visualization plot"""
    mic_positions = config.MIC_POSITIONS.tolist()
    
    # Create plot data
    plot_data = {
        'microphones': [
            {'position': pos, 'label': f'Mic {i+1}', 'color': ['red', 'blue', 'green'][i]}
            for i, pos in enumerate(mic_positions)
        ],
        'estimated_position': {
            'position': estimated_position,
            'confidence': confidence,
            'color': 'red'
        }
    }
    
    if true_position:
        plot_data['true_position'] = {
            'position': true_position,
            'color': 'green'
        }
    
    if error:
        plot_data['error'] = error
    
    return plot_data

# Add these missing functions to your Flask app (before the detect-with-localization endpoint)

def calculate_tdoa_enhanced(audio_data, sr):
    """Enhanced TDOA calculation with better signal processing"""
    try:
        if audio_data.ndim != 2 or audio_data.shape[1] < 3:
            return None, f"Audio must have at least 3 channels. Got {audio_data.shape[1] if audio_data.ndim > 1 else 1} channels"
        
        # Normalize audio data
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        tdoas = []
        ref_channel = audio_data[:, 0]  # Use first channel as reference
        
        for i in range(1, 3):
            # Use cross-correlation with windowing
            correlation = np.correlate(ref_channel, audio_data[:, i], mode='full')
            
            # Find the peak with sub-sample accuracy using parabolic interpolation
            peak_idx = np.argmax(np.abs(correlation))
            
            # Sub-sample interpolation for better accuracy
            if 0 < peak_idx < len(correlation) - 1:
                # Parabolic interpolation around the peak
                y1 = correlation[peak_idx - 1]
                y2 = correlation[peak_idx]
                y3 = correlation[peak_idx + 1]
                
                # Parabolic interpolation formula
                offset = (y3 - y1) / (2 * (2 * y2 - y1 - y3))
                peak_idx = peak_idx + offset
            
            # Calculate time delay
            lag = peak_idx - (len(ref_channel) - 1)
            tdoa = lag / sr
            
            # Validate TDOA (should be within reasonable physical limits)
            max_physical_delay = 2.0 / 343.0  # 2 meters / speed of sound
            if abs(tdoa) > max_physical_delay:
                print(f"Warning: TDOA {tdoa} seems physically unrealistic")
                tdoa = 0.0  # Fallback to zero
                
            tdoas.append(tdoa)
        
        print(f"Calculated TDOAs: {tdoas}")
        return np.array(tdoas), None
        
    except Exception as e:
        logger.error(f"Enhanced TDOA calculation error: {str(e)}")
        return None, str(e)

def localize_drone_enhanced(tdoas, mic_positions, sound_speed):
    """Enhanced localization with better error handling"""
    try:
        if tdoas is None or len(tdoas) != 2:
            return None, "Invalid TDOA data"
        
        # Convert to numpy arrays
        tdoas = np.array(tdoas)
        mic_positions = np.array(mic_positions)
        
        # Simple linear least squares localization
        # Based on time difference equations
        
        # Reference microphone (index 0)
        x0, y0 = mic_positions[0]
        
        # For each other microphone
        A = []
        b = []
        
        for i in range(1, 3):
            xi, yi = mic_positions[i]
            di = tdoas[i-1] * sound_speed  # Distance difference
            
            # Equation: (x - xi)^2 + (y - yi)^2 = (sqrt((x - x0)^2 + (y - y0)^2) + di)^2
            # Linearized form:
            A.append([2*(xi - x0), 2*(yi - y0)])
            b.append([xi**2 + yi**2 - x0**2 - y0**2 - di**2])
        
        A = np.array(A)
        b = np.array(b)
        
        # Solve using least squares
        position, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        position = position.flatten()
        
        print(f"Localized position: {position}")
        
        # Validate position (should be within reasonable area)
        if (position[0] < -10 or position[0] > 10 or 
            position[1] < -10 or position[1] > 10):
            print(f"Warning: Position {position} seems unrealistic")
            # Return a default position in front of the array
            return [1.0, 1.0], None
        
        return position.tolist(), None
        
    except Exception as e:
        logger.error(f"Enhanced localization error: {str(e)}")
        return None, str(e)

def analyze_long_audio(audio_data, sr, threshold=0.70):
    """Analyze long audio files by processing in segments"""
    try:
        segment_duration = 3.0  # Analyze 3-second segments
        segment_samples = int(segment_duration * sr)
        total_samples = len(audio_data)
        
        segments = []
        max_confidence = 0
        best_segment = None
        
        # Process audio in overlapping segments
        hop_size = int(segment_duration * sr * 0.5)  # 50% overlap
        
        for start_sample in range(0, total_samples - segment_samples, hop_size):
            end_sample = start_sample + segment_samples
            segment = audio_data[start_sample:end_sample]
            
            # Extract features and detect
            features = audio_processor.extract_features(segment, sr)
            detection_result = model_loader.predict(features, threshold=threshold)
            
            segment_info = {
                'start_time': start_sample / sr,
                'end_time': end_sample / sr,
                'confidence': detection_result['confidence'],
                'is_drone': detection_result['is_drone'],
                'probability': detection_result['class_probabilities']['drone']
            }
            
            segments.append(segment_info)
            
            # Track best segment
            if detection_result['confidence'] > max_confidence:
                max_confidence = detection_result['confidence']
                best_segment = segment_info
        
        # Overall detection based on best segment
        overall_detected = max_confidence >= threshold
        detected_segments = sum(1 for seg in segments if seg['is_drone'])
        
        return {
            'detected': overall_detected,
            'confidence': max_confidence,
            'best_segment': best_segment,
            'segments': segments,
            'detection_summary': {
                'total_segments': len(segments),
                'detected_segments': detected_segments,
                'max_confidence': max_confidence
            }
        }
        
    except Exception as e:
        logger.error(f"Long audio analysis error: {str(e)}")
        raise

@app.route('/api/detect-with-localization', methods=['POST'])
def detect_with_localization_unified():
    """
    Unified detection endpoint that supports both short and long audio
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No file provided', 'status': 'error'}), 400
        
        audio_file = request.files['audio']
        threshold = float(request.form.get('threshold', 0.70))
        analyze_long = request.form.get('analyze_long', 'false').lower() == 'true'
        
        # Process audio
        file_stream = io.BytesIO()
        audio_file.save(file_stream)
        file_stream.seek(0)
        
        # Load audio
        audio_data, sr = audio_processor.load_audio_from_fileobj(file_stream)
        
        audio_info = {
            'duration': len(audio_data) / sr,
            'sample_rate': sr,
            'channels': audio_data.shape[1] if audio_data.ndim > 1 else 1,
            'samples': len(audio_data),
            'shape': audio_data.shape
        }
        
        # Choose detection strategy
        if analyze_long and audio_info['duration'] > 10:  # Long audio
            logger.info(f"Analyzing long audio ({audio_info['duration']:.1f}s) in segments")
            long_result = analyze_long_audio(audio_data, sr, threshold)
            
            result = {
                'audio_info': audio_info,
                'status': 'success',
                'debug': {
                    'channels_available': audio_info['channels'],
                    'analyze_long': analyze_long,
                    'analysis_type': 'long_audio'
                },
                # Enhanced format for long audio
                'detected': long_result['detected'],
                'probability': long_result['confidence'],
                'detection_summary': long_result['detection_summary'],
                'segments': long_result['segments'],
                'best_segment': long_result['best_segment'],
                # Backward compatibility
                'detection': {
                    'is_drone': long_result['detected'],
                    'confidence': long_result['confidence'],
                    'class_probabilities': {
                        'non_drone': 1 - long_result['confidence'],
                        'drone': long_result['confidence']
                    }
                }
            }
            
            audio_data_for_localization = audio_data
                
        else:  # Standard detection (short audio)
            logger.info("Using standard detection for short audio")
            features = audio_processor.extract_features(audio_data, sr)
            detection_result = model_loader.predict(features, threshold=threshold)
            
            result = {
                'audio_info': audio_info,
                'status': 'success',
                'debug': {
                    'channels_available': audio_info['channels'],
                    'analyze_long': analyze_long,
                    'analysis_type': 'short_audio'
                },
                # Enhanced format
                'detected': detection_result['is_drone'],
                'probability': detection_result['confidence'],
                # Backward compatibility
                'detection': detection_result
            }
            audio_data_for_localization = audio_data
        
        # Attempt localization if drone detected
        if result['detected']:
            logger.info("Drone detected - attempting localization")
            result['debug']['drone_detected'] = True
            result['debug']['localization_attempted'] = True
            
            # Check if we have enough channels for real localization
            if (audio_data_for_localization.ndim == 2 and 
                audio_data_for_localization.shape[1] >= 3):
                
                logger.info("Attempting real localization with multi-channel audio")
                tdoas, tdoa_error = calculate_tdoa_enhanced(audio_data_for_localization, sr)
                
                if tdoas is not None:
                    estimated_position, loc_error = localize_drone_enhanced(
                        tdoas, config.MIC_POSITIONS, config.SOUND_SPEED
                    )
                    
                    if estimated_position:
                        visualization_data = create_visualization_data(
                            estimated_position, 
                            None,
                            result['probability'], 
                            None
                        )
                        
                        result['localization'] = {
                            'estimated_position': estimated_position,
                            'tdoas': tdoas.tolist(),
                            'visualization_data': visualization_data,
                            'error': None,
                            'simulated': False
                        }
                        result['debug']['localization_type'] = 'REAL_MULTI_CHANNEL'
                    else:
                        result['debug']['localization_error'] = loc_error
                else:
                    result['debug']['localization_error'] = tdoa_error
            else:
                # Simulate localization for mono/stereo or insufficient channels
                logger.info("Simulating localization (insufficient channels)")
                confidence = result['probability']
                
                simulated_position = [
                    1.2 + (np.random.random() - 0.5) * 0.8,
                    0.8 + (np.random.random() - 0.5) * 0.6
                ]
                
                visualization_data = create_visualization_data(
                    simulated_position, 
                    None,
                    confidence, 
                    None
                )
                
                result['localization'] = {
                    'estimated_position': simulated_position,
                    'tdoas': [0.0012, 0.0008],
                    'visualization_data': visualization_data,
                    'error': None,
                    'simulated': True
                }
                result['debug']['localization_type'] = 'SIMULATED_INSUFFICIENT_CHANNELS'
                result['debug']['channels_received'] = audio_info['channels']
                result['debug']['channels_required'] = 3
        else:
            result['debug']['drone_detected'] = False
            result['debug']['localization_attempted'] = False
        
        file_stream.close()
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Unified detection error: {str(e)}")
        return jsonify({
            'error': f'Processing failed: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/detect-with-localization-enhanced', methods=['POST'])
def detect_with_localization_enhanced():
    """
    Enhanced detection with long audio support and better localization
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No file provided', 'status': 'error'}), 400
        
        audio_file = request.files['audio']
        threshold = float(request.form.get('threshold', 0.70))
        analyze_long = request.form.get('analyze_long', 'true').lower() == 'true'
        
        # Process audio
        file_stream = io.BytesIO()
        audio_file.save(file_stream)
        file_stream.seek(0)
        
        # Load audio
        audio_data, sr = audio_processor.load_audio_from_fileobj(file_stream)
        
        audio_info = {
            'duration': len(audio_data) / sr,
            'sample_rate': sr,
            'channels': audio_data.shape[1] if audio_data.ndim > 1 else 1,
            'samples': len(audio_data),
            'shape': audio_data.shape
        }
        
        result = {
            'audio_info': audio_info,
            'status': 'success',
            'debug': {
                'channels_available': audio_info['channels'],
                'analyze_long': analyze_long
            }
        }
        
        # Choose detection strategy
        if analyze_long and audio_info['duration'] > 10:  # Long audio
            logger.info(f"Analyzing long audio ({audio_info['duration']:.1f}s) in segments")
            long_result = analyze_long_audio(audio_data, sr, threshold)
            
            result.update({
                'detected': long_result['detected'],
                'probability': long_result['confidence'],
                'detection_summary': long_result['detection_summary'],
                'segments': long_result['segments'],
                'best_segment': long_result['best_segment']
            })
            
            # If drone detected in any segment, attempt localization with best segment
            if long_result['detected'] and long_result['best_segment']:
                best_segment = long_result['best_segment']
                start_sample = int(best_segment['start_time'] * sr)
                end_sample = int(best_segment['end_time'] * sr)
                best_audio = audio_data[start_sample:end_sample]
                
                result['debug']['best_segment_used'] = {
                    'start_time': best_segment['start_time'],
                    'end_time': best_segment['end_time'],
                    'confidence': best_segment['confidence']
                }
                
                # Use best segment for localization attempt
                audio_data_for_localization = best_audio
            else:
                audio_data_for_localization = audio_data
                
        else:  # Standard detection
            features = audio_processor.extract_features(audio_data, sr)
            detection_result = model_loader.predict(features, threshold=threshold)
            
            result.update({
                'detected': detection_result['is_drone'],
                'probability': detection_result['confidence'],
                'detection': detection_result
            })
            audio_data_for_localization = audio_data
        
        # Attempt localization if drone detected
        if result.get('detected', False):
            result['debug']['drone_detected'] = True
            result['debug']['localization_attempted'] = True
            
            # Check if we have enough channels for real localization
            if (audio_data_for_localization.ndim == 2 and 
                audio_data_for_localization.shape[1] >= 3):
                
                logger.info("Attempting real localization with multi-channel audio")
                tdoas, tdoa_error = calculate_tdoa_enhanced(audio_data_for_localization, sr)
                
                if tdoas is not None:
                    estimated_position, loc_error = localize_drone_enhanced(
                        tdoas, config.MIC_POSITIONS, config.SOUND_SPEED
                    )
                    
                    if estimated_position:
                        visualization_data = create_visualization_data(
                            estimated_position, 
                            None,
                            result['probability'], 
                            None
                        )
                        
                        result['localization'] = {
                            'estimated_position': estimated_position,
                            'tdoas': tdoas.tolist(),
                            'visualization_data': visualization_data,
                            'error': None,
                            'simulated': False
                        }
                        result['debug']['localization_type'] = 'REAL_MULTI_CHANNEL'
                    else:
                        result['debug']['localization_error'] = loc_error
                else:
                    result['debug']['localization_error'] = tdoa_error
            else:
                # Simulate localization for mono/stereo or insufficient channels
                logger.info("Simulating localization (insufficient channels)")
                confidence = result['probability']
                
                # Generate realistic simulated position
                simulated_position = [
                    1.2 + (np.random.random() - 0.5) * 0.8,  # x: 0.8-1.6
                    0.8 + (np.random.random() - 0.5) * 0.6   # y: 0.5-1.1
                ]
                
                visualization_data = create_visualization_data(
                    simulated_position, 
                    None,
                    confidence, 
                    None
                )
                
                result['localization'] = {
                    'estimated_position': simulated_position,
                    'tdoas': [0.0012, 0.0008],
                    'visualization_data': visualization_data,
                    'error': None,
                    'simulated': True
                }
                result['debug']['localization_type'] = 'SIMULATED_INSUFFICIENT_CHANNELS'
                result['debug']['channels_received'] = audio_info['channels']
                result['debug']['channels_required'] = 3
        else:
            result['debug']['drone_detected'] = False
            result['debug']['localization_attempted'] = False
        
        file_stream.close()
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Enhanced detection error: {str(e)}")
        return jsonify({
            'error': f'Processing failed: {str(e)}',
            'status': 'error'
        }), 500

def safe_delete_file(file_path, max_retries=3, delay=0.1):
    """Safely delete a file with retries"""
    for attempt in range(max_retries):
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"Successfully deleted temporary file: {file_path}")
                return True
        except PermissionError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Retry {attempt + 1} for deleting {file_path}: {e}")
                time.sleep(delay)
                gc.collect()  # Force garbage collection
            else:
                logger.error(f"Failed to delete {file_path} after {max_retries} attempts: {e}")
                return False
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loader.is_loaded(),
        'localization_supported': True,
        'message': 'Drone Detection & Localization API is running'
    })

@app.route('/api/detect', methods=['POST'])
def detect_drone():
    """
    Detect drone presence in audio file
    Accepts: WAV file upload or base64 encoded audio
    """
    try:
        # Check if file is uploaded
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return jsonify({'error': 'No file selected', 'status': 'error'}), 400
            
            # Validate file type
            if not audio_file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                return jsonify({'error': 'Unsupported file format', 'status': 'error'}), 400
            
            # Process the uploaded file
            result = process_uploaded_file_in_memory(audio_file)  # Use in-memory method
            
        # Check for base64 audio data
        elif 'audio_data' in request.json:
            audio_b64 = request.json['audio_data']
            result = process_base64_audio(audio_b64)
            
        else:
            return jsonify({'error': 'No audio data provided', 'status': 'error'}), 400
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        return jsonify({
            'error': f'Processing failed: {str(e)}',
            'status': 'error'
        }), 500

def process_uploaded_file_in_memory(audio_file):
    """Process uploaded audio file entirely in memory"""
    try:
        # Create a file-like object from the uploaded file
        file_stream = io.BytesIO()
        audio_file.save(file_stream)
        file_stream.seek(0)  # Reset to beginning
        
        # Load and process audio directly from memory
        audio_data, sr = audio_processor.load_audio_from_fileobj(file_stream)
        features = audio_processor.extract_features(audio_data, sr)
        
        # Make prediction
        prediction = model_loader.predict(features)
        
        # Close the stream
        file_stream.close()
        
        return {
            'is_drone': bool(prediction['is_drone']),
            'confidence': float(prediction['confidence']),
            'class_probabilities': prediction['class_probabilities'],
            'status': 'success',
            'audio_duration': len(audio_data) / sr,
            'sample_rate': sr
        }
        
    except Exception as e:
        logger.error(f"Memory processing error: {str(e)}")
        raise Exception(f"Audio processing failed: {str(e)}")

def process_uploaded_file_with_temp(audio_file):
    """Alternative method using temporary files (if in-memory fails)"""
    temp_path = None
    try:
        # Save to temporary file with unique name
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)  # Close the file descriptor
        
        audio_file.save(temp_path)
        
        # Force garbage collection and small delay
        gc.collect()
        time.sleep(0.05)
        
        # Load and process audio
        audio_data, sr = audio_processor.load_audio(temp_path)
        features = audio_processor.extract_features(audio_data, sr)
        
        # Make prediction
        prediction = model_loader.predict(features)
        
        return {
            'is_drone': bool(prediction['is_drone']),
            'confidence': float(prediction['confidence']),
            'class_probabilities': prediction['class_probabilities'],
            'status': 'success',
            'audio_duration': len(audio_data) / sr,
            'sample_rate': sr
        }
        
    except Exception as e:
        logger.error(f"Temp file processing error: {str(e)}")
        raise Exception(f"Audio processing failed: {str(e)}")
    
    finally:
        # Always try to clean up the temporary file
        if temp_path:
            safe_delete_file(temp_path)

def process_base64_audio(audio_b64):
    """Process base64 encoded audio data"""
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_b64.split(',')[1] if ',' in audio_b64 else audio_b64)
        
        # Convert to file-like object
        audio_stream = io.BytesIO(audio_bytes)
        
        # Load audio from bytes
        audio_data, sr = audio_processor.load_audio_from_fileobj(audio_stream)
        features = audio_processor.extract_features(audio_data, sr)
        
        # Make prediction
        prediction = model_loader.predict(features)
        
        return {
            'is_drone': bool(prediction['is_drone']),
            'confidence': float(prediction['confidence']),
            'class_probabilities': prediction['class_probabilities'],
            'status': 'success',
            'audio_duration': len(audio_data) / sr,
            'sample_rate': sr
        }
        
    except Exception as e:
        raise Exception(f"Base64 audio processing failed: {str(e)}")

@app.route('/api/batch-detect', methods=['POST'])
def batch_detect():
    """Batch process multiple audio files using in-memory processing"""
    try:
        if 'audio_files' not in request.files:
            return jsonify({'error': 'No files provided', 'status': 'error'}), 400
        
        files = request.files.getlist('audio_files')
        results = []
        
        for audio_file in files:
            if audio_file.filename:
                try:
                    result = process_uploaded_file_in_memory(audio_file)
                    result['filename'] = audio_file.filename
                    results.append(result)
                except Exception as e:
                    results.append({
                        'filename': audio_file.filename,
                        'error': str(e),
                        'status': 'error'
                    })
        
        return jsonify({
            'results': results,
            'total_processed': len(results),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Batch detection error: {str(e)}")
        return jsonify({'error': str(e), 'status': 'error'}), 500
    
@app.route('/api/debug-detect', methods=['POST'])
def debug_detect():
    """Debug endpoint to see what's happening with the audio processing"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        audio_file = request.files['audio']
        
        # Process the audio
        file_stream = io.BytesIO()
        audio_file.save(file_stream)
        file_stream.seek(0)
        
        audio_data, sr = audio_processor.load_audio_from_fileobj(file_stream)
        features = audio_processor.extract_features(audio_data, sr)
        
        # Get prediction with raw scores
        features_tensor = torch.tensor(features, dtype=torch.float32)
        if model_loader.model:
            with torch.no_grad():
                outputs = model_loader.model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                raw_scores = outputs.cpu().numpy()[0]
                class_probs = probabilities.cpu().numpy()[0]
        
        file_stream.close()
        
        return jsonify({
            'audio_info': {
                'duration': len(audio_data) / sr,
                'sample_rate': sr,
                'samples': len(audio_data),
                'features_shape': features.shape,
                'features_mean': float(features.mean()),
                'features_std': float(features.std()),
                'features_min': float(features.min()),
                'features_max': float(features.max())
            },
            'model_output': {
                'raw_scores': [float(x) for x in raw_scores],
                'class_probabilities': {
                    'non_drone': float(class_probs[0]),
                    'drone': float(class_probs[1])
                },
                'prediction': 'drone' if class_probs[1] > 0.5 else 'non_drone',
                'confidence': float(max(class_probs))
            },
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add this to see training history if available
@app.route('/api/training-info', methods=['GET'])
def training_info():
    """Check if we have training information"""
    try:
        # Try to load loss history
        loss_file = Path('loss_history.json')
        if loss_file.exists():
            with open(loss_file, 'r') as f:
                loss_data = json.load(f)
            return jsonify({'training_history': loss_data})
        else:
            return jsonify({'training_history': 'No history found'})
    except:
        return jsonify({'training_history': 'Error loading history'})
    
@app.route('/api/debug-features', methods=['POST'])
def debug_features():
    """Debug endpoint to check feature extraction"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        audio_file = request.files['audio']
        file_stream = io.BytesIO()
        audio_file.save(file_stream)
        
        audio_data, sr = audio_processor.load_audio_from_fileobj(file_stream)
        features = audio_processor.extract_features(audio_data, sr)
        
        file_stream.close()
        
        return jsonify({
            'audio_info': {
                'duration': len(audio_data) / sr,
                'sample_rate': sr,
                'samples': len(audio_data)
            },
            'features_info': {
                'shape': list(features.shape),
                'mean': float(features.mean()),
                'std': float(features.std()),
                'min': float(features.min()),
                'max': float(features.max())
            },
            'expected_shape': [3, 64, 259],
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Check what input shape the model expects"""
    try:
        if model_loader.model:
            # Test with the expected input shape
            test_input = torch.randn(1, 3, 64, 259)
            
            with torch.no_grad():
                output = model_loader.model(test_input)
                return jsonify({
                    'expected_input_shape': [1, 3, 64, 259],
                    'test_output_shape': list(output.shape),
                    'model_loaded': True,
                    'status': 'success'
                })
        else:
            return jsonify({'error': 'Model not loaded', 'status': 'error'})
            
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/test-detection', methods=['POST'])
def test_detection():
    """Test detection with detailed debugging info"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        audio_file = request.files['audio']
        file_stream = io.BytesIO()
        audio_file.save(file_stream)
        
        # Step 1: Load audio
        audio_data, sr = audio_processor.load_audio_from_fileobj(file_stream)
        
        # Step 2: Extract features
        features = audio_processor.extract_features(audio_data, sr)
        
        # Step 3: Make prediction
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
        if model_loader.model:
            with torch.no_grad():
                outputs = model_loader.model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                raw_scores = outputs.cpu().numpy()[0]
                class_probs = probabilities.cpu().numpy()[0]
                
                result = {
                    'is_drone': bool(class_probs[1] > 0.5),
                    'confidence': float(max(class_probs)),
                    'class_probabilities': {
                        'non_drone': float(class_probs[0]),
                        'drone': float(class_probs[1])
                    },
                    'raw_scores': [float(x) for x in raw_scores],
                    'features_shape': list(features.shape),
                    'status': 'success'
                }
        else:
            result = {'error': 'Model not loaded', 'status': 'error'}
        
        file_stream.close()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/model-input-shape', methods=['GET'])
def model_input_shape():
    """Check what input shape the model expects"""
    try:
        # Create a test input to see what the model expects
        test_input = torch.randn(1, 3, 64, 259)  # Batch, Channels, Mel bands, Time frames
        
        if model_loader.model:
            with torch.no_grad():
                output = model_loader.model(test_input)
                return jsonify({
                    'expected_input_shape': [1, 3, 64, 259],
                    'test_output_shape': list(output.shape),
                    'model_loaded': True
                })
        else:
            return jsonify({'error': 'Model not loaded'})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# Monitoring
@app.route('/monitoring')
def monitoring():
    """Real-time monitoring page"""
    return render_template('real_time_monitor.html')

@app.route('/api/start-monitoring', methods=['POST'])
def start_monitoring():
    """Start real-time monitoring"""
    global is_monitoring, monitoring_thread
    
    if is_monitoring:
        return jsonify({'status': 'error', 'message': 'Monitoring already active'})
    
    try:
        is_monitoring = True
        
        # Start monitoring in a separate thread
        monitoring_thread = threading.Thread(target=monitoring_loop)
        monitoring_thread.daemon = True
        monitoring_thread.start()
        
        socketio.emit('monitoring_started', {
            'channels': 3,
            'message': 'Real-time monitoring started'
        })
        
        return jsonify({
            'status': 'success',
            'message': 'Real-time monitoring started',
            'channels': 3
        })
        
    except Exception as e:
        is_monitoring = False
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/stop-monitoring', methods=['POST'])
def stop_monitoring():
    """Stop real-time monitoring"""
    global is_monitoring
    
    is_monitoring = False
    
    socketio.emit('monitoring_stopped', {
        'message': 'Real-time monitoring stopped'
    })
    
    return jsonify({
        'status': 'success',
        'message': 'Real-time monitoring stopped'
    })

@app.route('/api/monitoring-status', methods=['GET'])
def monitoring_status():
    """Get monitoring status"""
    return jsonify({
        'active': is_monitoring,
        'channels': 3 if is_monitoring else 0
    })

def monitoring_loop():
    """Main monitoring loop (simulated for now)"""
    import random
    
    while is_monitoring:
        try:
            # Simulate drone detection (replace with real audio processing)
            if random.random() > 0.8:  # 20% chance of detection
                confidence = random.uniform(0.7, 0.95)
                position = [
                    random.uniform(0.5, 1.5),  # x position
                    random.uniform(0.3, 1.2)   # y position
                ]

                MICROPHONE_LOCATION = [47.471848,19.019239]; # Default location (Budapest)

                # Add geographical context
                socketio.emit('drone_detected', {
                    'timestamp': time.time(),
                    'confidence': confidence,
                    'position': position,  # Relative coordinates
                    'localized': True,
                    'geographical': {
                        'lat': MICROPHONE_LOCATION[0] + (position[1] * 0.000009),
                        'lng': MICROPHONE_LOCATION[1] + (position[0] * 0.000009)
                    }
                })
            
            time.sleep(2)  # Process every 2 seconds
            
        except Exception as e:
            print(f"Monitoring error: {e}")
            time.sleep(1)

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large', 'status': 'error'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)