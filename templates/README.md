# 🚁 Drone Detection & Localization System

A comprehensive web-based system for detecting drone presence in audio recordings and visualizing their estimated positions using microphone array technology.

## 🌟 Features

### 🎯 Core Capabilities
- **AI-Powered Detection**: Deep learning model for accurate drone audio classification
- **Real-time Localization**: Estimate drone position using Time Difference of Arrival (TDOA)
- **Multi-format Support**: Process WAV, MP3, M4A, and FLAC audio files
- **Long Audio Analysis**: Segment-based processing for files longer than 10 seconds
- **Interactive Visualization**: Plotly-based maps showing drone positions and microphone arrays

### 🎨 User Interface
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Feedback**: Live confidence meters and probability displays
- **Audio Preview**: Built-in audio player for uploaded files
- **Batch Processing**: Analyze multiple files simultaneously
- **Adjustable Sensitivity**: Customizable detection thresholds

### 🔧 Technical Features
- **Multi-channel Support**: 3+ channel audio for accurate localization
- **Simulated Localization**: Fallback visualization for mono/stereo files
- **Segment Analysis**: Identify drone activity in specific time segments
- **Debug Mode**: Detailed technical information for development
- **RESTful API**: Clean API endpoints for integration

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch
- Flask
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Raphtildai/drone-detection-api.git
   cd drone-detection-api
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the application**
   ```bash
   python app.py
   ```

4. **Access the web interface**
   ```
   Open http://localhost:5000 in your browser
   ```

## 📁 Project Structure

```
drone-detection-api/
├── app.py                 # Main Flask application
├── model_loader.py        # Neural network model management
├── audio_processor.py     # Audio feature extraction
├── requirements.txt       # Python dependencies
├── static/
│   ├── style.css         # Frontend styling
│   └── script.js         # Frontend functionality
├── templates/
│   └── index.html        # Main web interface
├── models/
│   └── best_model.pth    # Trained drone detection model
└── README.md
```

## 🎮 Usage Guide

### Basic Detection

1. **Upload Audio File**
   - Click "Choose File" and select an audio recording
   - Supported formats: WAV, MP3, M4A, FLAC
   - Preview audio using the built-in player

2. **Configure Detection Settings**
   - Adjust sensitivity using the threshold slider (0.1-1.0)
   - Choose from preset sensitivity levels:
     - High Sensitivity (0.50): More detections, potential false positives
     - Balanced (0.70): Recommended for most cases
     - Low Sensitivity (0.85): Fewer detections, higher confidence

3. **Enable Long Audio Analysis** (for files >10 seconds)
   - Check "Analyze entire file" to process long recordings
   - System analyzes 3-second segments with 50% overlap
   - Identifies specific time segments with drone activity

4. **View Results**
   - **Detection Status**: Clear indication of drone presence
   - **Confidence Meter**: Visual representation of detection certainty
   - **Localization Map**: Interactive plot showing estimated position
   - **Segment Analysis**: Detailed breakdown for long audio files

### Advanced Features

#### Multi-channel Localization
For accurate position estimation:
- Use 3+ channel audio recordings
- Record with microphone arrays
- Position microphones in known geometric patterns
- System calculates Time Difference of Arrival (TDOA) between channels

#### Batch Processing
- Upload multiple files simultaneously
- Process all files in one operation
- View consolidated results with individual file status

#### API Access
```bash
# Health check
curl http://localhost:5000/health

# Single file detection
curl -X POST -F "audio=@drone_audio.wav" http://localhost:5000/api/detect

# Batch processing
curl -X POST -F "audio_files=@file1.wav" -F "audio_files=@file2.wav" http://localhost:5000/api/batch-detect
```

## 🔬 Technical Details

### Detection Algorithm
- **Model Architecture**: CNN-based classifier (3, 64, 259 input shape)
- **Feature Extraction**: Mel-spectrogram analysis
- **Confidence Scoring**: Softmax probability outputs
- **Threshold Tuning**: Adjustable detection sensitivity

### Localization Method
- **TDOA Calculation**: Cross-correlation between microphone channels
- **Position Estimation**: Linear least squares solving
- **Microphone Array**: Default 3-mic L-shaped configuration
- **Sound Speed**: 343 m/s (standard atmospheric conditions)

### Audio Processing
- **Sample Rate**: 22.05 kHz standard processing
- **Segment Duration**: 3-second analysis windows
- **Overlap**: 50% between consecutive segments
- **Normalization**: Automatic audio level adjustment

## 🛠️ Configuration

### Microphone Array Setup
Default configuration (modifiable in `app.py`):
```python
MIC_POSITIONS = [
    [0, 0],    # Mic 1 - Reference (origin)
    [0.5, 0],  # Mic 2 - 0.5m right
    [0, 0.5]   # Mic 3 - 0.5m forward
]
```

### Detection Thresholds
- **Default**: 0.70 (70% confidence)
- **Range**: 0.10 to 1.00
- **Recommended**: 0.50-0.85 depending on environment

## 📊 Output Interpretation

### Detection Results
- **🚁 DRONE DETECTED**: High confidence of drone presence
- **🌳 NO DRONE DETECTED**: Background noise or no drone activity
- **Confidence Values**: 0-100% certainty of classification

### Localization Results
- **🎯 Real Localization**: Based on multi-channel TDOA calculations
- **🔄 Simulated Localization**: Estimated position for mono/stereo files
- **Position Format**: (X, Y) coordinates in meters relative to mic array

### Segment Analysis (Long Audio)
- **Detected Segments**: Number of time segments with drone activity
- **Best Detection**: Highest-confidence segment with timestamps
- **Total Analysis**: Complete file coverage with overlapping windows

## 🔧 Troubleshooting

### Common Issues

1. **"No localization data available"**
   - Cause: Mono or stereo audio file
   - Solution: Use 3+ channel recordings for accurate localization

2. **Low detection confidence**
   - Cause: Background noise or weak drone signals
   - Solution: Lower detection threshold or improve recording quality

3. **Long processing time**
   - Cause: Large audio files or system load
   - Solution: Enable segment analysis for faster processing

4. **Visualization not displaying**
   - Cause: Browser compatibility or Plotly loading issues
   - Solution: Check browser console for errors, refresh page

### Performance Tips
- Use WAV format for fastest processing
- Keep files under 5 minutes for quick analysis
- Enable long audio analysis for files >30 seconds
- Close other browser tabs during processing

## 🚀 API Reference

### Endpoints

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/` | GET | Web interface | - |
| `/health` | GET | System status | - |
| `/api/detect` | POST | Single file detection | `audio` (file) |
| `/api/detect-with-localization` | POST | Enhanced detection | `audio`, `threshold`, `analyze_long` |
| `/api/batch-detect` | POST | Multiple file processing | `audio_files` (multiple) |
| `/api/debug-detect` | POST | Detailed analysis info | `audio` (file) |

### Response Format
```json
{
  "status": "success",
  "detected": true,
  "probability": 0.95,
  "localization": {
    "estimated_position": [1.2, 0.8],
    "simulated": false,
    "visualization_data": {...}
  },
  "audio_info": {
    "duration": 45.2,
    "sample_rate": 22050,
    "channels": 3
  }
}
```

## 🤝 Contributing

I welcome contributions! Just let me know by contacting me in any of my contact options

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Areas for Improvement
- Additional model architectures
- Enhanced localization algorithms
- Real-time streaming support
- Mobile app development
- Cloud deployment options

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for deep learning framework
- Plotly for interactive visualization
- Flask community for web framework
- Contributors and testers

## 📞 Support

For support and questions:
- 📧 Email: raphael@tildaitech.co.ke
- 🐛 Issues: [GitHub Issues](https://github.com/Raphtildai/drone-detection-api/issues)
- 📚 Documentation: [Project Wiki](https://github.com/Raphtildai/drone-detection-api/wiki)

---

**Note**: This system is designed for research and educational purposes. Always comply with local regulations regarding drone detection and privacy laws.

*Last updated: ${new Date().toLocaleDateString()}*