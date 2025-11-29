// DOM elements
const audioFileInput = document.getElementById('audioFile');
const uploadArea = document.getElementById('uploadArea');
const selectedFileDiv = document.getElementById('selectedFile');
const fileNameSpan = document.getElementById('fileName');
const audioPreview = document.getElementById('audioPreview');
const detectBtn = document.getElementById('detectBtn');
const resultsSection = document.getElementById('resultsSection');
const resultCard = document.getElementById('resultCard');
const resultIcon = document.getElementById('resultIcon');
const resultText = document.getElementById('resultText');
const confidenceFill = document.getElementById('confidenceFill');
const confidenceValue = document.getElementById('confidenceValue');
const probNonDrone = document.getElementById('probNonDrone');
const probDrone = document.getElementById('probDrone');
const audioDuration = document.getElementById('audioDuration');
const sampleRate = document.getElementById('sampleRate');
const batchFilesInput = document.getElementById('batchFiles');
const batchCount = document.getElementById('batchCount');
const batchDetectBtn = document.getElementById('batchDetectBtn');
const batchResults = document.getElementById('batchResults');

// Event listeners
audioFileInput.addEventListener('change', handleFileSelect);
batchFilesInput.addEventListener('change', handleBatchFileSelect);

// Drag and drop support
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleAudioFile(files[0]);
    }
});

uploadArea.addEventListener('click', () => {
    audioFileInput.click();
});

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleAudioFile(file);
    }
}

function handleAudioFile(file) {
    // Validate file type
    const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/x-m4a', 'audio/flac'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(wav|mp3|m4a|flac)$/i)) {
        alert('Please select a valid audio file (WAV, MP3, M4A, FLAC)');
        return;
    }

    // Update UI
    fileNameSpan.textContent = file.name;
    selectedFileDiv.style.display = 'block';
    detectBtn.disabled = false;

    // Create preview URL
    const url = URL.createObjectURL(file);
    audioPreview.src = url;

    // Clear previous results
    clearResults();
}

function handleBatchFileSelect(e) {
    const files = e.target.files;
    batchCount.textContent = `${files.length} files selected`;
    batchDetectBtn.disabled = files.length === 0;
}

function clearFile() {
    audioFileInput.value = '';
    selectedFileDiv.style.display = 'none';
    detectBtn.disabled = true;
    audioPreview.src = '';
    clearResults();
}

function clearAll() {
    clearFile();
    clearResults();
    batchFilesInput.value = '';
    batchCount.textContent = '0 files selected';
    batchDetectBtn.disabled = true;
    batchResults.innerHTML = '';
}

function clearResults() {
    resultsSection.style.display = 'none';
    resultIcon.textContent = '🔍';
    resultText.textContent = 'Analyzing...';
    confidenceFill.style.width = '0%';
    confidenceValue.textContent = '0%';
    probNonDrone.textContent = '0%';
    probDrone.textContent = '0%';
    audioDuration.textContent = '-';
    sampleRate.textContent = '-';
}

async function analyzeAudio() {
    const fileInput = document.getElementById('audioFile');
    const threshold = document.getElementById('thresholdSlider').value;
    const analyzeLong = document.getElementById('analyzeLong').checked;
    
    if (!fileInput.files[0]) {
        alert('Please select an audio file first');
        return;
    }

    const detectBtn = document.getElementById('detectBtn');
    const originalText = detectBtn.textContent;
    
    detectBtn.disabled = true;
    detectBtn.textContent = 'Processing...';

    // Hide previous results
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('visualizationSection').style.display = 'none';
    document.getElementById('debugSection').style.display = 'none';

    const formData = new FormData();
    formData.append('audio', fileInput.files[0]);
    formData.append('threshold', threshold);
    formData.append('analyze_long', analyzeLong);

    try {
        console.log('Sending request to enhanced endpoint...');
        
        // Make sure you're calling the correct endpoint
        const response = await fetch('/api/detect-with-localization', {  // This should match your Flask route
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const result = await response.json();
        console.log('Enhanced server response:', result);
        
        if (result.status === 'success') {
            displayEnhancedResults(result);  // Use the corrected function
            
            // Always show debug information
            document.getElementById('debugSection').style.display = 'block';
            document.getElementById('debugInfo').textContent = JSON.stringify(result, null, 2);
            
            // Show localization if available
            if (result.localization && result.localization.visualization_data) {
                console.log('Localization data available:', result.localization);
                
                document.getElementById('localizationInfo').style.display = 'block';
                document.getElementById('positionInfo').textContent = 
                    `(${result.localization.estimated_position[0].toFixed(2)}, ${result.localization.estimated_position[1].toFixed(2)}) m`;
                
                // Show notice if data is simulated
                if (result.localization.simulated) {
                    document.getElementById('errorInfo').innerHTML = 
                        '<span style="color: orange; font-weight: bold;">SIMULATED DATA</span><br>' +
                        `<small>Channels: ${result.audio_info.channels}/3 needed</small>`;
                } else if (result.localization.error) {
                    document.getElementById('errorInfo').textContent = 
                        `${result.localization.error.toFixed(3)} m`;
                    document.getElementById('errorInfo').style.color = '';
                } else {
                    document.getElementById('errorInfo').textContent = 'N/A (no true position provided)';
                    document.getElementById('errorInfo').style.color = '';
                }
                
                // Show and create visualization
                document.getElementById('visualizationSection').style.display = 'block';
                setTimeout(() => {
                    createLocalizationPlot(result.localization.visualization_data);
                }, 100);
            } else {
                console.log('No localization data available');
                document.getElementById('visualizationSection').style.display = 'none';
                document.getElementById('localizationInfo').style.display = 'none';
            }
        } else {
            throw new Error(result.error || 'Detection failed');
        }
    } catch (error) {
        console.error('Detection error:', error);
        alert('Detection failed: ' + error.message);
    } finally {
        detectBtn.disabled = false;
        detectBtn.textContent = originalText;
    }
}

async function analyzeAudioEnhanced() {
    const fileInput = document.getElementById('audioFile');
    const threshold = document.getElementById('thresholdSlider').value;
    const analyzeLong = document.getElementById('analyzeLong').checked;
    
    if (!fileInput.files[0]) {
        alert('Please select an audio file first');
        return;
    }

    const detectBtn = document.getElementById('detectBtn');
    const originalText = detectBtn.textContent;
    
    detectBtn.disabled = true;
    detectBtn.textContent = 'Processing...';

    // Hide previous results
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('visualizationSection').style.display = 'none';
    // document.getElementById('debugSection').style.display = 'none';

    const formData = new FormData();
    formData.append('audio', fileInput.files[0]);
    formData.append('threshold', threshold);
    formData.append('analyze_long', analyzeLong);

    try {
        console.log('Sending request to enhanced endpoint...');
        const response = await fetch('/api/detect-with-localization-enhanced', {  // Call enhanced endpoint
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const result = await response.json();
        console.log('Enhanced server response:', result);
        
        if (result.status === 'success') {
            displayEnhancedResults(result);  // Use enhanced display function
            
            // Always show debug information
            document.getElementById('debugSection').style.display = 'block';
            // document.getElementById('debugInfo').textContent = JSON.stringify(result, null, 2);
            // Only log to console, don't show in UI
            console.log('Full server response for debugging:', result);

            // Show localization if available
            if (result.localization && result.localization.visualization_data) {
                console.log('Localization data available:', result.localization);
                
                document.getElementById('localizationInfo').style.display = 'block';
                document.getElementById('positionInfo').textContent = 
                    `(${result.localization.estimated_position[0].toFixed(2)}, ${result.localization.estimated_position[1].toFixed(2)}) m`;
                
                // Show notice if data is simulated
                if (result.localization.simulated) {
                    document.getElementById('errorInfo').innerHTML = 
                        '<span style="color: orange; font-weight: bold;">SIMULATED DATA</span><br>' +
                        `<small>Channels: ${result.audio_info.channels}/3 needed</small>`;
                } else if (result.localization.error) {
                    document.getElementById('errorInfo').textContent = 
                        `${result.localization.error.toFixed(3)} m`;
                    document.getElementById('errorInfo').style.color = '';
                } else {
                    document.getElementById('errorInfo').textContent = 'N/A (no true position provided)';
                    document.getElementById('errorInfo').style.color = '';
                }
                
                // Show and create visualization
                document.getElementById('visualizationSection').style.display = 'block';
                setTimeout(() => {
                    createLocalizationPlot(result.localization.visualization_data);
                }, 100);
            } else {
                console.log('No localization data available');
                document.getElementById('visualizationSection').style.display = 'none';
                document.getElementById('localizationInfo').style.display = 'none';
            }
        } else {
            throw new Error(result.error || 'Detection failed');
        }
    } catch (error) {
        console.error('Detection error:', error);
        alert('Detection failed: ' + error.message);
    } finally {
        detectBtn.disabled = false;
        detectBtn.textContent = 'Detect Drone & Localize';
    }
}

function displayEnhancedResults(result) {
    const resultsSection = document.getElementById('resultsSection');
    const resultCard = document.getElementById('resultCard');
    const resultIcon = document.getElementById('resultIcon');
    const resultText = document.getElementById('resultText');
    
    // Use the new response format - check for 'detected' field instead of 'detection.is_drone'
    const isDrone = result.detected; // Changed from result.detection.is_drone
    const confidence = result.probability; // Changed from result.detection.confidence
    
    // Clear any previous segment info
    const existingSegmentInfo = resultCard.querySelector('.segment-info');
    if (existingSegmentInfo) {
        existingSegmentInfo.remove();
    }
    
    // Update detection results
    if (isDrone) {
        resultIcon.textContent = '🚁';
        resultText.textContent = 'DRONE DETECTED';
        resultCard.style.borderLeftColor = '#e74c3c';
        resultCard.style.backgroundColor = '#fff5f5';
        
        // Show segment info if available
        if (result.segments) {
            const detectedSegments = result.detection_summary.detected_segments;
            const totalSegments = result.detection_summary.total_segments;
            resultText.textContent += ` (${detectedSegments}/${totalSegments} segments)`;
            
            if (result.best_segment) {
                const best = result.best_segment;
                const segmentInfo = document.createElement('div');
                segmentInfo.className = 'segment-info';
                segmentInfo.innerHTML = `
                    <div style="margin-top: 10px; font-size: 0.9em; color: #666;">
                        <strong>Best detection:</strong> ${best.start_time.toFixed(1)}s-${best.end_time.toFixed(1)}s 
                        (confidence: ${(best.confidence * 100).toFixed(1)}%)
                    </div>
                `;
                resultCard.querySelector('.result-details').appendChild(segmentInfo);
            }
        }
    } else {
        resultIcon.textContent = '🌳';
        resultText.textContent = 'NO DRONE DETECTED';
        resultCard.style.borderLeftColor = '#27ae60';
        resultCard.style.backgroundColor = '#f5fff5';
        
        // Show max confidence if available
        if (result.detection_summary) {
            resultText.textContent += ` (max confidence: ${(result.detection_summary.max_confidence * 100).toFixed(1)}%)`;
        }
    }
    
    // Update confidence meter
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceValue = document.getElementById('confidenceValue');
    
    const confidencePercent = confidence * 100;
    confidenceFill.style.width = `${confidencePercent}%`;
    confidenceValue.textContent = `${confidencePercent.toFixed(1)}%`;
    
    // Color confidence bar based on detection
    if (isDrone) {
        confidenceFill.style.backgroundColor = confidencePercent > 70 ? '#e74c3c' : '#f39c12';
    } else {
        confidenceFill.style.backgroundColor = '#27ae60';
    }
    
    // Update probabilities
    document.getElementById('probNonDrone').textContent = 
        `${((1 - confidence) * 100).toFixed(1)}%`;
    document.getElementById('probDrone').textContent = 
        `${(confidence * 100).toFixed(1)}%`;
    
    // Update audio info
    document.getElementById('audioDuration').textContent = 
        `${result.audio_info.duration.toFixed(2)}s`;
    document.getElementById('sampleRate').textContent = 
        `${result.audio_info.sample_rate} Hz`;
    document.getElementById('channelsInfo').textContent = 
        `${result.audio_info.channels} channel${result.audio_info.channels !== 1 ? 's' : ''}`;
    
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function analyzeBatch() {
    const files = batchFilesInput.files;
    if (files.length === 0) return;

    batchDetectBtn.innerHTML = '<div class="loading"></div>Processing...';
    batchDetectBtn.disabled = true;
    batchResults.innerHTML = '';

    try {
        const formData = new FormData();
        for (let file of files) {
            formData.append('audio_files', file);
        }

        const response = await fetch('/api/batch-detect', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.status === 'success') {
            displayBatchResults(data.results);
        } else {
            throw new Error(data.error || 'Batch processing failed');
        }

    } catch (error) {
        batchResults.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    } finally {
        batchDetectBtn.innerHTML = 'Process Batch';
        batchDetectBtn.disabled = false;
    }
}

function displayResults(data) {
    const isDrone = data.is_drone;
    const confidence = data.confidence;
    
    // Update result display
    if (isDrone) {
        resultIcon.textContent = '🚁';
        resultText.textContent = 'DRONE DETECTED';
        resultCard.style.borderLeftColor = '#e74c3c';
        resultCard.style.background = '#ffeaa7';
    } else {
        resultIcon.textContent = '✅';
        resultText.textContent = 'NO DRONE DETECTED';
        resultCard.style.borderLeftColor = '#2ecc71';
        resultCard.style.background = '#d5f4e6';
    }

    // Update confidence meter
    const confidencePercent = Math.round(confidence * 100);
    confidenceFill.style.width = `${confidencePercent}%`;
    confidenceValue.textContent = `${confidencePercent}%`;

    // Update probabilities
    probNonDrone.textContent = `${Math.round(data.class_probabilities.non_drone * 100)}%`;
    probDrone.textContent = `${Math.round(data.class_probabilities.drone * 100)}%`;

    // Update audio info
    audioDuration.textContent = `${data.audio_duration.toFixed(2)}s`;
    sampleRate.textContent = `${data.sample_rate} Hz`;
}

function displayBatchResults(results) {
    batchResults.innerHTML = '';
    
    results.forEach((result, index) => {
        const item = document.createElement('div');
        item.className = 'batch-result-item';
        
        if (result.status === 'success') {
            const icon = result.is_drone ? '🚁' : '✅';
            const confidence = Math.round(result.confidence * 100);
            item.innerHTML = `
                <div>
                    <strong>${result.filename}</strong>
                    <div style="font-size: 0.9em; color: #666;">
                        ${icon} ${result.is_drone ? 'Drone' : 'No Drone'} (${confidence}% confidence)
                    </div>
                </div>
                <div style="color: ${result.is_drone ? '#e74c3c' : '#2ecc71'}; font-weight: bold;">
                    ${result.is_drone ? 'DRONE' : 'CLEAR'}
                </div>
            `;
        } else {
            item.innerHTML = `
                <div>
                    <strong>${result.filename}</strong>
                    <div style="font-size: 0.9em; color: #e74c3c;">
                        Error: ${result.error}
                    </div>
                </div>
                <div style="color: #e74c3c;">ERROR</div>
            `;
        }
        
        batchResults.appendChild(item);
    });
}

function displayError(message) {
    resultIcon.textContent = '❌';
    resultText.textContent = 'ERROR';
    resultCard.style.borderLeftColor = '#e74c3c';
    resultCard.style.background = '#fadbd8';
    
    confidenceFill.style.width = '0%';
    confidenceValue.textContent = '0%';
    probNonDrone.textContent = '0%';
    probDrone.textContent = '0%';
    audioDuration.textContent = '-';
    sampleRate.textContent = '-';
    
    // Add error message
    const errorDiv = document.createElement('div');
    errorDiv.style.color = '#e74c3c';
    errorDiv.style.marginTop = '10px';
    errorDiv.textContent = `Error: ${message}`;
    resultCard.appendChild(errorDiv);
}

// Check API health on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        console.log('API Health:', data);
    } catch (error) {
        console.error('API health check failed:', error);
    }
});

// Threshold management
let currentThreshold = 0.70;

function updateThresholdDisplay() {
    document.getElementById('thresholdValue').textContent = currentThreshold.toFixed(2);
    document.getElementById('thresholdSlider').value = currentThreshold;
    
    // Update active preset button
    document.querySelectorAll('.btn-preset').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Find matching preset and activate it
    const presets = [0.50, 0.70, 0.85];
    const presetIndex = presets.findIndex(preset => Math.abs(preset - currentThreshold) < 0.01);
    if (presetIndex !== -1) {
        document.querySelectorAll('.btn-preset')[presetIndex].classList.add('active');
    }
}

function setThreshold(value) {
    currentThreshold = value;
    updateThresholdDisplay();
    console.log(`Threshold set to: ${currentThreshold}`);
}

// Initialize threshold slider
document.getElementById('thresholdSlider').addEventListener('input', function(e) {
    currentThreshold = parseFloat(e.target.value);
    updateThresholdDisplay();
});

// Update the analyzeAudio function to include threshold
async function analyzeAudio() {
    const file = audioFileInput.files[0];
    if (!file) return;

    // Show loading state
    detectBtn.innerHTML = '<div class="loading"></div>Analyzing...';
    detectBtn.disabled = true;
    resultsSection.style.display = 'block';
    resultCard.style.background = '#f8f9fa';

    try {
        const formData = new FormData();
        formData.append('audio', file);

        // Include threshold in the request
        const response = await fetch(`/api/detect?threshold=${currentThreshold}`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.status === 'success') {
            displayResults(data);
        } else {
            throw new Error(data.error || 'Detection failed');
        }

    } catch (error) {
        displayError(error.message);
    } finally {
        detectBtn.innerHTML = 'Detect Drone';
        detectBtn.disabled = false;
    }
}

// Update the analyzeBatch function to include threshold
async function analyzeBatch() {
    const files = batchFilesInput.files;
    if (files.length === 0) return;

    batchDetectBtn.innerHTML = '<div class="loading"></div>Processing...';
    batchDetectBtn.disabled = true;
    batchResults.innerHTML = '';

    try {
        const formData = new FormData();
        for (let file of files) {
            formData.append('audio_files', file);
        }

        // Include threshold in batch request
        const response = await fetch(`/api/batch-detect?threshold=${currentThreshold}`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.status === 'success') {
            displayBatchResults(data.results);
        } else {
            throw new Error(data.error || 'Batch processing failed');
        }

    } catch (error) {
        batchResults.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    } finally {
        batchDetectBtn.innerHTML = 'Process Batch';
        batchDetectBtn.disabled = false;
    }
}

// Update displayResults to show threshold info
function displayResults(data) {
    const isDrone = data.is_drone;
    const confidence = data.confidence;
    const thresholdUsed = data.threshold_used || currentThreshold;
    
    // Update result display
    if (isDrone) {
        resultIcon.textContent = '🚁';
        resultText.textContent = 'DRONE DETECTED';
        resultCard.style.borderLeftColor = '#e74c3c';
        resultCard.style.background = '#ffeaa7';
    } else {
        resultIcon.textContent = '✅';
        resultText.textContent = 'NO DRONE DETECTED';
        resultCard.style.borderLeftColor = '#2ecc71';
        resultCard.style.background = '#d5f4e6';
    }

    // Update confidence meter
    const confidencePercent = Math.round(confidence * 100);
    confidenceFill.style.width = `${confidencePercent}%`;
    confidenceValue.textContent = `${confidencePercent}%`;

    // Update probabilities
    probNonDrone.textContent = `${Math.round(data.class_probabilities.non_drone * 100)}%`;
    probDrone.textContent = `${Math.round(data.class_probabilities.drone * 100)}%`;

    // Update audio info
    audioDuration.textContent = `${data.audio_duration.toFixed(2)}s`;
    sampleRate.textContent = `${data.sample_rate} Hz`;
    
    // Add threshold info
    let thresholdInfo = document.getElementById('thresholdInfo');
    if (!thresholdInfo) {
        thresholdInfo = document.createElement('div');
        thresholdInfo.id = 'thresholdInfo';
        thresholdInfo.className = 'threshold-info';
        resultCard.appendChild(thresholdInfo);
    }
    thresholdInfo.innerHTML = `Threshold: ${thresholdUsed.toFixed(2)} | Drone probability: ${(data.class_probabilities.drone * 100).toFixed(1)}%`;
}

// Initialize threshold on page load
document.addEventListener('DOMContentLoaded', function() {
    updateThresholdDisplay();
});

// Enhanced visualization functions
function createLocalizationPlot(visualizationData) {
    console.log('Creating localization plot with data:', visualizationData);
    
    const plotDiv = document.getElementById('localization-plot');
    if (!plotDiv) {
        console.error('Plot div not found');
        return;
    }

    try {
        // Clear any existing plot
        plotDiv.innerHTML = '';

        const traces = [];

        // 1. Add microphone positions
        if (visualizationData.microphones && visualizationData.microphones.length > 0) {
            visualizationData.microphones.forEach((mic, i) => {
                const trace = {
                    x: [mic.position[0]],
                    y: [mic.position[1]],
                    mode: 'markers+text',
                    type: 'scatter',
                    marker: {
                        size: 20,
                        color: mic.color,
                        symbol: 'circle',
                        line: { width: 2, color: 'white' }
                    },
                    text: [mic.label],
                    textposition: 'top center',
                    name: mic.label,
                    hoverinfo: 'x+y+text',
                    showlegend: false
                };
                traces.push(trace);
            });
        }

        // 2. Add estimated position
        if (visualizationData.estimated_position && visualizationData.estimated_position.position) {
            const estPos = visualizationData.estimated_position.position;
            const estTrace = {
                x: [estPos[0]],
                y: [estPos[1]],
                mode: 'markers+text',
                type: 'scatter',
                marker: {
                    size: 25,
                    color: visualizationData.estimated_position.color || 'red',
                    symbol: 'x',
                    line: { width: 3, color: 'black' }
                },
                text: [`Estimated<br>(${estPos[0].toFixed(2)}, ${estPos[1].toFixed(2)})`],
                textposition: 'bottom center',
                name: 'Estimated Position',
                hoverinfo: 'x+y+text'
            };
            traces.push(estTrace);

            // 3. Add confidence circle
            if (visualizationData.estimated_position.confidence) {
                const confidence = visualizationData.estimated_position.confidence;
                const radius = 0.1 + (confidence * 0.3);
                const center = estPos;
                
                const theta = Array.from({length: 100}, (_, i) => i * 2 * Math.PI / 100);
                const x = theta.map(t => center[0] + radius * Math.cos(t));
                const y = theta.map(t => center[1] + radius * Math.sin(t));
                
                const confidenceTrace = {
                    x: x,
                    y: y,
                    mode: 'lines',
                    type: 'scatter',
                    line: { 
                        dash: 'dash', 
                        color: visualizationData.estimated_position.color || 'red', 
                        width: 2 
                    },
                    fill: 'none',
                    name: `Confidence: ${(confidence * 100).toFixed(1)}%`,
                    hoverinfo: 'name',
                    showlegend: true
                };
                traces.push(confidenceTrace);
            }
        }

        // 4. Add true position if available
        if (visualizationData.true_position && visualizationData.true_position.position) {
            const truePos = visualizationData.true_position.position;
            const trueTrace = {
                x: [truePos[0]],
                y: [truePos[1]],
                mode: 'markers+text',
                type: 'scatter',
                marker: {
                    size: 25,
                    color: visualizationData.true_position.color || 'green',
                    symbol: 'star'
                },
                text: [`True Position<br>(${truePos[0].toFixed(2)}, ${truePos[1].toFixed(2)})`],
                textposition: 'top center',
                name: 'True Position',
                hoverinfo: 'x+y+text'
            };
            traces.push(trueTrace);

            // 5. Add error line if both positions exist
            if (visualizationData.estimated_position && visualizationData.estimated_position.position) {
                const errorTrace = {
                    x: [visualizationData.estimated_position.position[0], truePos[0]],
                    y: [visualizationData.estimated_position.position[1], truePos[1]],
                    mode: 'lines',
                    type: 'scatter',
                    line: {
                        color: 'orange',
                        width: 2,
                        dash: 'dot'
                    },
                    name: `Error: ${visualizationData.error ? visualizationData.error.toFixed(3) + 'm' : 'N/A'}`,
                    hoverinfo: 'name',
                    showlegend: true
                };
                traces.push(errorTrace);
            }
        }

        const layout = {
            title: {
                text: 'Drone Localization Map',
                font: { size: 16, weight: 'bold' }
            },
            xaxis: {
                title: 'X Position (meters)',
                range: [-2, 3],
                gridcolor: 'lightgray',
                zeroline: true,
                zerolinecolor: 'black',
                zerolinewidth: 2
            },
            yaxis: {
                title: 'Y Position (meters)',
                range: [-1, 2],
                gridcolor: 'lightgray',
                zeroline: true,
                zerolinecolor: 'black',
                zerolinewidth: 2
            },
            showlegend: true,
            legend: { 
                x: 0, 
                y: 1.1,
                orientation: 'h'
            },
            width: plotDiv.clientWidth,
            height: 500,
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            margin: { l: 60, r: 40, t: 60, b: 60 }
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
        };

        Plotly.newPlot(plotDiv, traces, layout, config)
            .then(() => {
                console.log('Plot created successfully');
            })
            .catch(error => {
                console.error('Plotly error:', error);
                plotDiv.innerHTML = `
                    <div style="text-align: center; padding: 2rem; color: #666;">
                        <h4>Visualization Error</h4>
                        <p>Could not create localization plot.</p>
                        <p><small>Error: ${error.message}</small></p>
                    </div>
                `;
            });

    } catch (error) {
        console.error('Error creating plot:', error);
        plotDiv.innerHTML = `
            <div style="text-align: center; padding: 2rem; color: #666;">
                <h4>Visualization Error</h4>
                <p>${error.message}</p>
            </div>
        `;
    }
}

// Enhanced analyzeAudio function with better logging
async function analyzeAudio() {
    const fileInput = document.getElementById('audioFile');
    const threshold = document.getElementById('thresholdSlider').value;
    
    if (!fileInput.files[0]) {
        alert('Please select an audio file first');
        return;
    }

    const detectBtn = document.getElementById('detectBtn');
    const originalText = detectBtn.textContent;
    
    detectBtn.disabled = true;
    detectBtn.textContent = 'Processing...';

    // Hide previous results
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('visualizationSection').style.display = 'none';

    const formData = new FormData();
    formData.append('audio', fileInput.files[0]);
    formData.append('threshold', threshold);

    try {
        console.log('Sending request to server...');
        const response = await fetch('/api/detect-with-localization', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const result = await response.json();
        console.log('Server response:', result);
        
        if (result.status === 'success') {
            displayResults(result);
            
            // Show localization if available
            if (result.localization && result.localization.visualization_data) {
                console.log('Localization data available:', result.localization);
                
                document.getElementById('localizationInfo').style.display = 'block';
                document.getElementById('positionInfo').textContent = 
                    `(${result.localization.estimated_position[0].toFixed(2)}, ${result.localization.estimated_position[1].toFixed(2)}) m`;
                
                if (result.localization.error) {
                    document.getElementById('errorInfo').textContent = 
                        `${result.localization.error.toFixed(3)} m`;
                } else {
                    document.getElementById('errorInfo').textContent = 'N/A (no true position provided)';
                }
                
                // Show and create visualization
                document.getElementById('visualizationSection').style.display = 'block';
                setTimeout(() => {
                    createLocalizationPlot(result.localization.visualization_data);
                }, 100);
            } else {
                console.log('No localization data available');
                document.getElementById('visualizationSection').style.display = 'none';
                document.getElementById('localizationInfo').style.display = 'none';
            }
        } else {
            throw new Error(result.error || 'Detection failed');
        }
    } catch (error) {
        console.error('Detection error:', error);
        alert('Detection failed: ' + error.message);
    } finally {
        detectBtn.disabled = false;
        detectBtn.textContent = originalText;
    }
}

// Enhanced displayResults function
function displayResults(result) {
    const resultsSection = document.getElementById('resultsSection');
    const resultCard = document.getElementById('resultCard');
    const resultIcon = document.getElementById('resultIcon');
    const resultText = document.getElementById('resultText');
    
    const detection = result.detection;
    const isDrone = detection.is_drone;
    const confidence = detection.confidence;
    
    // Update detection results
    if (isDrone) {
        resultIcon.textContent = '🚁';
        resultText.textContent = 'DRONE DETECTED';
        resultCard.style.borderLeftColor = '#e74c3c';
        resultCard.style.backgroundColor = '#fff5f5';
    } else {
        resultIcon.textContent = '🌳';
        resultText.textContent = 'NO DRONE DETECTED';
        resultCard.style.borderLeftColor = '#27ae60';
        resultCard.style.backgroundColor = '#f5fff5';
    }
    
    // Update confidence meter
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceValue = document.getElementById('confidenceValue');
    
    const confidencePercent = confidence * 100;
    confidenceFill.style.width = `${confidencePercent}%`;
    confidenceValue.textContent = `${confidencePercent.toFixed(1)}%`;
    
    // Color confidence bar based on detection
    if (isDrone) {
        confidenceFill.style.backgroundColor = confidencePercent > 70 ? '#e74c3c' : '#f39c12';
    } else {
        confidenceFill.style.backgroundColor = '#27ae60';
    }
    
    // Update probabilities
    document.getElementById('probNonDrone').textContent = 
        `${(detection.class_probabilities.non_drone * 100).toFixed(1)}%`;
    document.getElementById('probDrone').textContent = 
        `${(detection.class_probabilities.drone * 100).toFixed(1)}%`;
    
    // Update audio info
    document.getElementById('audioDuration').textContent = 
        `${result.audio_info.duration.toFixed(2)}s`;
    document.getElementById('sampleRate').textContent = 
        `${result.audio_info.sample_rate} Hz`;
    document.getElementById('channelsInfo').textContent = 
        `${result.audio_info.channels} channel${result.audio_info.channels !== 1 ? 's' : ''}`;
    
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Test function to verify Plotly is working
function testPlotly() {
    console.log('Testing Plotly...');
    if (typeof Plotly === 'undefined') {
        console.error('Plotly not loaded');
        return false;
    }
    
    const testDiv = document.createElement('div');
    testDiv.style.position = 'absolute';
    testDiv.style.left = '-9999px';
    document.body.appendChild(testDiv);
    
    try {
        Plotly.newPlot(testDiv, [{
            x: [1, 2, 3],
            y: [2, 1, 3],
            type: 'scatter'
        }], {
            title: 'Test Plot',
            width: 400,
            height: 300
        });
        
        document.body.removeChild(testDiv);
        console.log('Plotly test successful');
        return true;
    } catch (error) {
        console.error('Plotly test failed:', error);
        document.body.removeChild(testDiv);
        return false;
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing drone detection system...');
    
    // Test Plotly
    setTimeout(() => {
        if (!testPlotly()) {
            console.warn('Plotly not functioning properly');
        }
    }, 1000);
    
    // Add channels info to HTML if not present
    if (!document.getElementById('channelsInfo')) {
        const audioInfo = document.querySelector('.audio-info');
        if (audioInfo) {
            const channelsItem = document.createElement('div');
            channelsItem.className = 'info-item';
            channelsItem.innerHTML = '<span>Channels:</span><span id="channelsInfo">-</span>';
            audioInfo.appendChild(channelsItem);
        }
    }
});

// Add this function to test visualization without needing specific audio files
function testLocalization() {
    console.log('Testing localization visualization...');
    
    // Create synthetic test data
    const testData = {
        detection: {
            is_drone: true,
            confidence: 0.85,
            class_probabilities: {
                non_drone: 0.15,
                drone: 0.85
            }
        },
        localization: {
            estimated_position: [1.2, 0.8],
            tdoas: [0.001, 0.002],
            visualization_data: {
                microphones: [
                    { position: [0, 0], label: 'Mic 1 (Ref)', color: 'red' },
                    { position: [0.5, 0], label: 'Mic 2', color: 'blue' },
                    { position: [0, 0.5], label: 'Mic 3', color: 'green' }
                ],
                estimated_position: {
                    position: [1.2, 0.8],
                    confidence: 0.85,
                    color: 'red'
                },
                true_position: {
                    position: [1.0, 0.9],
                    color: 'green'
                },
                error: 0.223
            }
        },
        audio_info: {
            duration: 5.0,
            sample_rate: 44100,
            channels: 3,
            samples: 220500
        },
        debug: {
            channels_available: 3,
            drone_detected: true,
            localization_possible: true,
            localization_reason: "Test data",
            localization_success: true
        },
        status: 'success'
    };
    
    // Display the test results
    displayResults(testData);
    document.getElementById('debugSection').style.display = 'block';
    document.getElementById('debugInfo').textContent = JSON.stringify(testData.debug, null, 2);
    
    document.getElementById('localizationInfo').style.display = 'block';
    document.getElementById('positionInfo').textContent = 
        `(${testData.localization.estimated_position[0].toFixed(2)}, ${testData.localization.estimated_position[1].toFixed(2)}) m`;
    document.getElementById('errorInfo').textContent = 
        `${testData.localization.visualization_data.error.toFixed(3)} m`;
    
    document.getElementById('visualizationSection').style.display = 'block';
    createLocalizationPlot(testData.localization.visualization_data);
    
    console.log('Test localization completed');
}

// Add a test button to your HTML temporarily for debugging
function addTestButton() {
    const testBtn = document.createElement('button');
    testBtn.textContent = 'Test Visualization';
    testBtn.className = 'btn-secondary';
    testBtn.style.marginLeft = '10px';
    testBtn.onclick = testLocalization;
    
    const controls = document.querySelector('.controls');
    controls.appendChild(testBtn);
}

// Call this in your DOMContentLoaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing drone detection system...');
    
    // Test Plotly
    setTimeout(() => {
        if (!testPlotly()) {
            console.warn('Plotly not functioning properly');
        }
    }, 1000);
    
    // Add test button for debugging
    addTestButton();
    
    // Add channels info to HTML if not present
    if (!document.getElementById('channelsInfo')) {
        const audioInfo = document.querySelector('.audio-info');
        if (audioInfo) {
            const channelsItem = document.createElement('div');
            channelsItem.className = 'info-item';
            channelsItem.innerHTML = '<span>Channels:</span><span id="channelsInfo">-</span>';
            audioInfo.appendChild(channelsItem);
        }
    }
});