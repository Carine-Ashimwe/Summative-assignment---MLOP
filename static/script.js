// Global variables
let predictionFile = null;
let batchFiles = [];
let retrainFiles = [];
let uptimeInterval = null;
let retrainStatusInterval = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    startUptimeMonitoring();
    initializeCharts();
});

// Initialize event listeners
function initializeEventListeners() {
    // Prediction file upload
    const predictionUpload = document.getElementById('prediction-upload');
    const predictionFileInput = document.getElementById('prediction-file');
    const predictBtn = document.getElementById('predict-btn');
    
    predictionUpload.addEventListener('click', () => predictionFileInput.click());
    predictionUpload.addEventListener('dragover', handleDragOver);
    predictionUpload.addEventListener('drop', handlePredictionDrop);
    predictionFileInput.addEventListener('change', handlePredictionFileSelect);
    predictBtn.addEventListener('click', handlePredict);
    
    // Batch prediction
    const batchUpload = document.getElementById('batch-upload');
    const batchFilesInput = document.getElementById('batch-files');
    const batchPredictBtn = document.getElementById('batch-predict-btn');
    
    batchUpload.addEventListener('click', () => batchFilesInput.click());
    batchFilesInput.addEventListener('change', handleBatchFilesSelect);
    batchPredictBtn.addEventListener('click', handleBatchPredict);
    
    // Retrain data upload
    const retrainUpload = document.getElementById('retrain-upload');
    const retrainFilesInput = document.getElementById('retrain-files');
    const uploadRetrainBtn = document.getElementById('upload-retrain-btn');
    
    retrainUpload.addEventListener('click', () => retrainFilesInput.click());
    retrainFilesInput.addEventListener('change', handleRetrainFilesSelect);
    uploadRetrainBtn.addEventListener('click', handleUploadRetrainData);
    
    // Retraining trigger
    const retrainBtn = document.getElementById('retrain-btn');
    retrainBtn.addEventListener('click', handleTriggerRetrain);
}

// Drag and drop handlers
function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handlePredictionDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        handleFileSelection(files[0], 'prediction');
    }
}

// File selection handlers
function handlePredictionFileSelect(e) {
    if (e.target.files.length > 0) {
        handleFileSelection(e.target.files[0], 'prediction');
    }
}

function handleFileSelection(file, type) {
    if (type === 'prediction') {
        predictionFile = file;
        displayImagePreview(file);
        document.getElementById('predict-btn').disabled = false;
    }
}

function displayImagePreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const preview = document.getElementById('prediction-preview');
        const img = document.getElementById('preview-img');
        img.src = e.target.result;
        preview.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function handleBatchFilesSelect(e) {
    batchFiles = Array.from(e.target.files);
    document.getElementById('batch-predict-btn').disabled = batchFiles.length === 0;
    if (batchFiles.length > 0) {
        showMessage('batch-results', `Selected ${batchFiles.length} files`, 'info');
    }
}

function handleRetrainFilesSelect(e) {
    retrainFiles = Array.from(e.target.files);
    document.getElementById('upload-retrain-btn').disabled = retrainFiles.length === 0;
}

// Prediction handlers
async function handlePredict() {
    if (!predictionFile) return;
    
    const btn = document.getElementById('predict-btn');
    btn.disabled = true;
    btn.textContent = 'Predicting...';
    
    const formData = new FormData();
    formData.append('image', predictionFile);
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayPredictionResult(result);
        } else {
            showError('prediction-result', result.error || 'Prediction failed');
        }
    } catch (error) {
        showError('prediction-result', 'Error: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Predict';
    }
}

function displayPredictionResult(result) {
    const resultDiv = document.getElementById('prediction-result');
    resultDiv.style.display = 'block';
    resultDiv.className = 'result-box success';
    
    let html = `<h3>Prediction Result</h3>`;
    html += `<p><strong>Predicted Class:</strong> ${result.predicted_class}</p>`;
    html += `<p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%</p>`;
    html += `<p><strong>Prediction Time:</strong> ${result.prediction_time_ms}ms</p>`;
    
    if (result.probabilities) {
        html += `<h4>All Probabilities:</h4><ul>`;
        const sorted = Object.entries(result.probabilities)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5);
        sorted.forEach(([cls, prob]) => {
            html += `<li>${cls}: ${(prob * 100).toFixed(2)}%</li>`;
        });
        html += `</ul>`;
    }
    
    resultDiv.innerHTML = html;
}

async function handleBatchPredict() {
    if (batchFiles.length === 0) return;
    
    const btn = document.getElementById('batch-predict-btn');
    btn.disabled = true;
    btn.textContent = 'Predicting...';
    
    const formData = new FormData();
    batchFiles.forEach(file => formData.append('images', file));
    
    try {
        const response = await fetch('/api/predict/batch', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayBatchResults(result);
        } else {
            showError('batch-results', result.error || 'Batch prediction failed');
        }
    } catch (error) {
        showError('batch-results', 'Error: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Predict All';
    }
}

function displayBatchResults(result) {
    const resultsDiv = document.getElementById('batch-results');
    resultsDiv.style.display = 'block';
    
    let html = `<h3>Batch Prediction Results</h3>`;
    html += `<p>Processed ${result.total_images} images in ${result.total_time_ms}ms (avg: ${result.avg_time_per_image_ms}ms per image)</p>`;
    html += `<div class="results-container">`;
    
    result.results.forEach((r, idx) => {
        if (r.error) {
            html += `<div class="result-item error">${r.image_path}: ${r.error}</div>`;
        } else {
            const reader = new FileReader();
            reader.onload = function(e) {
                // Image preview would go here
            };
            reader.readAsDataURL(batchFiles[idx]);
            
            html += `<div class="result-item">`;
            html += `<strong>${r.predicted_class}</strong><br>`;
            html += `Confidence: ${(r.confidence * 100).toFixed(2)}%`;
            html += `</div>`;
        }
    });
    
    html += `</div>`;
    resultsDiv.innerHTML = html;
}

// Retrain data upload
async function handleUploadRetrainData() {
    if (retrainFiles.length === 0) return;
    
    const btn = document.getElementById('upload-retrain-btn');
    btn.disabled = true;
    btn.textContent = 'Uploading...';
    
    const statusDiv = document.getElementById('upload-status');
    const progressBar = document.getElementById('upload-progress');
    progressBar.style.display = 'block';
    
    const formData = new FormData();
    retrainFiles.forEach(file => formData.append('data', file));
    
    try {
        const response = await fetch('/api/upload/retrain', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            statusDiv.className = 'status-message success';
            statusDiv.textContent = result.message;
            progressBar.style.display = 'none';
        } else {
            statusDiv.className = 'status-message error';
            statusDiv.textContent = result.error || 'Upload failed';
        }
    } catch (error) {
        statusDiv.className = 'status-message error';
        statusDiv.textContent = 'Error: ' + error.message;
    } finally {
        btn.disabled = false;
        btn.textContent = 'Upload Training Data';
    }
}

// Retraining trigger
async function handleTriggerRetrain() {
    const btn = document.getElementById('retrain-btn');
    btn.disabled = true;
    btn.textContent = 'Starting...';
    
    const progressSection = document.getElementById('retrain-progress');
    progressSection.style.display = 'block';
    
    try {
        const response = await fetch('/api/retrain', {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (response.ok) {
            startRetrainStatusMonitoring();
        } else {
            showError('retrain-result', result.error || 'Failed to start retraining');
            btn.disabled = false;
            btn.textContent = 'Trigger Retraining';
            progressSection.style.display = 'none';
        }
    } catch (error) {
        showError('retrain-result', 'Error: ' + error.message);
        btn.disabled = false;
        btn.textContent = 'Trigger Retraining';
        progressSection.style.display = 'none';
    }
}

function startRetrainStatusMonitoring() {
    if (retrainStatusInterval) {
        clearInterval(retrainStatusInterval);
    }
    
    retrainStatusInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/retrain/status');
            const status = await response.json();
            
            updateRetrainProgress(status);
            
            if (status.status === 'completed' || status.status === 'failed') {
                clearInterval(retrainStatusInterval);
                document.getElementById('retrain-btn').disabled = false;
                document.getElementById('retrain-btn').textContent = 'Trigger Retraining';
                
                if (status.status === 'completed') {
                    // Reload model uptime after retraining
                    setTimeout(startUptimeMonitoring, 1000);
                }
            }
        } catch (error) {
            console.error('Error checking retrain status:', error);
        }
    }, 2000);
}

function updateRetrainProgress(status) {
    const progressFill = document.getElementById('retrain-progress-fill');
    const statusMessage = document.getElementById('retrain-status-message');
    const resultDiv = document.getElementById('retrain-result');
    
    progressFill.style.width = status.progress + '%';
    statusMessage.textContent = status.message;
    
    if (status.status === 'completed') {
        resultDiv.style.display = 'block';
        resultDiv.className = 'result-box success';
        resultDiv.innerHTML = `
            <h3>Retraining Completed!</h3>
            <p>${status.message}</p>
            ${status.accuracy ? `<p><strong>Test Accuracy:</strong> ${(status.accuracy * 100).toFixed(2)}%</p>` : ''}
            ${status.completed_at ? `<p><strong>Completed At:</strong> ${new Date(status.completed_at).toLocaleString()}</p>` : ''}
        `;
    } else if (status.status === 'failed') {
        resultDiv.style.display = 'block';
        resultDiv.className = 'result-box error';
        resultDiv.innerHTML = `
            <h3>Retraining Failed</h3>
            <p>${status.message}</p>
        `;
    }
}

// Uptime monitoring
function startUptimeMonitoring() {
    if (uptimeInterval) {
        clearInterval(uptimeInterval);
    }
    
    updateUptime();
    uptimeInterval = setInterval(updateUptime, 1000);
}

async function updateUptime() {
    try {
        const response = await fetch('/api/model/uptime');
        const data = await response.json();
        
        const statusDot = document.getElementById('status-dot');
        const statusText = document.getElementById('status-text');
        const uptimeValue = document.getElementById('uptime-value');
        const loadedAt = document.getElementById('loaded-at');
        
        if (data.status === 'loaded') {
            statusDot.className = 'dot loaded';
            statusText.textContent = 'Model Loaded';
            uptimeValue.textContent = data.uptime_formatted || formatUptime(data.uptime_seconds);
            loadedAt.textContent = new Date(data.model_loaded_at).toLocaleString();
        } else {
            statusDot.className = 'dot error';
            statusText.textContent = 'Model Not Loaded';
            uptimeValue.textContent = '--';
            loadedAt.textContent = '--';
        }
    } catch (error) {
        console.error('Error fetching uptime:', error);
    }
}

function formatUptime(seconds) {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    const parts = [];
    if (days > 0) parts.push(`${days}d`);
    if (hours > 0) parts.push(`${hours}h`);
    if (minutes > 0) parts.push(`${minutes}m`);
    parts.push(`${secs}s`);
    
    return parts.join(' ');
}

// Initialize charts
async function initializeCharts() {
    // Fetch real data from API if available
    let classData = null;
    try {
        const response = await fetch('/api/stats');
        if (response.ok) {
            const stats = await response.json();
            if (stats.class_distribution) {
                classData = stats.class_distribution;
            }
        }
    } catch (error) {
        console.log('Could not fetch stats, using sample data');
    }
    
    // Class Distribution Chart
    const classCtx = document.getElementById('class-distribution-chart');
    const classLabels = classData ? Object.keys(classData) : ['Daisy', 'Dandelion', 'Roses', 'Sunflowers', 'Tulips'];
    const classValues = classData ? Object.values(classData) : [700, 900, 700, 700, 700];
    
    new Chart(classCtx, {
        type: 'bar',
        data: {
            labels: classLabels,
            datasets: [{
                label: 'Number of Samples',
                data: classValues,
                backgroundColor: 'rgba(102, 126, 234, 0.6)',
                borderColor: 'rgba(102, 126, 234, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                title: {
                    display: true,
                    text: 'Training Data Distribution'
                }
            },
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
    
    // Performance Chart
    const perfCtx = document.getElementById('performance-chart');
    new Chart(perfCtx, {
        type: 'line',
        data: {
            labels: Array.from({length: 20}, (_, i) => `Epoch ${i+1}`),
            datasets: [{
                label: 'Training Accuracy',
                data: Array.from({length: 20}, () => Math.random() * 0.2 + 0.75),
                borderColor: 'rgba(102, 126, 234, 1)',
                tension: 0.4
            }, {
                label: 'Validation Accuracy',
                data: Array.from({length: 20}, () => Math.random() * 0.2 + 0.70),
                borderColor: 'rgba(118, 75, 162, 1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: false, min: 0.6, max: 1.0 }
            }
        }
    });
    
    // Confidence Distribution Chart
    const confCtx = document.getElementById('confidence-chart');
    new Chart(confCtx, {
        type: 'histogram',
        data: {
            datasets: [{
                label: 'Prediction Confidence',
                data: Array.from({length: 100}, () => Math.random() * 0.3 + 0.7),
                backgroundColor: 'rgba(102, 126, 234, 0.6)'
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: { 
                    type: 'linear',
                    title: { display: true, text: 'Confidence' },
                    min: 0.5,
                    max: 1.0
                },
                y: { 
                    title: { display: true, text: 'Frequency' },
                    beginAtZero: true
                }
            }
        }
    });
}

// Utility functions
function showError(elementId, message) {
    const element = document.getElementById(elementId);
    element.style.display = 'block';
    element.className = 'result-box error';
    element.innerHTML = `<p><strong>Error:</strong> ${message}</p>`;
}

function showMessage(elementId, message, type = 'info') {
    const element = document.getElementById(elementId);
    element.style.display = 'block';
    element.className = `status-message ${type}`;
    element.textContent = message;
}

