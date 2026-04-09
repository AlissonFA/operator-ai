let ws;
let video = document.getElementById('video');
let canvas = document.getElementById('output-canvas');
let ctx = canvas.getContext('2d');
let modeSelect = document.getElementById('mode-select');
let labelInput = document.getElementById('label-input');
let recordBtn = document.getElementById('record-btn');
let isRecording = false;
let lastFrameTime = 0;

async function setupWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        video.srcObject = stream;
        return new Promise((resolve) => video.onloadedmetadata = () => { video.play(); resolve(); });
    } catch (e) { console.error("Webcam error:", e); }
}

function updateState() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'config',
            mode: modeSelect.value,
            label: labelInput.value,
            isRecording: isRecording
        }));
    }
}

function toggleRecording() {
    isRecording = !isRecording;
    recordBtn.innerText = isRecording ? "Stop Recording" : "Start Recording";
    updateState();
}

function updateLabels(labels) {
    const container = document.getElementById('labels-container');
    container.innerHTML = '';
    labels.forEach(label => {
        const div = document.createElement('div');
        div.innerText = label;
        div.style.padding = '10px 20px';
        div.style.background = 'rgba(255, 255, 255, 0.1)';
        div.style.backdropFilter = 'blur(10px)';
        div.style.borderRadius = '8px';
        div.style.borderLeft = '4px solid #00ffcc';
        div.style.color = 'white';
        div.style.fontWeight = 'bold';
        div.style.fontSize = '0.9rem';
        div.style.boxShadow = '0 4px 15px rgba(0,0,0,0.2)';
        div.style.animation = 'fadeIn 0.3s ease-out';
        container.appendChild(div);
    });
}

// Add animation to the CSS
const style = document.createElement('style');
style.innerHTML = `
    @keyframes fadeIn {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
`;
document.head.appendChild(style);

function connectWS() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        console.log('Connected to WS');
        updateState();
        requestAnimationFrame(sendFrame);
    };

    ws.onmessage = async (event) => {
        if (typeof event.data === 'string') {
            try {
                const msg = JSON.parse(event.data);
                if (msg.type === 'labels') {
                    updateLabels(msg.data);
                }
            } catch (e) { console.error("JSON error:", e); }
        } else if (event.data instanceof ArrayBuffer) {
            const blob = new Blob([event.data], { type: 'image/jpeg' });
            const url = URL.createObjectURL(blob);
            const img = new Image();
            img.onload = () => {
                ctx.drawImage(img, 0, 0);
                URL.revokeObjectURL(url);
                requestAnimationFrame(sendFrame);
            };
            img.src = url;
        }
    };
    
    ws.onclose = () => setTimeout(connectWS, 1000);
}

const tempCanvas = document.createElement('canvas');
const tempCtx = tempCanvas.getContext('2d');
tempCanvas.width = 640;
tempCanvas.height = 480;

function sendFrame(time) {
    if (time - lastFrameTime < 33) { // Cap at ~30fps
        requestAnimationFrame(sendFrame);
        return;
    }
    lastFrameTime = time;

    if (ws && ws.readyState === WebSocket.OPEN && video.readyState >= 2) {
        tempCtx.drawImage(video, 0, 0, 640, 480);
        tempCanvas.toBlob((blob) => {
            if (blob) ws.send(blob);
        }, 'image/jpeg', 0.6);
    } else {
        requestAnimationFrame(sendFrame);
    }
}

setupWebcam().then(connectWS);
