let ws;
let video = document.getElementById('video');
let canvas = document.getElementById('output-canvas');
let ctx = canvas.getContext('2d');
let modeSelect = document.getElementById('mode-select');
let qualitySlider = document.getElementById('quality-slider');
let landmarksCheckbox = document.getElementById('landmarks-checkbox');
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
            quality: parseInt(qualitySlider.value),
            show_landmarks: landmarksCheckbox.checked
        }));
    }
}

function updateLabels(labels, images) {
    const container = document.getElementById('labels-container');
    container.innerHTML = '';
    labels.forEach((label, index) => {
        const div = document.createElement('div');
        div.className = 'glass-panel';
        div.style.padding = '20px 30px';
        div.style.display = 'flex';
        div.style.flexDirection = 'column';
        div.style.alignItems = 'center';
        div.style.minWidth = '180px';
        div.style.animation = 'scaleIn 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards';
        div.style.borderTop = '4px solid var(--secondary)';

        const textDiv = document.createElement('div');
        textDiv.innerText = label;
        textDiv.style.fontSize = '1.1rem';
        textDiv.style.fontWeight = '700';
        textDiv.style.textTransform = 'uppercase';
        textDiv.style.letterSpacing = '1px';
        div.appendChild(textDiv);

        if (images && images[index]) {
            const imgContainer = document.createElement('div');
            imgContainer.style.marginTop = '15px';
            imgContainer.style.background = 'rgba(255,255,255,0.05)';
            imgContainer.style.padding = '10px';
            imgContainer.style.borderRadius = '12px';
            imgContainer.style.display = 'flex';
            imgContainer.style.justifyContent = 'center';

            const img = document.createElement('img');
            img.src = `/images/${images[index]}`;
            img.style.width = '80px';
            img.style.height = '80px';
            img.style.objectFit = 'contain';
            img.style.borderRadius = '8px';
            imgContainer.appendChild(img);
            div.appendChild(imgContainer);
        }

        container.appendChild(div);
    });
}

// Add animation to the CSS
const style = document.createElement('style');
style.innerHTML = `
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
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
                    updateLabels(msg.data, msg.images);
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

if (qualitySlider) qualitySlider.addEventListener('input', updateState);
if (landmarksCheckbox) landmarksCheckbox.addEventListener('change', updateState);

setupWebcam().then(connectWS);
