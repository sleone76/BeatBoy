document.addEventListener('DOMContentLoaded', () => {
    const generateBtn = document.getElementById('generateBtn');
    const bpmInput = document.getElementById('bpm');
    const bpmSlider = document.getElementById('bpmSlider');
    const audioPlayer = document.getElementById('audioPlayer');
    const downloadLink = document.getElementById('downloadLink');
    const instrumentsContainer = document.querySelector('.instrument-section');
    const pianoContainer = document.getElementById('piano');

    // Sync BPM Inputs
    bpmInput.addEventListener('input', () => bpmSlider.value = bpmInput.value);
    bpmSlider.addEventListener('input', () => bpmInput.value = bpmSlider.value);

    // State
    let isGenerating = false;
    let noteEvents = [];
    let isPlaying = false;

    // VISUALS STATE
    let audioCtx, analyser, dataArray, source;
    let visualsInitialized = false;

    // --- BUILD PIANO ---
    const startNote = 36; // C2
    const endNote = 84;   // C6
    const keysMap = {};

    function isBlackKey(midi) {
        return [1, 3, 6, 8, 10].includes(midi % 12);
    }

    for (let i = startNote; i <= endNote; i++) {
        const isBlack = isBlackKey(i);
        const key = document.createElement('div');
        key.classList.add('key');
        if (isBlack) key.classList.add('black');
        else key.classList.add('white');

        key.dataset.note = i;
        keysMap[i] = key;
        pianoContainer.appendChild(key);
    }

    // --- AUDIO VISUALizer SETUP ---
    function initVisuals() {
        if (visualsInitialized) return;

        try {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioCtx.createAnalyser();
            analyser.fftSize = 256; // We only need bass presence, low res is fine

            source = audioCtx.createMediaElementSource(audioPlayer);
            source.connect(analyser);
            analyser.connect(audioCtx.destination);

            dataArray = new Uint8Array(analyser.frequencyBinCount);
            visualsInitialized = true;
        } catch (e) {
            console.warn("Web Audio API not supported still?", e);
        }
    }

    // --- GENERATE ---
    generateBtn.addEventListener('click', async () => {
        if (isGenerating) return;

        // Init audio context on user gesture
        initVisuals();
        if (audioCtx && audioCtx.state === 'suspended') {
            audioCtx.resume();
        }

        isGenerating = true;
        generateBtn.classList.add('loading');

        const checked = Array.from(instrumentsContainer.querySelectorAll('input:checked'))
            .map(cb => cb.value);
        const bpm = parseInt(bpmInput.value);

        try {
            // Use relative path for production compatibility
            const response = await fetch('/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ bpm, instruments: checked })
            });

            if (!response.ok) throw new Error('Generation failed');

            const data = await response.json();

            const audioSrc = `data:audio/wav;base64,${data.audio}`;
            audioPlayer.src = audioSrc;
            downloadLink.href = audioSrc;
            noteEvents = data.events;

            audioPlayer.play()
                .then(() => isPlaying = true)
                .catch(e => console.log("Auto-play blocked"));

        } catch (err) {
            alert('Error: ' + err.message);
        } finally {
            isGenerating = false;
            generateBtn.classList.remove('loading');
        }
    });

    audioPlayer.addEventListener('play', () => isPlaying = true);
    audioPlayer.addEventListener('pause', () => isPlaying = false);
    audioPlayer.addEventListener('ended', () => { isPlaying = false; clearPiano(); });

    function clearPiano() {
        Object.values(keysMap).forEach(k => k.classList.remove('active'));
    }

    function renderFrame() {
        requestAnimationFrame(renderFrame);

        if (!isPlaying) {
            // Slowly decay background if paused
            document.body.style.setProperty('--bass-level', '0');
            return;
        }

        // 1. Piano Logic (Time based)
        const ct = audioPlayer.currentTime;
        clearPiano();
        for (const ev of noteEvents) {
            if (ct >= ev.t && ct < (ev.t + ev.len)) {
                const el = keysMap[ev.note];
                if (el) el.classList.add('active');
            }
        }

        // 2. Background Logic (Frequency based)
        if (visualsInitialized) {
            analyser.getByteFrequencyData(dataArray);

            // Average the first few bins (Bass)
            // Bin size = 44100 / 256 ~= 172Hz per bin? No. sr/fftSize. 
            // 256 FFT size -> 128 bins. 0-22kHz.
            // Bins 0-5 are deep bass.

            let bassSum = 0;
            for (let i = 0; i < 5; i++) {
                bassSum += dataArray[i];
            }
            const bassAvg = bassSum / 5;

            // Normalize 0-255 -> 0.0-1.0
            const val = bassAvg / 255.0;

            // Smooth it? CSS transition handles smoothing usually, but setting var per frame is noisy.
            // Direct mapping is punchier.
            document.body.style.setProperty('--bass-level', val.toFixed(3));
        }
    }

    renderFrame();
});
