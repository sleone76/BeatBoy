import numpy as np
import random
import io
import base64
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter, fftconvolve

# --- DSP CORE --- 
# (Unchanged)

def get_sine(freq, t):
    return np.sin(2 * np.pi * freq * t)

def get_square(freq, t):
    return np.sign(np.sin(2 * np.pi * freq * t))

def get_saw(freq, t):
    return 2 * (t * freq - np.floor(t * freq + 0.5))

def butter_lowpass(cutoff, fs, order=4):
    if cutoff <= 0: cutoff = 1
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0: normal_cutoff = 0.99
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return lfilter(b, a, data)

def apply_chorus(audio, sr, depth=0.002, rate=1.5, mix=0.5):
    n = len(audio)
    t = np.linspace(0, n/sr, n)
    mod = depth * np.sin(2 * np.pi * rate * t)
    avg_delay_samps = int(0.02 * sr) 
    mod_samps = (mod * sr).astype(int)
    indices = np.arange(n) - avg_delay_samps - mod_samps
    indices = np.clip(indices, 0, n-1)
    wet = audio[indices]
    return audio * (1-mix) + wet * mix

def apply_reverb(audio, sr, room_size=0.6):
    duration = 1.0 * room_size + 0.1
    n_ir = int(sr * duration)
    t = np.linspace(0, duration, n_ir)
    noise = np.random.uniform(-1, 1, n_ir)
    decay = np.exp(-t * (5.0 / (room_size + 0.1))) 
    ir = noise * decay
    ir = apply_filter(ir, 3000, sr, order=2)
    wet = fftconvolve(audio, ir, mode='full')
    wet = wet / (np.max(np.abs(wet)) + 1e-9)
    dry_gain = 0.8
    wet_gain = 0.25
    pad_len = len(wet) - len(audio)
    if pad_len > 0: audio_padded = np.pad(audio, (0, pad_len))
    else: audio_padded = audio
    return audio_padded * dry_gain + wet * wet_gain

def adsr_envelope(n_samples, attack, decay, sustain_level, release, note_len_samples):
    a_samp = int(attack * n_samples)
    d_samp = int(decay * n_samples)
    r_samp = int(release * n_samples)
    
    if a_samp + d_samp + r_samp > n_samples:
        factor = n_samples / (a_samp + d_samp + r_samp + 1)
        a_samp = int(a_samp * factor)
        d_samp = int(d_samp * factor)
        r_samp = int(r_samp * factor)

    total_len = a_samp + d_samp + r_samp
    s_samp = n_samples - total_len
    if s_samp < 0: s_samp = 0

    env = np.zeros(n_samples)
    current = 0
    if a_samp > 0:
        env[:a_samp] = np.linspace(0, 1, a_samp)
        current += a_samp
    if d_samp > 0:
        env[current:current+d_samp] = np.linspace(1, sustain_level, d_samp)
        current += d_samp
    if s_samp > 0:
        env[current:current+s_samp] = sustain_level
        current += s_samp
    if r_samp > 0:
        remaining = n_samples - current
        count = min(r_samp, remaining)
        env[current:current+count] = np.linspace(sustain_level, 0, count)
    return env

# --- SYNTHESIZER --- 

class Synthesizer:
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate

    def add_transient(self, wave, type, level):
        n = len(wave)
        noise = np.random.uniform(-1, 1, n)
        
        if type == 'breath': 
            env = np.exp(-np.linspace(0, 20, n))
            transient = apply_filter(noise, 2000, self.sr, order=2) * env
        elif type == 'bow': 
            env = np.exp(-np.linspace(0, 10, n))
            mod = np.sin(2 * np.pi * 50 * np.linspace(0, n/self.sr, n)) 
            transient = apply_filter(noise, 4000, self.sr, order=2) * env * (0.5 + 0.5*mod)
        elif type == 'pluck': 
            env = np.exp(-np.linspace(0, 50, n))
            transient = apply_filter(noise, 3000, self.sr, order=2) * env
        else:
            return wave
            
        return wave + (transient * level)

    def generate_tone(self, freq, duration, preset, velocity=0.8):
        freq_drift = freq + np.random.uniform(-0.3, 0.3)
        n_samples = int(self.sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        osc_type = preset.get('osc', 'sine')
        
        if osc_type == 'karplus':
            wave = self.karplus_strong(freq_drift, duration, decay_stretch=preset.get('decay_stretch', 1.0))
        elif osc_type == 'fm':
            wave = self.fm_synthesis(freq_drift, duration, preset)
        elif osc_type == 'piano_sim':
            wave = self.piano_simulation(freq_drift, duration)
        elif osc_type == 'saw':
            wave = get_saw(freq_drift, t)
            wave += 0.25 * get_square(freq_drift/2, t)
        elif osc_type == 'square':
            wave = get_square(freq_drift, t)
        elif osc_type == 'sine':
            wave = get_sine(freq_drift, t)
            wave = np.tanh(wave * 1.5) 
            
        cutoff_base = preset.get('filter_cutoff', 20000) 
        if cutoff_base < 19000:
            dynamic_cutoff = cutoff_base * (0.5 + 0.5 * velocity)
            dynamic_cutoff += freq * 0.8
            wave = apply_filter(wave, dynamic_cutoff, self.sr)
            
        if 'transient_type' in preset:
            wave = self.add_transient(wave, preset['transient_type'], preset.get('transient_level', 0.1))

        if 'vibrato_rate' in preset:
            rate = preset['vibrato_rate']
            depth = preset.get('vibrato_depth', 0.1)
            mod = 1.0 - depth/2 + (depth/2) * np.sin(2 * np.pi * rate * t)
            wave *= mod

        att = preset.get('attack', 0.01)
        dec = preset.get('decay', 0.1)
        sus = preset.get('sustain', 0.7)
        rel = preset.get('release', 0.1)
        
        env = adsr_envelope(n_samples, att, dec, sus, rel, n_samples)
        wave *= env
        
        if preset.get('chorus', False):
            wave = apply_chorus(wave, self.sr)
            
        if preset.get('overdrive', False):
            drive = 5.0 * velocity
            wave = np.tanh(wave * drive) 
        
        final_vol = preset.get('volume', 0.5) * (velocity ** 2) 
        return wave * final_vol

    def karplus_strong(self, freq, duration, decay_stretch=1.0):
        if freq < 20: freq = 20
        N = int(self.sr / freq)
        burst = apply_filter(np.random.uniform(-1, 1, N), 4000, self.sr) 
        n_samples = int(self.sr * duration)
        passes = int(n_samples / N) + 1
        output_blocks = []
        current = burst
        loss = 0.996 ** (1.0/decay_stretch)
        for _ in range(passes):
            output_blocks.append(current)
            current = loss * 0.5 * (current + np.roll(current, 1))
        return np.concatenate(output_blocks)[:n_samples]

    def fm_synthesis(self, freq, duration, preset):
        n_samples = int(self.sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        ratio = preset.get('fm_ratio', 1.0)
        index = preset.get('fm_index', 1.0)
        mod_freq = freq * ratio
        mod_env = np.exp(-t * 5) 
        modulator = index * mod_env * np.sin(2 * np.pi * mod_freq * t)
        carrier = np.sin(2 * np.pi * freq * t + modulator)
        return carrier

    def piano_simulation(self, freq, duration):
        n_samples = int(self.sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        h_len = int(0.04 * self.sr)
        thump = np.zeros(n_samples)
        if h_len < n_samples:
            thump[:h_len] = apply_filter(np.random.uniform(-0.5, 0.5, h_len) * np.linspace(1,0,h_len), 300, self.sr)
        w1 = get_sine(freq, t)
        w2 = 0.5 * get_sine(freq*2.005, t)
        w3 = 0.2 * get_sine(freq*3.01, t)
        wave = w1 + w2 + w3 + thump
        wave = apply_filter(wave, 1500, self.sr)
        return wave


INSTRUMENTS = {
    # KEYS
    'grand_piano': {'osc': 'piano_sim', 'attack': 0.01, 'decay': 0.8, 'sustain': 0.0, 'release': 0.4, 'volume': 0.8, 'transient_type': 'pluck', 'transient_level': 0.05},
    'e_piano': {'osc': 'fm', 'fm_ratio': 14.0, 'fm_index': 0.8, 'attack': 0.01, 'decay': 0.5, 'sustain': 0.7, 'release': 0.3, 'chorus': True, 'volume': 0.7},
    'vibraphone': {'osc': 'fm', 'fm_ratio': 3.5, 'fm_index': 0.5, 'attack': 0.01, 'decay': 1.5, 'sustain': 0.1, 'release': 1.0, 'vibrato_rate': 4.5, 'volume': 0.6, 'transient_type': 'pluck', 'transient_level': 0.1},
    'organ': {'osc': 'sine', 'attack': 0.05, 'decay': 0.1, 'sustain': 1.0, 'release': 0.1, 'chorus': True, 'volume': 0.5, 'transient_type': 'breath', 'transient_level': 0.05}, 
    
    # GUITARS
    'acoustic_guitar': {'osc': 'karplus', 'decay_stretch': 1.0, 'volume': 0.7, 'transient_type': 'pluck', 'transient_level': 0.1},
    'electric_guitar': {'osc': 'karplus', 'decay_stretch': 2.5, 'chorus': True, 'volume': 0.6, 'transient_type': 'pluck', 'transient_level': 0.05},
    'bass_guitar': {'osc': 'fm', 'fm_ratio': 0.5, 'fm_index': 1.5, 'attack': 0.02, 'decay': 0.3, 'sustain': 0.6, 'release': 0.2, 'volume': 0.9, 'transient_type': 'pluck', 'transient_level': 0.1},
    'distorted_guitar': {'osc': 'saw', 'filter_cutoff': 3000, 'attack': 0.02, 'decay': 0.5, 'sustain': 0.8, 'release': 0.2, 'overdrive': True, 'volume': 0.5},
    
    # STRINGS
    'violin': {'osc': 'saw', 'filter_cutoff': 1800, 'attack': 0.1, 'decay': 0.3, 'sustain': 0.9, 'release': 0.4, 'vibrato_rate': 6.0, 'chorus': True, 'volume': 0.5, 'transient_type': 'bow', 'transient_level': 0.15},
    'cello': {'osc': 'saw', 'filter_cutoff': 700, 'attack': 0.2, 'decay': 0.5, 'sustain': 0.9, 'release': 0.6, 'vibrato_rate': 4.0, 'volume': 0.6, 'transient_type': 'bow', 'transient_level': 0.12},
    'pizzicato': {'osc': 'sine', 'attack': 0.005, 'decay': 0.2, 'sustain': 0.0, 'release': 0.1, 'volume': 0.6, 'transient_type': 'pluck', 'transient_level': 0.1},
    'harp': {'osc': 'karplus', 'decay_stretch': 2.0, 'volume': 0.7, 'transient_type': 'pluck', 'transient_level': 0.05},
    
    # WINDS
    'flute': {'osc': 'sine', 'attack': 0.1, 'decay': 0.2, 'sustain': 0.9, 'release': 0.2, 'vibrato_rate': 5, 'volume': 0.4, 'transient_type': 'breath', 'transient_level': 0.2},
    'brass_section': {'osc': 'saw', 'filter_cutoff': 2500, 'attack': 0.1, 'decay': 0.2, 'sustain': 0.8, 'release': 0.3, 'chorus': True, 'volume': 0.5, 'transient_type': 'breath', 'transient_level': 0.1},
    'trumpet': {'osc': 'saw', 'filter_cutoff': 4000, 'attack': 0.05, 'decay': 0.3, 'sustain': 0.7, 'release': 0.2, 'volume': 0.5, 'transient_type': 'breath', 'transient_level': 0.15},
    'saxophone': {'osc': 'square', 'filter_cutoff': 1500, 'attack': 0.1, 'decay': 0.2, 'sustain': 0.8, 'release': 0.2, 'vibrato_rate': 5.5, 'volume': 0.5, 'transient_type': 'breath', 'transient_level': 0.25},

    # SYNTHS
    'classic_synth': {'osc': 'saw', 'filter_cutoff': 5000, 'attack': 0.01, 'decay': 0.2, 'sustain': 0.6, 'release': 0.2, 'chorus': True, 'volume': 0.5},
    'sci_fi': {'osc': 'fm', 'fm_ratio': 2.01, 'fm_index': 3.0, 'attack': 0.2, 'decay': 1.0, 'sustain': 0.5, 'release': 1.0, 'volume': 0.4},
    'chiptune': {'osc': 'square', 'attack': 0.001, 'decay': 0.1, 'sustain': 0.3, 'release': 0.1, 'volume': 0.35},
    '80s_pad': {'osc': 'saw', 'filter_cutoff': 1000, 'attack': 1.0, 'decay': 1.0, 'sustain': 1.0, 'release': 1.0, 'chorus': True, 'volume': 0.4},
}

# --- RHYTHM & MELODY ---

def generate_kick(sr):
    t = np.linspace(0, 0.4, int(0.4*sr))
    freq_env = np.linspace(200, 40, len(t))
    amp_env = np.exp(-t * 10)
    wave = np.sin(2 * np.pi * freq_env * t) * amp_env
    click = apply_filter(np.random.uniform(-0.5, 0.5, int(0.01*sr)), 5000, sr, order=2)
    wave[:len(click)] += click
    return wave * 0.8

def generate_snare(sr):
    t = np.linspace(0, 0.25, int(0.25*sr))
    tone = np.sin(2 * np.pi * 180 * t) * np.exp(-t * 20)
    noise = apply_filter(np.random.uniform(-1, 1, len(t)), 8000, sr) 
    noise *= np.exp(-t * 15)
    return (tone * 0.5 + noise * 0.5) * 0.7

def generate_hihat(sr):
    t = np.linspace(0, 0.08, int(0.08*sr))
    noise = np.random.uniform(-1, 1, len(t))
    mod = np.sin(2 * np.pi * 9000 * t)
    wave = noise * mod * np.exp(-t * 60)
    return apply_filter(wave, 10000, sr) 

def get_drum_sequence_16():
    styles = [
        ([1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0], [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0], [0,0,1,0, 0,0,1,0, 0,0,1,0, 0,0,1,0]),
        ([1,0,0,0, 0,0,1,0, 0,0,1,0, 1,0,0,0], [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0], [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0]),
        ([1,0,0,1, 0,0,1,0, 0,0,0,0, 1,0,0,0], [0,0,0,0, 1,0,0,0, 0,0,1,0, 1,0,0,0], [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1]),
    ]
    k, s, h = random.choice(styles)
    return list(k), list(s), list(h)

def generate_melody_16():
    motif = []
    for i in range(16):
        if random.random() > 0.4:
            motif.append({"idx": random.randint(0, 5), "len": random.choice([1, 2])})
        else:
            motif.append(None)
    return motif

# Helper to map generic index to MIDI-ish note number (Roughly C4 = 60)
def index_to_midi_note(idx, base_note_freq):
    # Mapping freq back to note number for frontend visual
    # MIDI 69 = A4 = 440Hz.
    # n = 12*log2(f/440) + 69
    import math
    
    scale_semitones = [0, 3, 5, 7, 10, 12, 15, 17]
    idx_mod = idx % len(scale_semitones)
    octave = idx // len(scale_semitones)
    semitone_offset = scale_semitones[idx_mod] + (12 * octave)
    
    # Calculate freq just to reverse it (or map base note to midi)
    # Let's map base 'idx' 0 to C4 (60) for simplicity in visualization, 
    # even if pitch varies randomly. 
    # Scale: C, Eb, F, G, Bb, C
    scale_midi = [0, 3, 5, 7, 10, 12, 15, 17]
    base_midi = 60 # Default to C4
    
    # Simple heuristic
    current_midi = base_midi + semitone_offset
    return current_midi

def note_to_freq(note_index, base_freq): 
    scale_semitones = [0, 3, 5, 7, 10, 12, 15, 17]
    idx = note_index % len(scale_semitones)
    octave = note_index // len(scale_semitones)
    semitone = scale_semitones[idx] + (12 * octave)
    return base_freq * (2 ** (semitone / 12.0))

def generate_beat(bpm: int, instrument_names: list[str], duration_sec: int = 15):
    if not instrument_names: instrument_names = ['classic_synth']
    sample_rate = 44100
    synth = Synthesizer(sample_rate)
    
    beat_sec = 60 / bpm
    step_sec = beat_sec / 4 
    step_samps = int(step_sec * sample_rate)
    loop_steps = 16
    
    k_pat, s_pat, h_pat = get_drum_sequence_16()
    melody_pat = generate_melody_16()
    
    total_samps = int(sample_rate * duration_sec)
    master = np.zeros(total_samps + sample_rate * 2) 
    
    kick = generate_kick(sample_rate)
    snare = generate_snare(sample_rate)
    hihat = generate_hihat(sample_rate)
    
    base_note =  random.choice([261.63, 220.0, 196.0, 293.66]) 
    
    # --- EVENTS LIST FOR FRONTEND ---
    note_events = []
    
    steps = int(total_samps / step_samps)
    for step in range(steps):
        if step * step_samps >= total_samps: break
        
        step_mod = step % loop_steps 
        pos = step * step_samps
        start_time = pos / sample_rate
        
        pos += random.randint(-80, 80)
        if pos < 0: pos = 0
        
        if k_pat[step_mod]:
            l = len(kick)
            master[pos:pos+l] += kick * random.uniform(0.95, 1.05)
        if s_pat[step_mod]:
            l = len(snare)
            master[pos:pos+l] += snare * random.uniform(0.9, 1.0)
        if h_pat[step_mod]:
            l = len(hihat)
            master[pos:pos+l] += hihat * random.uniform(0.8, 1.0)
            
        note = melody_pat[step_mod]
        if note:
            velocity = random.uniform(0.7, 1.0)
            freq = note_to_freq(note['idx'], base_note)
            
            # Helper: Calculate approximate MIDI note 
            # f = 440 * 2^((d-69)/12) -> d = 69 + 12*log2(f/440)
            import math
            midi_note = int(69 + 12 * math.log2(freq / 440.0))
            
            # Store Event
            duration_real = float(note['len'] * step_samps) / sample_rate
            note_events.append({
                "t": start_time,
                "note": midi_note, 
                "len": duration_real
            })

            for inst in instrument_names:
                if inst not in INSTRUMENTS: continue
                preset = INSTRUMENTS[inst].copy() 
                preset['volume'] = preset.get('volume', 0.5) * velocity
                
                run_freq = freq
                if 'bass' in inst or 'cello' in inst: run_freq /= 2
                if 'piccolo' in inst: run_freq *= 2
                
                wave = synth.generate_tone(run_freq, duration_real, preset, velocity=velocity)
                
                l = len(wave)
                e = min(pos + l, len(master))
                mix_vol = 0.8 / max(1, len(instrument_names) * 0.7)
                master[pos:e] += wave[:e-pos] * mix_vol

    final = master[:total_samps]
    final = apply_reverb(final, sample_rate, room_size=0.7)
    
    m = np.max(np.abs(final))
    if m > 1e-9: final = final / m * 0.95
        
    final_int16 = (final * 32767).astype(np.int16)
    wav_io = io.BytesIO()
    write(wav_io, sample_rate, final_int16)
    wav_io.seek(0)
    
    return wav_io, note_events
