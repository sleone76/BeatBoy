# BeatBoy - Setup & Start

## Voraussetzungen
- **Python 3.8+** muss installiert sein.
- **FFmpeg** wird für MP3-Export empfohlen (sonst Fallback auf WAV oder Fehler, falls pydub es nicht findet).
    - Windows: `winget install Gyan.FFmpeg` oder manuell herunterladen und zu PATH hinzufügen.

## Installation
1. Öffne ein Terminal in diesem Ordner.
2. Installiere die Abhängigkeiten:
   ```bash
   pip install -r requirements.txt
   ```
   *(Falls `pip` nicht gefunden wird, versuche `python -m pip install -r requirements.txt`)*

## Starten

### 1. Backend starten
Öffne ein Terminal und navigiere in den Ordner:
```bash
uvicorn backend.main:app --reload
```
Der Server läuft nun unter `http://127.0.0.1:8000`.

### 2. Frontend nutzen
Öffne die Datei `frontend/index.html` einfach in deinem Browser (Doppelklick).

## Nutzung
- Gib eine BPM Zahl ein (40-240).
- Klicke "Beat Erstellen".
- Warte kurz bis der Audio-Player erscheint.
- Play drücken oder Download nutzen.
