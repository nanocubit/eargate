# EarGate

> Acoustic siren detection → automatic barrier opening for emergency vehicles.
> Works with Yandex Alice Smart Home, runs on Raspberry Pi 4B.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%204B-red.svg)](https://www.raspberrypi.com/)

## What it does

Listens for emergency vehicle sirens (fire truck, ambulance, police), recognizes
the class by acoustic fingerprint (MFCC + SVM), and:

1. **Opens the barrier** — locally via GPIO, no internet required
2. **Notifies Yandex Alice** — voice announcement on Mini Station (optional)

Critical path (barrier opening) is fully local. Non-critical (voice notification)
uses the cloud. If internet is down — barrier still opens.

## Why this exists

- Emergency vehicles lose 30–60 seconds at barriers waiting for a guard
- Camera-based recognition fails at night, in fog, in snow, at bad angles
- Acoustic recognition works in any weather, from 300+ meters away
- Commercial acoustic EVP systems cost 100k+ ₽ and don't integrate with Alice

## Features

- 🎙 **Robust ML pipeline** — 48 features (MFCC + spectral + autocorrelation)
- 🚦 **4-level false-positive protection** — band filter, moving average,
  duration filter, consecutive hits
- ⚡ **Local critical path** — GPIO relay pulses in <10 ms, no cloud dependency
- 🗣 **Yandex Alice integration** — via Smart Home Provider API
- 🔧 **Two microphones** — separate recognition for entry/exit
- 📊 **Detection logging** — audio saved for false-positive analysis
- 🐳 **systemd-ready** — auto-restart, healthcheck endpoint

## Architecture

```
   ┌──────────────────────────────┐
   │   Raspberry Pi 4B + relay    │
   │                              │
[mic]──USB──┤  Detector → GPIO ────┼──GPIO──[relay]──[barrier]
[mic]──USB──┤  Webhook :8080       │
   │                              │
   └────────────┬─────────────────┘
                │ Cloudflare Tunnel (HTTPS)
                ↓
   ┌──────────────────────────────┐
   │       Yandex Cloud           │
   │      (Provider API)          │
   └────────────┬─────────────────┘
                ↓
   ┌──────────────────────────────┐
   │      Mini Station 2          │
   │  "Внимание, скорая на въезде"│
   └──────────────────────────────┘
```

## Quick start

```bash
# 1. Install
git clone https://github.com/yourname/eargate
cd eargate
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Train on synthetic data
python main.py train

# 3. Configure
cp deploy/eargate.env.example .env
# Fill in YANDEX_SKILL_ID and YANDEX_OAUTH_TOKEN

# 4. Run
set -a && source .env && set +a
python main.py run
```

## Hardware

| Component | Price | Notes |
|---|---|---|
| Raspberry Pi 4B 2GB | ~5000 ₽ | Two USB ports for two mics |
| USB microphone × 2 | ~600 ₽ | Any cardioid USB mic |
| Relay module 5V | ~200 ₽ | Opto-isolated, active-low |
| External 5V PSU | ~300 ₽ | For relay — separate from Pi |
| microSD 32GB | ~500 ₽ | A1/A2 class |
| **Total** | **~7000 ₽** | |

## How the ML works

1. **Audio capture** — 22050 Hz, 3-second circular buffer, 0.5 s hop
2. **Feature extraction** — 48 features: 13 MFCC + Δ + std, spectral centroid,
   ZCR, flatness, rolloff, RMS, modulation rate, autocorrelation
3. **Fast band rejection** — if <30% energy in 150–2000 Hz (ГОСТ Р 50574-2019) → skip
4. **Classification** — SVM (RBF) + CalibratedClassifierCV
5. **False-positive filters**:
   - Moving average over 3 frames
   - Duration ≥ 0.8 s (claxon is 0.2–0.5 s)
   - Consecutive hits = 2
   - Cooldown 30 s per class+direction

## Detection delay

End-to-end from siren start to relay pulse: **~5–6 seconds**.
At 300+ m detection range, barrier opens well before vehicle arrives.

## Config

All constants in `siren_config.py`. Override via environment:

```env
YANDEX_SKILL_ID=...
YANDEX_OAUTH_TOKEN=...
AUDIO_SOURCE=microphone          # or "rtsp"
MIC_ENTRY_DEVICE_INDEX=0
MIC_EXIT_DEVICE_INDEX=1
WEBHOOK_PORT=8080
```

## Dataset collection

For production quality, collect real siren recordings:

```bash
python scripts/collect_dataset.py   # downloads from YouTube, slices to 3s
```

Then fine-tune: merge synthetic + real, retrain.

## Deployment

```bash
sudo cp deploy/eargate.service /etc/systemd/system/
sudo cp deploy/cloudflared.service /etc/systemd/system/
sudo systemctl enable --now eargate cloudflared
```

Cloudflare Tunnel gives HTTPS endpoint for the webhook without a static IP.

## Roadmap

- [x] Synthetic dataset with Doppler, reverb, distortion
- [x] Real-time detector with false-positive protection
- [x] GPIO relay control
- [x] Yandex Alice integration
- [ ] Real dataset collection (15–20 min per class)
- [ ] Multimodal upgrade: video (flashing lights) + audio
- [ ] E2PANNs / small CNN if SVM accuracy plateau

## License

MIT — see [LICENSE](LICENSE).
