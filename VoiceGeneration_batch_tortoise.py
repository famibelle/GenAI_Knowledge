import os
import re
import pandas as pd
from tqdm import tqdm

import torch
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio
import numpy as np
import soundfile as sf

# Paramètres
EXCEL_PATH = "VoixOff/Voix Off.xlsx"
REFERENCE_WAV = "Voices/MédhiCloneHigh.wav"  # fichier de référence pour la voix
OUTPUT_DIR = "tortoise_outputs"
PRESET = "fast"  # autres: "ultra_fast", "standard", "high_quality" (si supporté)
USE_PRESET = True  # sera désactivé si la lib ne supporte pas 'preset'

# Configuration GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_DEEPSPEED = False  # Option pour DeepSpeed si disponible
KVCACHE = True  # Cache pour améliorer les performances sur GPU

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Vérification de la disponibilité du GPU
print(f"📊 Configuration d'exécution:")
print(f"   - Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"   - GPU: {torch.cuda.get_device_name(0)}")
    print(f"   - Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("   - Mode CPU (GPU non disponible)")
print(f"   - DeepSpeed: {USE_DEEPSPEED}")
print(f"   - KV Cache: {KVCACHE}")

# Chargement du sample de référence
if not os.path.exists(REFERENCE_WAV):
    raise FileNotFoundError(f"Fichier voix référence introuvable: {REFERENCE_WAV}")

ref_sample = load_audio(REFERENCE_WAV, 22050)
voice_samples = [ref_sample]
conditioning_latents = None  # Laisser None pour calcul automatique (plus lent la 1ère fois)

# Init du modèle
print(f"🔥 Initialisation de Tortoise TTS sur {DEVICE}...")
try:
    # Initialisation du modèle avec configuration GPU
    if DEVICE == "cuda":
        # Optimisations pour GPU
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Initialisation avec paramètres GPU optimisés
        tts = TextToSpeech(
            device=DEVICE,
            kv_cache=KVCACHE,
            use_deepspeed=USE_DEEPSPEED
        )
    else:
        # Mode CPU
        tts = TextToSpeech(device="cpu")
        
    print(f"✅ Modèle Tortoise TTS initialisé avec succès")
except Exception as e:
    print(f"❌ Erreur lors de l'initialisation du modèle: {e}")
    # Fallback: essayer sans paramètres spéciaux
    try:
        tts = TextToSpeech()
        print(f"⚠️ Modèle initialisé en mode de compatibilité")
    except Exception as e2:
        raise RuntimeError(f"Impossible d'initialiser Tortoise TTS: {e2}")

# Gestion de la mémoire GPU
if DEVICE == "cuda":
    torch.cuda.empty_cache()
    print(f"🧹 Cache GPU vidé après initialisation")

# Pré-calcul des conditioning latents (plus rapide ensuite)
try:
    if 'voice_samples' in globals() and voice_samples:
        conditioning_latents = tts.get_conditioning_latents(voice_samples=voice_samples)
        print(f"✅ Conditioning latents pré-calculés")
except Exception as _e:
    print(f"⚠️ Impossible de pré-calculer les conditioning latents: {_e}")

# --- Nouveau: fonction de découpage ---
MAX_CHARS = 400  # ajuster si nécessaire
SENTENCE_SEP_REGEX = r'(?<=[.!?])\s+'

def chunk_text(text: str, max_chars: int = MAX_CHARS):
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    import re
    sentences = re.split(SENTENCE_SEP_REGEX, text)
    chunks = []
    current = ''
    for s in sentences:
        if not s:
            continue
        candidate = (current + ' ' + s).strip() if current else s.strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # si phrase seule trop longue, on coupe brutalement
            if len(s) > max_chars:
                for i in range(0, len(s), max_chars):
                    chunks.append(s[i:i+max_chars])
                current = ''
            else:
                current = s
    if current:
        chunks.append(current)
    return chunks
# --- Fin nouveau ---

def slugify(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[\\s]+", "_", value)
    value = re.sub(r"[^A-Za-z0-9_-]", "", value)
    return value[:60]  # limiter longueur

# Charger Excel
data = pd.read_excel(EXCEL_PATH)

required_cols = {"Slide Number", "Slide Title", "Voice Over Text"}
missing = required_cols - set(data.columns)
if missing:
    raise ValueError(f"Colonnes manquantes dans le fichier Excel: {missing}")

for _, row in tqdm(data.iterrows(), total=len(data), desc="Génération Tortoise"):
    slide_number = row['Slide Number']
    slide_title = str(row['Slide Title']) if not pd.isna(row['Slide Title']) else f"{slide_number}"
    voice_text = str(row['Voice Over Text']) if not pd.isna(row['Voice Over Text']) else ""

    if not voice_text.strip():
        print(f"⚠️ Texte vide pour la slide {slide_number}, passage.")
        continue

    safe_title = slugify(slide_title)
    output_path = os.path.join(OUTPUT_DIR, f"Slide{slide_number}_{safe_title}_tortoise-optim.wav")

    # Génération
    try:
        # Nouveau: utilisation d'un pipeline par segments si texte trop long
        segments = chunk_text(voice_text)
        segment_audios = []
        for seg_idx, seg in enumerate(segments, start=1):
            # Choisir API disponible
            if hasattr(tts, 'tts'):
                # Essai avec 'preset' puis fallback sans
                try:
                    if USE_PRESET:
                        result = tts.tts(
                            text=seg,
                            voice_samples=voice_samples,
                            conditioning_latents=conditioning_latents,
                            preset=PRESET,
                        )
                    else:
                        result = tts.tts(
                            text=seg,
                            voice_samples=voice_samples,
                            conditioning_latents=conditioning_latents,
                        )
                except (TypeError, ValueError) as e_call:
                    msg = str(e_call)
                    if 'preset' in msg or 'model_kwargs' in msg or 'unexpected keyword argument' in msg:
                        if USE_PRESET:
                            print("   -> 'preset' non supporté par cette version, on recommence sans.")
                            USE_PRESET = False
                            result = tts.tts(
                                text=seg,
                                voice_samples=voice_samples,
                                conditioning_latents=conditioning_latents,
                            )
                        else:
                            raise
                    else:
                        raise
                if isinstance(result, dict):
                    audio = result.get('wav') or result.get('audio')
                else:
                    audio = result
                if isinstance(audio, torch.Tensor):
                    audio = audio.squeeze().cpu().numpy()
                segment_audios.append(audio)
            elif hasattr(tts, 'tts_to_file') and len(segments) == 1:
                # Cas simple (un seul segment) on peut utiliser tts_to_file directement
                try:
                    if USE_PRESET:
                        tts.tts_to_file(
                            text=voice_text,
                            voice_samples=voice_samples,
                            conditioning_latents=conditioning_latents,
                            file_path=output_path,
                            preset=PRESET,
                        )
                    else:
                        tts.tts_to_file(
                            text=voice_text,
                            voice_samples=voice_samples,
                            conditioning_latents=conditioning_latents,
                            file_path=output_path,
                        )
                except (TypeError, ValueError) as e_call:
                    msg = str(e_call)
                    if 'preset' in msg or 'model_kwargs' in msg or 'unexpected keyword argument' in msg:
                        if USE_PRESET:
                            print("   -> 'preset' non supporté par cette version, on recommence sans.")
                            USE_PRESET = False
                            tts.tts_to_file(
                                text=voice_text,
                                voice_samples=voice_samples,
                                conditioning_latents=conditioning_latents,
                                file_path=output_path,
                            )
                        else:
                            raise
                segment_audios = []  # déjà écrit
                break
            else:
                raise AttributeError("API Tortoise inconnue: ni tts() (pour segmentation) ni tts_to_file adaptée.")
            print(f"   -> Segment {seg_idx}/{len(segments)} ok")

        # Concaténation & écriture si segments générés en mémoire
        if segment_audios:
            full = np.concatenate(segment_audios)
            sf.write(output_path, full, 22050)

        print(f"✅ Fichier généré : {output_path}")
    except Exception as e:
        print(f"❌ Erreur slide {slide_number}: {e}")

    # Gestion de la mémoire GPU si nécessaire
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

print("Terminé.")
