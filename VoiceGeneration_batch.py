import torch
from TTS.api import TTS
import pandas as pd

# Import de la classe XTTS à autoriser dans le unpickler
from TTS.tts.models.xtts import XttsArgs
from tqdm import tqdm

# Patch pour permettre à torch.load de dé-sérialiser cette classe
torch.serialization.add_safe_globals([XttsArgs])

# Charger le modèle XTTS-v2 en CPU
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# Charger le fichier Excel
excel_path = "Voix Off.xlsx"
data = pd.read_excel(excel_path)

# Parcourir chaque ligne du fichier Excel
for index, row in tqdm(data.iterrows(), total=len(data), desc="Génération audio"):
    slide_number = row['Slide Number']
    slide_title = row['Slide Title']
    voice_text = row['Voice Over Text']

    # Générer le nom du fichier de sortie
    output_path = f"Slide{slide_number}_{slide_title}.wav"

    # Génération audio
    tts.tts_to_file(
        text=voice_text,
        file_path=output_path,
        speaker_wav="MédhiCloneHigh.wav",  # mettre un fichier wav pour faire du clonage de voix
        language="fr"
    )

    print(f"✅ Fichier généré : {output_path}")
