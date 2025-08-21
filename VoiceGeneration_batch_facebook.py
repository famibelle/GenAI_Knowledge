import os
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import scipy

from transformers import VitsModel, AutoTokenizer
import torch

# Paramètres
EXCEL_PATH = "VoixOff/Voix Off.xlsx"
OUTPUT_DIR = "Outputs_MMS_FRA"

# Configuration GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import VitsModel, AutoTokenizer
import torch

# Charger modèle + tokenizer
model = VitsModel.from_pretrained("facebook/mms-tts-fra").to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-fra")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Charger le fichier Excel
data = pd.read_excel(EXCEL_PATH)

# Générer pour chaque ligne
for index, row in tqdm(data.iterrows(), total=len(data), desc="Génération audio"):
    slide_number = row['Slide Number']
    slide_title = str(row['Slide Title']).replace(" ", "_")
    voice_text = str(row['Voice Over Text'])

    # Nom du fichier de sortie
    output_path = f"{OUTPUT_DIR}/Slide{slide_number}_{slide_title}.wav"

    # inputs = tokenizer(voice_text, return_tensors="pt")
    # Préparer entrée
    inputs = tokenizer(voice_text, return_tensors="pt").to(DEVICE)
    
    # Génération audio 
    with torch.no_grad():
        output = model(**inputs).waveform

    # Génération audio
    with torch.no_grad():
        waveform = model(**inputs).waveform.cpu().numpy()

    # Sauvegarde
    sf.write(output_path, waveform.squeeze(), model.config.sampling_rate)

    print(f"✅ Fichier généré : {output_path}")
