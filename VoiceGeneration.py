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

# Texte à générer
# Texte à dire
text = """
L'IA est un domaine de l'informatique qui vise à créer des systèmes capables d'imiter ou de simuler l'intelligence humaine.
L'apprentissage automatique (Machine Learning) se concentre sur la construction de systèmes qui apprennent et s'améliorent à partir de l'expérience sans être explicitement programmés.
L'apprentissage profond (Deep Learning) utilise des réseaux neuronaux avec de nombreuses couches pour modéliser des motifs complexes dans les données.
L'IA générative peut créer ou générer de nouveaux contenus, idées ou données qui ressemblent à la créativité humaine.
"""

slide14 ="""Le Machine Learning trouve des applications dans de nombreux domaines. En santé, il aide au diagnostic et à la découverte de médicaments. En finance, il est utilisé pour la détection de fraudes et la prédiction de marchés. Dans le marketing, il permet la personnalisation et la segmentation de clients. Et dans l'industrie, il optimise les processus de fabrication et la maintenance prédictive."""
slide15 ="""Un exemple marqueur d'apprentissage supervisé appliqué à la vidéo est l'algorithme YOLO, qui signifie 'You Only Look Once'. Ce système permet de détecter et de classifier des objets dans des vidéos en temps réel, avec une seule évaluation du réseau de neurones par image. C'est une approche particulièrement efficace pour des applications comme la surveillance vidéo ou les voitures autonomes."""

# Fichier de sortie
output_path = "output_slide15.wav"

# Génération audio
tts.tts_to_file(
    text=slide15,
    file_path=output_path,
    speaker_wav="Enregistremen6.wav",  # mettre un fichier wav pour faire du clonage de voix
    language="fr"
)

print(f"✅ Fichier généré : {output_path}")

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
        speaker_wav="Enregistremen6.wav",  # mettre un fichier wav pour faire du clonage de voix
        language="fr"
    )

    print(f"✅ Fichier généré : {output_path}")
