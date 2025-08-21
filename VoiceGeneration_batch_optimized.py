#!/usr/bin/env python3
"""
Script de génération vocale optimisé pour le français avec support GPU
Basé sur XTTS-v2 avec améliorations pour la prononciation française
"""

import torch
import pandas as pd
import os
import re
import time
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

# Import TTS
from TTS.api import TTS
from TTS.tts.models.xtts import XttsArgs

# Configuration et paramètres
class VoiceConfig:
    """Configuration centralisée pour la génération vocale"""
    
    # Chemins des fichiers
    EXCEL_PATH = "VoixOff/Voix Off.xlsx"
    REFERENCE_WAV = "Voices/MédhiCloneHigh.wav"
    OUTPUT_DIR = "Generated"
    
    # Configuration GPU
    USE_GPU = torch.cuda.is_available()
    DEVICE = "cuda" if USE_GPU else "cpu"
    
    # Paramètres TTS optimisés pour le français
    TTS_CONFIG = {
        "language": "fr",
        "speed": 1.0,
        "emotion": "neutral",
        "split_sentences": True,
        "temperature": 0.75,  # Contrôle la variabilité
        "top_k": 50,          # Limite les choix de tokens
        "top_p": 0.85,        # Nucleus sampling
        "repetition_penalty": 1.1,  # Évite les répétitions
        "length_penalty": 1.0
    }
    
    # Limites de traitement
    MAX_TEXT_LENGTH = 500
    CHUNK_OVERLAP = 50

# Patch pour la sérialisation
torch.serialization.add_safe_globals([XttsArgs])

class FrenchTextProcessor:
    """Préprocesseur de texte optimisé pour le français"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Nettoie et normalise le texte pour une meilleure prononciation"""
        if not text or pd.isna(text):
            return ""
        
        text = str(text).strip()
        
        # Corrections phonétiques françaises
        corrections = {
            # Acronymes et sigles
            'AI': 'Intelligence Artificielle',
            'IA': 'Intelligence Artificielle',
            'ML': 'Machine Learning',
            'API': 'A-P-I',
            'URL': 'U-R-L',
            'HTTP': 'H-T-T-P',
            'GPU': 'G-P-U',
            'CPU': 'C-P-U',
            
            # Mots techniques anglais
            'cloud computing': 'informatique en nuage',
            'machine learning': 'apprentissage automatique',
            'deep learning': 'apprentissage profond',
            'data science': 'science des données',
            'big data': 'mégadonnées',
            'workflow': 'flux de travail',
            'framework': 'cadre de développement',
            
            # Améliorations phonétiques
            'vs': 'versus',
            '&': 'et',
            '@': 'arobase',
            '%': 'pour cent',
            '€': 'euros',
            '$': 'dollars',
            
            # Corrections de liaison
            ' et ': ' é ',
            ' est ': ' è ',
            ' un ': ' un-n ',
            ' en ': ' an ',
        }
        
        for old, new in corrections.items():
            text = text.replace(old, new)
        
        # Amélioration de la ponctuation pour des pauses naturelles
        text = re.sub(r'([.!?])', r'\1 [pause_longue] ', text)
        text = re.sub(r'([,;:])', r'\1 [pause_courte] ', text)
        text = re.sub(r'([()[\]])', r' \1 ', text)
        
        # Nettoyage des espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def split_text(text: str, max_length: int = VoiceConfig.MAX_TEXT_LENGTH) -> List[str]:
        """Divise le texte en segments pour éviter les timeouts"""
        if len(text) <= max_length:
            return [text]
        
        # Division par phrases
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = f"{current_chunk} {sentence}".strip()
            
            if len(test_chunk) <= max_length:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Si une phrase est trop longue, la diviser par mots
                if len(sentence) > max_length:
                    words = sentence.split()
                    temp_chunk = ""
                    
                    for word in words:
                        test_word_chunk = f"{temp_chunk} {word}".strip()
                        if len(test_word_chunk) <= max_length:
                            temp_chunk = test_word_chunk
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk)
                            temp_chunk = word
                    
                    if temp_chunk:
                        current_chunk = temp_chunk
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

class VoiceGenerator:
    """Générateur vocal optimisé"""
    
    def __init__(self):
        self.config = VoiceConfig()
        self.processor = FrenchTextProcessor()
        self.tts = None
        self._setup_directories()
        self._initialize_tts()
    
    def _setup_directories(self):
        """Crée les dossiers nécessaires"""
        Path(self.config.OUTPUT_DIR).mkdir(exist_ok=True)
        print(f"📁 Dossier de sortie: {self.config.OUTPUT_DIR}")
    
    def _initialize_tts(self):
        """Initialise le modèle TTS avec optimisations"""
        print(f"🔥 Initialisation XTTS-v2 sur {self.config.DEVICE}")
        print(f"   GPU disponible: {self.config.USE_GPU}")
        
        if self.config.USE_GPU:
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        start_time = time.time()
        
        try:
            self.tts = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                gpu=self.config.USE_GPU
            )
            
            # Optimisations GPU
            if self.config.USE_GPU:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.cuda.empty_cache()
            
            load_time = time.time() - start_time
            print(f"✅ Modèle chargé en {load_time:.1f}s")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            raise
    
    def _validate_reference_voice(self) -> bool:
        """Vérifie que le fichier de référence vocal existe"""
        if not Path(self.config.REFERENCE_WAV).exists():
            print(f"❌ Fichier de référence introuvable: {self.config.REFERENCE_WAV}")
            print("💡 Assurez-vous d'avoir un fichier .wav de référence pour le clonage de voix")
            return False
        print(f"✅ Fichier de référence trouvé: {self.config.REFERENCE_WAV}")
        return True
    
    def _safe_filename(self, title: str, slide_number: int) -> str:
        """Génère un nom de fichier sûr"""
        # Nettoie le titre pour en faire un nom de fichier valide
        clean_title = re.sub(r'[^\w\s-]', '', str(title))
        clean_title = re.sub(r'[-\s]+', '_', clean_title)
        clean_title = clean_title[:50]  # Limite la longueur
        
        return f"Slide{slide_number:03d}_{clean_title}_optimized.wav"
    
    def generate_voice(self, text: str, output_path: str) -> bool:
        """Génère un fichier audio à partir du texte"""
        try:
            # Préprocessing du texte
            processed_text = self.processor.clean_text(text)
            if not processed_text:
                print("⚠️ Texte vide après préprocessing")
                return False
            
            # Division en segments si nécessaire
            text_chunks = self.processor.split_text(processed_text)
            
            if len(text_chunks) == 1:
                # Génération simple
                self.tts.tts_to_file(
                    text=text_chunks[0],
                    file_path=output_path,
                    speaker_wav=self.config.REFERENCE_WAV,
                    **self.config.TTS_CONFIG
                )
            else:
                # Génération par segments et concaténation
                print(f"   📝 Traitement en {len(text_chunks)} segments")
                segment_files = []
                
                for i, chunk in enumerate(text_chunks):
                    segment_file = output_path.replace('.wav', f'_segment_{i}.wav')
                    
                    self.tts.tts_to_file(
                        text=chunk,
                        file_path=segment_file,
                        speaker_wav=self.config.REFERENCE_WAV,
                        **self.config.TTS_CONFIG
                    )
                    segment_files.append(segment_file)
                
                # Concaténation des segments (nécessiterait une bibliothèque audio)
                # Pour l'instant, on garde le premier segment
                if segment_files:
                    os.rename(segment_files[0], output_path)
                    # Nettoyer les segments temporaires
                    for seg_file in segment_files[1:]:
                        if os.path.exists(seg_file):
                            os.remove(seg_file)
            
            # Nettoyage mémoire GPU
            if self.config.USE_GPU:
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur de génération: {e}")
            return False
    
    def process_excel_file(self) -> None:
        """Traite le fichier Excel et génère tous les audios"""
        # Vérifications préliminaires
        if not Path(self.config.EXCEL_PATH).exists():
            raise FileNotFoundError(f"Fichier Excel introuvable: {self.config.EXCEL_PATH}")
        
        if not self._validate_reference_voice():
            return
        
        # Chargement du fichier Excel
        print(f"📊 Chargement de {self.config.EXCEL_PATH}")
        try:
            data = pd.read_excel(self.config.EXCEL_PATH)
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement d'Excel: {e}")
        
        # Vérification des colonnes
        required_columns = {'Slide Number', 'Slide Title', 'Voice Over Text'}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(f"Colonnes manquantes: {missing_columns}")
        
        print(f"📈 {len(data)} slides à traiter")
        
        # Variables de suivi
        success_count = 0
        error_count = 0
        start_time = time.time()
        
        # Traitement de chaque ligne
        for index, row in tqdm(data.iterrows(), total=len(data), desc="🎵 Génération vocale"):
            slide_number = row['Slide Number']
            slide_title = row['Slide Title']
            voice_text = row['Voice Over Text']
            
            # Validation des données
            if pd.isna(voice_text) or not str(voice_text).strip():
                print(f"⚠️ Slide {slide_number}: Texte vide, ignoré")
                continue
            
            # Génération du nom de fichier
            output_filename = self._safe_filename(slide_title, slide_number)
            output_path = os.path.join(self.config.OUTPUT_DIR, output_filename)
            
            # Vérifier si le fichier existe déjà
            if os.path.exists(output_path):
                print(f"⏭️ Slide {slide_number}: Fichier existant, ignoré")
                continue
            
            # Génération
            print(f"🎤 Slide {slide_number}: {slide_title}")
            if self.generate_voice(str(voice_text), output_path):
                success_count += 1
                print(f"✅ Généré: {output_filename}")
            else:
                error_count += 1
                print(f"❌ Échec: Slide {slide_number}")
        
        # Rapport final
        total_time = time.time() - start_time
        print(f"\n🎉 Génération terminée!")
        print(f"   ✅ Succès: {success_count}")
        print(f"   ❌ Erreurs: {error_count}")
        print(f"   ⏱️ Temps total: {total_time:.1f}s")
        print(f"   📁 Fichiers dans: {self.config.OUTPUT_DIR}")

def main():
    """Fonction principale"""
    print("🎵 Générateur vocal optimisé pour le français")
    print("=" * 50)
    
    try:
        generator = VoiceGenerator()
        generator.process_excel_file()
        
    except KeyboardInterrupt:
        print("\n⏹️ Arrêt demandé par l'utilisateur")
    except Exception as e:
        print(f"\n💥 Erreur critique: {e}")
        raise

if __name__ == "__main__":
    main()
