import torch
import pandas as pd
import os
from tqdm import tqdm
import argparse

# Solution pour PyTorch 2.6+ - Ajouter les classes TTS aux globals sûres
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig
    from TTS.config.shared_configs import BaseDatasetConfig, BaseAudioConfig
    
    # Ajouter les classes (pas les strings) aux globals sûres
    torch.serialization.add_safe_globals([
        XttsConfig,
        Xtts, 
        XttsArgs,
        XttsAudioConfig,
        BaseDatasetConfig,
        BaseAudioConfig
    ])
    print("🔐 Classes TTS ajoutées aux globals sûres")
    
except ImportError as e:
    print(f"⚠️  Impossible d'importer certaines classes TTS: {e}")
    print("📝 Le chargement pourrait échouer, mais on continue...")

from TTS.api import TTS

# Patch pour corriger le problème GPT2 avec transformers 4.50+
def apply_gpt2_patch():
    """
    Applique un patch global pour corriger le problème GPT2InferenceModel
    """
    try:
        from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel
        from transformers.generation.utils import GenerationMixin
        
        # Vérifier si GPT2PreTrainedModel hérite déjà de GenerationMixin
        if not issubclass(GPT2PreTrainedModel, GenerationMixin):
            print("🔧 Application du patch GPT2 global...")
            
            # Créer une nouvelle classe qui hérite des deux
            class PatchedGPT2PreTrainedModel(GPT2PreTrainedModel, GenerationMixin):
                pass
            
            # Remplacer la classe dans le module
            import transformers.models.gpt2.modeling_gpt2
            transformers.models.gpt2.modeling_gpt2.GPT2PreTrainedModel = PatchedGPT2PreTrainedModel
            
            # Aussi patcher GPT2LMHeadModel si il existe
            if hasattr(transformers.models.gpt2.modeling_gpt2, 'GPT2LMHeadModel'):
                class PatchedGPT2LMHeadModel(transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel, GenerationMixin):
                    pass
                transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel = PatchedGPT2LMHeadModel
            
            print("✅ Patch GPT2 global appliqué avec succès")
            return True
            
    except Exception as e:
        print(f"⚠️  Patch GPT2 global échoué: {e}")
        return False
    
    return False

# Appliquer le patch avant de charger TTS
print("🔧 Application du patch de compatibilité transformers...")
patch_success = apply_gpt2_patch()

# Patch alternatif pour forcer weights_only=False globalement
print("🔧 Configuration de torch.load pour compatibilité...")
import torch
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

# Configuration GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Utilisation du device: {DEVICE}")


def split_text_by_sentences(text, max_chars=250):
    """
    Divise un texte en segments plus courts en respectant les phrases
    """
    import re
    
    # Nettoyer le texte
    text = str(text).strip()
    
    # Si le texte est déjà court, le retourner tel quel
    if len(text) <= max_chars:
        return [text]
    
    # Diviser par phrases (points, points d'exclamation, points d'interrogation)
    sentences = re.split(r'[.!?]+', text)
    
    segments = []
    current_segment = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Si ajouter cette phrase dépasse la limite, sauvegarder le segment actuel
        if len(current_segment + sentence) > max_chars and current_segment:
            segments.append(current_segment.strip())
            current_segment = sentence
        else:
            if current_segment:
                current_segment += ". " + sentence
            else:
                current_segment = sentence
    
    # Ajouter le dernier segment
    if current_segment.strip():
        segments.append(current_segment.strip())
    
    # Si un segment est encore trop long, le diviser par virgules/points-virgules
    final_segments = []
    for segment in segments:
        if len(segment) <= max_chars:
            final_segments.append(segment)
        else:
            # Diviser par virgules
            parts = re.split(r'[,;]+', segment)
            temp_segment = ""
            for part in parts:
                part = part.strip()
                if len(temp_segment + part) > max_chars and temp_segment:
                    final_segments.append(temp_segment.strip())
                    temp_segment = part
                else:
                    if temp_segment:
                        temp_segment += ", " + part
                    else:
                        temp_segment = part
            if temp_segment.strip():
                final_segments.append(temp_segment.strip())
    
    return [seg for seg in final_segments if seg.strip()]


def generate_audio_with_fallback(tts_model, text, output_path, speaker_wav, language="fr"):
    """
    Génère de l'audio avec gestion des erreurs et division automatique des textes longs
    """
    import gc
    
    # Vérifier la longueur du texte
    if len(text) > 273:
        print(f"📝 Texte trop long ({len(text)} caractères), division en segments...")
        
        segments = split_text_by_sentences(text, max_chars=250)
        print(f"🔢 Division en {len(segments)} segments")
        
        # Générer chaque segment
        segment_files = []
        for i, segment in enumerate(segments):
            segment_path = output_path.replace('.wav', f'_segment_{i+1}.wav')
            
            try:
                print(f"🎙️  Génération segment {i+1}/{len(segments)}: {segment[:50]}...")
                
                # Essayer plusieurs méthodes pour chaque segment
                success = False
                
                # Méthode 1: Standard
                try:
                    tts_model.tts_to_file(
                        text=segment,
                        file_path=segment_path,
                        speaker_wav=speaker_wav,
                        language=language,
                        split_sentences=False
                    )
                    success = True
                except Exception as e1:
                    print(f"⚠️  Méthode 1 échouée: {e1}")
                
                # Méthode 2: Recharger le modèle et appliquer le patch
                if not success:
                    try:
                        print("🔄 Rechargement du modèle avec patch...")
                        del tts_model
                        gc.collect()
                        
                        # Recharger
                        tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
                        tts_model.to(DEVICE)
                        
                        # Appliquer le patch GPT2
                        patch_gpt2_model(tts_model)
                        
                        tts_model.tts_to_file(
                            text=segment,
                            file_path=segment_path,
                            speaker_wav=speaker_wav,
                            language=language,
                            split_sentences=False,
                            # Paramètres pour une voix plus chaleureuse
                            temperature=0.75,  # Plus élevé = plus d'expression (0.1-1.5)
                            length_penalty=1.0,  # Contrôle le rythme
                            repetition_penalty=5.0,  # Évite les répétitions
                            top_k=50,  # Diversité du vocabulaire
                            top_p=0.85,  # Nucleus sampling pour plus de naturel
                        )
                        success = True
                    except Exception as e2:
                        print(f"⚠️  Méthode 2 échouée: {e2}")
                        
                        # Méthode 3: Downgrade temporaire de transformers
                        try:
                            print("🔧 Tentative de correction avec version compatible transformers...")
                            
                            # Forcer l'ajout de GenerationMixin
                            if hasattr(tts_model.synthesizer.tts_model, 'gpt'):
                                gpt_model = tts_model.synthesizer.tts_model.gpt
                                
                                # Méthode alternative: créer une méthode generate simple
                                def simple_generate(self, input_ids, **kwargs):
                                    # Utiliser forward ou __call__ comme fallback
                                    if hasattr(self, 'forward'):
                                        return self.forward(input_ids, **kwargs)
                                    else:
                                        return self(input_ids, **kwargs)
                                
                                # Attacher la méthode
                                import types
                                gpt_model.generate = types.MethodType(simple_generate, gpt_model)
                                
                                # Retry la génération
                                tts_model.tts_to_file(
                                    text=segment,
                                    file_path=segment_path,
                                    speaker_wav=speaker_wav,
                                    language=language,
                                    split_sentences=False
                                )
                                success = True
                                
                        except Exception as e3:
                            print(f"⚠️  Méthode 3 échouée: {e3}")
                
                if success:
                    segment_files.append(segment_path)
                    print(f"✅ Segment {i+1} généré")
                else:
                    print(f"❌ Échec du segment {i+1}")
                    return False, tts_model
                    
            except Exception as segment_error:
                print(f"❌ Erreur segment {i+1}: {segment_error}")
                return False, tts_model
        
        # Combiner les segments avec pydub
        try:
            from pydub import AudioSegment
            
            print("🔗 Combinaison des segments...")
            combined = AudioSegment.empty()
            
            for segment_file in segment_files:
                if os.path.exists(segment_file):
                    audio_segment = AudioSegment.from_wav(segment_file)
                    combined += audio_segment
                    combined += AudioSegment.silent(duration=300)  # 300ms de pause entre segments
                    
                    # Supprimer le fichier temporaire
                    os.remove(segment_file)
            
            # Sauvegarder le fichier final
            combined.export(output_path, format="wav")
            print("✅ Segments combinés avec succès")
            return True, tts_model
            
        except ImportError:
            print("❌ pydub non installé. Installation: pip install pydub")
            print("💡 Les segments restent séparés dans le dossier")
            return True, tts_model
        except Exception as combine_error:
            print(f"❌ Erreur lors de la combinaison: {combine_error}")
            print("💡 Les segments restent séparés dans le dossier")
            return True, tts_model
    
    else:
        # Texte court, génération directe
        try:
            tts_model.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=speaker_wav,
                language=language,
                split_sentences=False,
                # Paramètres pour une voix plus chaleureuse
                temperature=0.55,
                length_penalty=0.9,
                repetition_penalty=1.5,
                top_k=30,
                top_p=0.7,
            )
            return True, tts_model
            
        except AttributeError as attr_error:
            if "'GPT2InferenceModel' object has no attribute 'generate'" in str(attr_error):
                print("🔄 Correction GPT2 pour texte court...")
                try:
                    # Appliquer le patch GPT2
                    patch_gpt2_model(tts_model)
                    
                    # Retry
                    tts_model.tts_to_file(
                        text=text,
                        file_path=output_path,
                        speaker_wav=speaker_wav,
                        language=language,
                        split_sentences=False
                    )
                    return True, tts_model
                    
                except Exception as patch_error:
                    print(f"❌ Patch échoué: {patch_error}")
                    
                    # Méthode de fallback ultime: recharger complètement
                    try:
                        del tts_model
                        gc.collect()
                        
                        # Réinitialiser avec une approche différente
                        print("🔄 Réinitialisation complète du modèle...")
                        
                        # Forcer weights_only=False pour éviter d'autres problèmes
                        original_load = torch.load
                        torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
                        
                        tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
                        tts_model.to(DEVICE)
                        
                        # Restaurer torch.load
                        torch.load = original_load
                        
                        # Appliquer le patch immédiatement
                        patch_gpt2_model(tts_model)
                        
                        # Retry final
                        tts_model.tts_to_file(
                            text=text,
                            file_path=output_path,
                            speaker_wav=speaker_wav,
                            language=language,
                            split_sentences=False
                        )
                        return True, tts_model
                        
                    except Exception as final_reload_error:
                        print(f"❌ Échec final après rechargement: {final_reload_error}")
                        return False, tts_model
            else:
                raise attr_error
        except Exception as other_error:
            print(f"❌ Autre erreur: {other_error}")
            return False, tts_model


# Charger le modèle XTTS-v2
print("📥 Chargement du modèle XTTS-v2...")

def patch_gpt2_model(tts_model):
    """
    Patch le modèle GPT2 pour corriger le problème de generate
    """
    try:
        if hasattr(tts_model.synthesizer.tts_model, 'gpt'):
            gpt_model = tts_model.synthesizer.tts_model.gpt
            
            if not hasattr(gpt_model, 'generate'):
                print("🔧 Correction du modèle GPT2 individuel...")
                
                # Méthode 1: Ajouter GenerationMixin à la classe
                from transformers.generation.utils import GenerationMixin
                
                # Créer une nouvelle classe qui hérite de GenerationMixin
                original_class = gpt_model.__class__
                
                class PatchedGPTModel(original_class, GenerationMixin):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                
                # Remplacer la classe de l'instance
                gpt_model.__class__ = PatchedGPTModel
                
                # Forcer l'initialisation des attributs GenerationMixin si nécessaire
                if not hasattr(gpt_model, 'generation_config'):
                    from transformers import GenerationConfig
                    gpt_model.generation_config = GenerationConfig()
                
                # Vérifier que generate fonctionne maintenant
                if hasattr(gpt_model, 'generate'):
                    print("✅ Modèle GPT2 corrigé avec succès")
                    return True
                else:
                    # Méthode alternative: créer generate manuellement
                    def custom_generate(self, input_ids, **kwargs):
                        # Version simplifiée de generate
                        max_length = kwargs.get('max_length', 50)
                        current_length = input_ids.shape[1]
                        
                        for _ in range(max_length - current_length):
                            outputs = self(input_ids)
                            next_token_logits = outputs.logits[:, -1, :]
                            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                            input_ids = torch.cat([input_ids, next_token], dim=1)
                            
                            # Simple stopping criterion
                            if next_token.item() in [50256, 0]:  # EOS tokens
                                break
                                
                        return type('GenerateOutput', (), {'sequences': input_ids})()
                    
                    import types
                    gpt_model.generate = types.MethodType(custom_generate, gpt_model)
                    print("✅ Méthode generate personnalisée ajoutée")
                    return True
                    
            else:
                print("✅ Modèle GPT2 déjà fonctionnel")
                return True
                
    except Exception as patch_error:
        print(f"❌ Échec du patch GPT2: {patch_error}")
        
        # Méthode de dernier recours
        try:
            if hasattr(tts_model.synthesizer.tts_model, 'gpt'):
                gpt_model = tts_model.synthesizer.tts_model.gpt
                
                # Créer une méthode generate très simple
                def emergency_generate(self, input_ids, **kwargs):
                    # Juste faire un forward pass
                    outputs = self(input_ids)
                    return type('Output', (), {'sequences': input_ids, 'logits': outputs.logits})()
                
                import types
                gpt_model.generate = types.MethodType(emergency_generate, gpt_model)
                print("🚨 Méthode generate d'urgence appliquée")
                return True
        except:
            pass
        
        return False
    
    return False

try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True)
    tts.to(DEVICE)
    print("✅ Modèle chargé avec succès")
    
    # Appliquer le patch pour corriger le problème GPT2
    patch_success = patch_gpt2_model(tts)
    if not patch_success:
        print("⚠️  Le patch GPT2 a échoué, le modèle pourrait avoir des problèmes")
    
except Exception as e:
    print(f"❌ Erreur lors du chargement: {e}")
    print("💡 Tentative avec weights_only=False...")
    
    try:
        # Sauvegarder la fonction originale
        original_load = torch.load
        
        # Créer une fonction qui force weights_only=False
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        # Appliquer le patch temporairement
        torch.load = patched_load
        
        # Charger le modèle
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True)
        tts.to(DEVICE)
        
        # Restaurer la fonction originale
        torch.load = original_load
        
        print("✅ Modèle chargé avec succès (weights_only=False)")
        
    except Exception as final_error:
        print(f"❌ Échec final: {final_error}")
        print("💡 Solutions possibles:")
        print("   - pip install --upgrade TTS transformers torch")
        print("   - pip uninstall TTS && pip install TTS==0.22.0")
        print("   - Redémarrer le kernel/environnement Python")
        exit(1)

# --- Ajout du support CLI ---
def parse_args():
    parser = argparse.ArgumentParser(description="Génération audio XTTS batch")
    parser.add_argument('--excel', type=str, default="VoixOff/Voix Off.xlsx", help="Fichier Excel des voix off")
    parser.add_argument('--ref', type=str, default="Voices/MédhiCloneHigh.wav", help="Fichier wav de référence pour le clonage de voix")
    parser.add_argument('--out', type=str, default="Generated/coqui-xtts_v2", help="Dossier de sortie")
    return parser.parse_args()

args = parse_args()
EXCEL_PATH = args.excel
REFERENCE_WAV = args.ref
OUTPUT_DIR = args.out

# Paramètres
# EXCEL_PATH = "VoixOff/Voix Off.xlsx"
# REFERENCE_WAV = "Voices/MédhiCloneHigh.wav"
# OUTPUT_DIR = "Generated/coqui-xtts_v2"

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"📁 Dossier de sortie: {OUTPUT_DIR}")

# Vérifier que les fichiers existent
if not os.path.exists(EXCEL_PATH):
    print(f"❌ Fichier Excel non trouvé: {EXCEL_PATH}")
    exit(1)

if not os.path.exists(REFERENCE_WAV):
    print(f"❌ Fichier de référence vocal non trouvé: {REFERENCE_WAV}")
    exit(1)

try:
    # Charger le fichier Excel
    print("📊 Chargement du fichier Excel...")
    data = pd.read_excel(EXCEL_PATH)
    print(f"✅ {len(data)} lignes trouvées dans le fichier Excel")
    
    # Afficher les colonnes disponibles pour debug
    print(f"📋 Colonnes disponibles: {list(data.columns)}")
    
except Exception as e:
    print(f"❌ Erreur lors du chargement du fichier Excel: {e}")
    exit(1)

# Parcourir chaque ligne du fichier Excel
successful_generations = 0
failed_generations = 0

for index, row in tqdm(data.iterrows(), total=len(data), desc="Génération audio"):
    try:
        slide_number = row['Slide Number']
        slide_title = str(row['Slide Title']).replace('/', '_').replace('\\', '_')  # Nettoyer le titre pour le nom de fichier
        voice_text = row['Voice Over Text']
        
        # Vérifier que le texte n'est pas vide
        if pd.isna(voice_text) or str(voice_text).strip() == '':
            print(f"⚠️  Slide {slide_number}: Texte vide, passage à la suivante")
            continue
        
        # Générer le nom du fichier de sortie
        output_path = f"{OUTPUT_DIR}/{os.path.basename(EXCEL_PATH).replace('.xlsx', '')}_Slide{slide_number}_{slide_title}_{os.path.basename(EXCEL_PATH)}.wav"
        
        # Utiliser la fonction de génération robuste
        print(f"🎙️  Traitement Slide {slide_number}: {slide_title[:50]}...")
        print(f"📝 Longueur du texte: {len(str(voice_text))} caractères")
        
        success, tts = generate_audio_with_fallback(
            tts, 
            str(voice_text), 
            output_path, 
            REFERENCE_WAV, 
            "fr"
        )
        
        if success:
            print(f"✅ Fichier généré : {output_path}")
            successful_generations += 1
        else:
            print(f"❌ Échec de la génération pour Slide {slide_number}")
            failed_generations += 1
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération pour la ligne {index + 1}: {e}")
        failed_generations += 1
        continue

print(f"\n🎉 Génération terminée!")
print(f"✅ Succès: {successful_generations}")
print(f"❌ Échecs: {failed_generations}")
print(f"📁 Fichiers générés dans: {OUTPUT_DIR}")