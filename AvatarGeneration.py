#!/usr/bin/env python3
"""
SadTalker - VERSION ENCORE PLUS SIMPLE avec Gradio Client
Utilise directement le Space HuggingFace
"""

from gradio_client import Client
import os

def ultra_simple_sadtalker(photo_path, audio_path, output_path="result.mp4"):
    """
    Version ultra-simple avec Gradio Client
    Pas besoin de token HF !
    """
    print("🎭 SadTalker Ultra-Simple")
    print("=" * 40)
    
    # Vérifications
    if not os.path.exists(photo_path):
        print(f"❌ Photo non trouvée: {photo_path}")
        return False
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio non trouvé: {audio_path}")
        return False
    
    try:
        # Connexion au Space HuggingFace
        print("🔗 Connexion à SadTalker...")
        client = Client("vinthony/SadTalker")
        
        # Génération (super simple !)
        print("🚀 Génération en cours...")
        result = client.predict(
            source_image=photo_path,     # Votre photo
            driven_audio=audio_path,     # Votre audio
            preprocess="crop",           # Preprocessing
            still=False,                 # Mouvements de tête
            use_enhancer=False,          # Pas d'amélioration (plus rapide)
            batch_size=2,                # Taille du batch
            size_of_image=256,           # Résolution
            pose_style=0,                # Style de pose
            facerender="facevid2vid",    # Renderer
            exp_scale=1.0,               # Échelle expressions
            use_ref_video=False,         # Pas de vidéo de référence
            ref_video=None,              # Vidéo de référence
            ref_info="pose",             # Info de référence
            use_idle_mode=False,         # Mode idle
            length_of_audio=0,           # Longueur audio auto
            api_name="/test"             # Point d'API
        )
        
        print(f"✅ Résultat reçu: {result}")
        
        # Le résultat est normalement un chemin vers la vidéo générée
        if result and os.path.exists(result):
            # Copier vers le nom souhaité
            import shutil
            shutil.copy2(result, output_path)
            print(f"✅ Vidéo sauvée: {output_path}")
            return True
        else:
            print("❌ Pas de vidéo générée")
            return False
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("💡 Vérifiez votre connexion internet")
        return False

def main():
    """Utilisation ultra-simple"""
    
    # 🎯 CHANGEZ CES CHEMINS avec vos vrais fichiers
    photo = "moi.jpg"           # Votre photo
    audio = "presentation.wav"  # Votre audio
    output = "mon_avatar.mp4"   # Vidéo finale
    
    print("🎬 Création avatar pour Teams")
    
    # Une seule ligne pour tout faire !
    success = ultra_simple_sadtalker(photo, audio, output)
    
    if success:
        print(f"\n🎉 TERMINÉ ! Votre avatar Teams: {output}")
    else:
        print("\n❌ Échec. Vérifiez vos fichiers.")

if __name__ == "__main__":
    main()

# Installation requise:
# pip install gradio-client
#
# Usage:
# 1. Mettez votre photo dans le même dossier (ex: moi.jpg)
# 2. Mettez votre audio dans le même dossier (ex: audio.wav)
# 3. Lancez: python script.py
# 4. Récupérez votre vidéo avatar !