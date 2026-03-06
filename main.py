import glob
import cv2
import numpy as np
import os

# Zielordner für Ausgabe
os.makedirs("output", exist_ok=True)

# Bilder aus img/ laden (relativer Pfad)
img_path = "img/*.jpeg"  # Ihre Dateien sind .jpeg!
for file in glob.glob(img_path):
    print(f"Lade: {file}")
    
    img = cv2.imread(file)
    if img is None:
        print(f"❌ Fehler bei {file}")
        continue
    
    print(f"✅ Shape: {img.shape}")
    
    # Pipeline (wie vorher)...
    img_blur = cv2.GaussianBlur(img, (15, 15), 1.5)
    noise = np.random.normal(0, 15, img_blur.shape).astype(np.uint8)
    img_noisy = cv2.add(img_blur, noise)
    
    img_hsv = cv2.cvtColor(img_noisy, cv2.COLOR_BGR2HSV).astype(np.float32)
    img_hsv[:, :, 1] *= 0.75
    img_film = cv2.cvtColor(np.clip(img_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # ABSOLUTER Ausgabepfad
    base_name = os.path.splitext(os.path.basename(file))[0]
    output_path = os.path.abspath(f"output/film_{base_name}.jpeg")
    success = cv2.imwrite(output_path, img_film)
    
    if success:
        print(f"💾 Erfolgreich gespeichert: {output_path}")
    else:
        print(f"❌ Speichern fehlgeschlagen: {output_path}")
