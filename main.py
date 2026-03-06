import glob
import cv2
import numpy as np
import os

# Zielordner für Ausgabe
os.makedirs("output", exist_ok=True)

# k1=0.001   # Subtil sichtbar
# k1=0.005   # Deutlich fisheye
# k1=0.02  # Stark (wie GoPro)
def barrel_distortion(img, k1=0.02, k2=0.00008):
    """Barrel Distortion (Fisheye-Effekt)"""
    h, w = img.shape[:2]
    center = (w//2, h//2)
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Polarkoordinaten
    dx = map_x - center[0]
    dy = map_y - center[1]
    r = np.sqrt(dx*dx + dy*dy)
    r_max = np.sqrt(center[0]**2 + center[1]**2)
    
    # Barrel-Faktor (positiv = fisheye)
    factor = 1 + k1 * (r/r_max) + k2 * (r/r_max)**2
    map_x_new = center[0] + dx * factor
    map_y_new = center[1] + dy * factor
    
    return cv2.remap(img, map_x_new.astype(np.float32), map_y_new.astype(np.float32), cv2.INTER_LINEAR)

# Bilder aus img/ laden
# Bilder aus img/ laden (mehrere Formate)
extensions = ['*.jpeg', '*.jpg', '*.png']
all_files = []
for ext in extensions:
    all_files.extend(glob.glob(f"img/{ext}"))
    
for file in all_files:
    print(f"Lade: {file}")
    
    img = cv2.imread(file)
    if img is None:
        print(f"❌ Fehler bei {file}")
        continue
    
    print(f"✅ Shape: {img.shape}")

    # 1. Fisheye-Verzerrung
    img_distorted = barrel_distortion(img)
    
    # 2. Gauß-Unschärfe
    img_blur = cv2.GaussianBlur(img_distorted, (15, 15), 1.5)

    # 3. Monochromes Filmkorn
    rows, cols, _ = img_blur.shape
    noise_mono = np.random.normal(0, 16, (rows, cols, 1)).astype(np.uint8)
    noise_mono = cv2.cvtColor(noise_mono, cv2.COLOR_GRAY2BGR)
    img_noisy = cv2.addWeighted(img_blur, 0.95, noise_mono, 0.15, 0)

    # 4. Farbanpassung HSV (Film-Look)
    img_hsv = cv2.cvtColor(img_noisy, cv2.COLOR_BGR2HSV).astype(np.float32)
    img_hsv[:, :, 1] *= 0.75  # Sättigung reduzieren
    img_film = cv2.cvtColor(np.clip(img_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 5. Ausgabe
    base_name = os.path.splitext(os.path.basename(file))[0]
    output_path = os.path.abspath(f"output/film_{base_name}.jpeg")
    success = cv2.imwrite(output_path, img_film)
    
    if success:
        print(f"💾 Erfolgreich gespeichert: {output_path}")
    else:
        print(f"❌ Speichern fehlgeschlagen: {output_path}")
