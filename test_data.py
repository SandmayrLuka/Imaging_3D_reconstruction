import os
from tensorflow.keras.preprocessing import image
import numpy as np
from evaluation_functions import chamfer_distance, intersection_over_union, earth_movers_distance, hausdorff_distance, normalize_point_cloud

def load_3d_points_for_test(keypoints_file):
    if os.path.exists(keypoints_file):
            keypoints = np.loadtxt(keypoints_file)  # loads data from a text file; returns an array containing data from the text file

            expected_num_points = 50  # expected number of 3D points (e.g., 100) [can include in given variables]
            if keypoints.ndim == 1: # Falls die Daten flach geladen werden
                keypoints = keypoints.reshape(-1, 3) # Umformen zu (n, 3)
            
            if len(keypoints) < expected_num_points: # Pad with zeros if there are fewer points than expected
                padding = np.zeros((expected_num_points - len(keypoints), keypoints.shape[1]))
                keypoints = np.vstack([keypoints, padding])#
            elif len(keypoints) > expected_num_points: # Truncate if there are more points than expected
                keypoints = keypoints[:expected_num_points]
            
            return keypoints.flatten()  # Flatten des Arrays

def load_image(img_path, target_size=(64, 64)):
    """
    Bild in numpy-Array umwandeln und Normalisieren (Bildpixel zwischen 0 und 1)
    takes img_path (and target_size = (64,64))
    return array of img
    """
    img = image.load_img(img_path, target_size=target_size) 
    img_array = image.img_to_array(img)  
    img_array = img_array / 255.0  
    return img_array

def test_dif_predictions(new_image_path, model):
    for object_folder in os.listdir(new_image_path):
        object_path = os.path.join(new_image_path, object_folder)
        if os.path.isdir(object_path):
            print(f"--- Processing object: {object_folder} ---")
            
            # Bildpfad und Modellpfad suchen
            image_path = None
            model_path = None
            
            for item in os.listdir(object_path):
                if item.lower().endswith(('.jpg', '.png')):
                    image_path = os.path.join(object_path, item)
                elif item == "3d_keypoints.txt":
                    model_path = os.path.join(object_path, item)
            
            if image_path and model_path:
                # Neues Bild laden
                new_image = load_image(image_path, target_size=(64, 64))
                new_image = np.expand_dims(new_image, axis=0)
                
                # Vorhersage der 3D-Punkte
                predicted_3d_points = model.predict(new_image, verbose=0)
                predicted_3d_points = predicted_3d_points.reshape(-1, 3)

                # Ground-Truth laden
                ground_truth_3d_points = load_3d_points_for_test(model_path)
                ground_truth_3d_points = ground_truth_3d_points.reshape(-1, 3)

                # Normalisieren
                predicted_3d_points = normalize_point_cloud(predicted_3d_points)
                ground_truth_3d_points = normalize_point_cloud(ground_truth_3d_points)

                # Metriken berechnen
                cd = chamfer_distance(predicted_3d_points, ground_truth_3d_points)
                iou = intersection_over_union(predicted_3d_points, ground_truth_3d_points)
                emd = earth_movers_distance(predicted_3d_points, ground_truth_3d_points)
                hausdorff = hausdorff_distance(predicted_3d_points, ground_truth_3d_points)

                # Ausgabe der Ergebnisse
                print(f"Results for {object_folder}:")
                print(f"  Chamfer Distance: {cd}")
                print(f"  IoU: {iou}")
                print(f"  Earth Mover's Distance: {emd}")
                print(f"  Hausdorff Distance: {hausdorff}")
                print()

                # Optional: 3D-Plot
                #fig = plt.figure()
                #ax = fig.add_subplot(111, projection='3d')
                #ax.scatter(predicted_3d_points[:, 0], predicted_3d_points[:, 1], predicted_3d_points[:, 2], label='Predicted')
                #ax.scatter(ground_truth_3d_points[:, 0], ground_truth_3d_points[:, 1], ground_truth_3d_points[:, 2], label='Ground Truth', alpha=0.6)
                #ax.set_title(f"3D Comparison: {object_folder}")
                #ax.legend()
                #plt.show()
            else:
                print(f"  Missing image or 3D model for {object_folder}.")

def test_dif_predictions_with_mean(new_image_path, model):
    """
    Berechnet die Metriken für alle Bilder in einem Ordner basierend auf einer gemeinsamen `keypoints.txt`-Datei.
    Gibt den Mittelwert der Metriken aus.
    """
    # Pfad zur Keypoints-Datei
    keypoints_path = os.path.join(new_image_path, "3d_keypoints.txt")
    if not os.path.exists(keypoints_path):
        print("Error: 3d_keypoints.txt not found in the folder.")
        return

    # Lade und normalisiere die Ground-Truth-Punkte
    ground_truth_3d_points = load_3d_points_for_test(keypoints_path)
    ground_truth_3d_points = ground_truth_3d_points.reshape(-1, 3)
    ground_truth_3d_points = normalize_point_cloud(ground_truth_3d_points)

    # Vorbereitung für die Metriken
    metrics = {
        "Chamfer Distance": [],
        "IoU": [],
        "Earth Mover Distance": [],
        "Hausdorff Distance": []
    }

    # Schleife durch alle Bilder im Ordner
    for item in os.listdir(new_image_path):
        if item.lower().endswith(('.jpg', '.png')):
            image_path = os.path.join(new_image_path, item)
            
            # Bild laden und vorbereiten
            new_image = load_image(image_path, target_size=(64, 64))
            new_image = np.expand_dims(new_image, axis=0)
            
            # Vorhersage der 3D-Punkte
            predicted_3d_points = model.predict(new_image, verbose=0)
            predicted_3d_points = predicted_3d_points.reshape(-1, 3)
            predicted_3d_points = normalize_point_cloud(predicted_3d_points)
            
            # Berechne die Metriken
            cd = chamfer_distance(predicted_3d_points, ground_truth_3d_points)
            iou = intersection_over_union(predicted_3d_points, ground_truth_3d_points)
            emd = earth_movers_distance(predicted_3d_points, ground_truth_3d_points)
            hausdorff = hausdorff_distance(predicted_3d_points, ground_truth_3d_points)
            
            # Speichere die Ergebnisse
            metrics["Chamfer Distance"].append(cd)
            metrics["IoU"].append(iou)
            metrics["Earth Mover Distance"].append(emd)
            metrics["Hausdorff Distance"].append(hausdorff)

    # Mittelwerte berechnen und ausgeben
    print("\n--- Final Mean Metrics ---")
    if metrics["Chamfer Distance"]:
        print(f"  Mean Chamfer Distance: {np.mean(metrics['Chamfer Distance']):.4f}")
        print(f"  Mean IoU: {np.mean(metrics['IoU']):.4f}")
        print(f"  Mean Earth Mover's Distance: {np.mean(metrics['Earth Mover Distance']):.4f}")
        print(f"  Mean Hausdorff Distance: {np.mean(metrics['Hausdorff Distance']):.4f}")
    else:
        print("  No valid data found.")
