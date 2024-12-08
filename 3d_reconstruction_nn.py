#import pkg_resources
#installed_packages = [p.project_name for p in pkg_resources.working_set]
#print("tensorflow" in installed_packages)  # Should print True

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from pyemd import emd
from scipy.spatial.distance import cdist

from evaluation_functions import chamfer_distance, intersection_over_union, earth_movers_distance, hausdorff_distance, normalize_point_cloud

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


def load_3d_points(model_folder, object_folder):
    """
    Durchsuche alle Unterordner, bis ich die erste Datei mit dem Namen "3d_keypoints.txt" finde; die anderen lade ich nicht
    takes model folder and object_folder
    return array of img
    """
    object_model_folder = os.path.join(model_folder, object_folder)
    
    for subfolder in os.listdir(object_model_folder):
        subfolder_path = os.path.join(object_model_folder, subfolder)
        keypoints_file = os.path.join(subfolder_path, "3d_keypoints.txt")

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
    
    raise FileNotFoundError(f"3D keypoints file not found in any subfolder of {object_model_folder}")


def load_data(image_folder, model_folder):
    """ 
    Laden der Bilder und 3D Modelle.
    :takes image_folder and model_folder
    :return array of images and point clouds
    """
    image_paths = []  
    point_clouds = []

    # Durchlaufe alle Objekte (z. B. 'bed', 'chair', etc.)
    for object_folder in os.listdir(image_folder):
        object_image_folder = os.path.join(image_folder, object_folder)
        object_model_folder = os.path.join(model_folder, object_folder)

        if os.path.isdir(object_image_folder) and os.path.isdir(object_model_folder):
            point_cloud = load_3d_points(model_folder, object_folder)  # Laden der 3D-Punkte für das Objekt
            # Lade Bilder für das Objekt
            for img_name in os.listdir(object_image_folder):
                img_path = os.path.join(object_image_folder, img_name)
                if img_path.lower().endswith(('.jpg', '.png')):
                    image_paths.append(img_path)
                    point_clouds.append(point_cloud)

    X_train = np.array([load_image(img_path) for img_path in image_paths])
    y_train = np.array(point_clouds)
    return X_train, y_train

def create_model(input_shape, output_size):
    """
    Sequential-Modell wird erstellt:
    Eingabe: Bilddaten für ein RGB Bild
    Flatten-Layer: Wandelt das Bild in einen Vektor um
    Mehrere Dense-Layer extrahieren relevante Merkmale
    Ausgabe: Eine flache Liste von 3D-Koordinaten für die Punktwolke
    """
    model = Sequential([
        Input(shape = input_shape),
        Flatten(), # Flatten der Eingabebilder
        Dense(512, activation='relu'),  
        Dense(256, activation='relu'), 
        Dense(128, activation='relu'), 
        Dense(output_size, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    """
    Modell wird mit den Bilddaten als Eingabe und den 3D-Punktwolken als Ziel trainiert
    Optimizer: adam; mse-Loss, Metric: mae
    """
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)


def save_to_ply(points, output_file="output.ply"):
    """
    Speichert Punkte in einer PLY-Datei.
    :param points: Numpy-Array der Form (n, 3), wobei n die Anzahl der Punkte ist.
    :param output_file: Pfad zur Ausgabedatei.
    """
    with open(output_file, 'w') as f:
        # Header für PLY
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        # Punkte schreiben
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

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

def test_dif_predictions(new_image_path):
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

# Hauptblock zum Ausführen des Codes
if __name__ == "__main__":
    image_folder = "C:/Users/User/OneDrive - Universität Salzburg/Dokumente/Studium/DataScience/5. Semester/Imaging/Daten_Programm/img"  # Pfad zum Ordner mit den Bildern
    model_folder = "C:/Users/User/OneDrive - Universität Salzburg/Dokumente/Studium/DataScience/5. Semester/Imaging/Daten_Programm/model"  # Pfad zum Ordner mit den Modell-Dateien
    
    print("--- trainingsdaten erstellen")
    X_train, y_train = load_data(image_folder, model_folder)
    print("--- trainingsdaten erfolgreich erstellen")

    print(f"Input shape: {X_train.shape}")  # (Anzahl der Bilder, Bildhöhe, Bildbreite, Farbkanäle)
    print(f"Output shape: {y_train.shape}")  # (Anzahl der Bilder, 3D-Punkte * 3 Koordinaten)

    print("--- model trainieren")
    model = create_model(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), output_size=y_train.shape[1])

    model.summary()

    train_model(model, X_train, y_train, epochs=50, batch_size=128)
    print("--- model erfolgreich trainiert")
    print()
    
    # Speichere das Modell
    # model.save('3d_reconstruction_model_nn.keras')
    # print("--- Modell erfolgreich trainiert und gespeichert")

##### Neue Bilder #####
    # Der folgende Ordner enthält mehrer Subordner (einen/pro_objektart) mit jeweils einem Foto und einem 3D Modell
    new_image_path = "C:/Users/User/OneDrive - Universität Salzburg/Dokumente/Studium/DataScience/5. Semester/Imaging/Testdaten"
    test_dif_predictions(new_image_path)


    # save_to_ply(points, "predicted_model.ply")
    # Mit Trimesh anzeigen
    # point_cloud = trimesh.points.PointCloud(points)
    # point_cloud.show()
