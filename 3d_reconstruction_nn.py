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


# Hauptblock zum Ausführen des Codes
if __name__ == "__main__":
    image_folder = "C:/Users/User/OneDrive - Universität Salzburg/Dokumente/Studium/DataScience/5. Semester/Imaging/Daten_Programm/img"  # Pfad zum Ordner mit den Bildern
    model_folder = "C:/Users/User/OneDrive - Universität Salzburg/Dokumente/Studium/DataScience/5. Semester/Imaging/Daten_Programm/model"  # Pfad zum Ordner mit den Modell-Dateien
    new_image_path = "C:/Users/User/OneDrive - Universität Salzburg/Dokumente/Studium/DataScience/5. Semester/Imaging/Daten_Programm/img/bed/0001.png"  

    print("--- trainingsdaten erstellen")
    X_train, y_train = load_data(image_folder, model_folder)
    print("--- trainingsdaten erfolgreich erstellen")

    print(f"Input shape: {X_train.shape}")  # (Anzahl der Bilder, Bildhöhe, Bildbreite, Farbkanäle)
    print(f"Output shape: {y_train.shape}")  # (Anzahl der Bilder, 3D-Punkte * 3 Koordinaten)

    print("--- model trainieren")
    model = create_model(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), output_size=y_train.shape[1])

    model.summary()

    train_model(model, X_train, y_train, epochs=50, batch_size=32)
    print("--- model erfolgreich trainiert")
    
    # Speichere das Modell
    model.save('3d_reconstruction_model_nn.keras')
    print("--- Model saved!")

    # Vorhersage eines neuen Bildes
    print("--- prediction new picture as a test")
    new_image = load_image(new_image_path, target_size=(64, 64))
    new_image = np.expand_dims(new_image, axis=0)

    # 3D-Punktwolke vorhersagen
    predicted_3d_points = model.predict(new_image)
    print("Predicted 3D Point Cloud:", predicted_3d_points)

    # Punkte extrahieren
    x, y, z = predicted_3d_points[0][::3], predicted_3d_points[0][1::3], predicted_3d_points[0][2::3]

    # 3D-Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_title("Predicted 3D Point Cloud")
    plt.show()
##############################################

    points = predicted_3d_points.reshape(-1, 3)
    save_to_ply(points, "predicted_model.ply")
    
    # Mit Trimesh anzeigen
    point_cloud = trimesh.points.PointCloud(points)
    point_cloud.show()

    

