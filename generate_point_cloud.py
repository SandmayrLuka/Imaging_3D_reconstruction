import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_point_cloud_from_mesh(file_path, num_points=100000):
    """
    Generiert eine Punktwolke aus einem 3D-Modell.

    Parameters:
        file_path (str): Pfad zur 3D-Modell-Datei (.glb, .obj, etc.).
        num_points (int): Anzahl der Punkte in der Punktwolke.

    Returns:
        np.ndarray: Punktwolke als Nx3-Array.
    """
    # Laden des 3D-Modells
    mesh_data = trimesh.load(file_path)
    
    # Prüfen, ob es sich um eine Scene handelt
    if isinstance(mesh_data, trimesh.Scene):
        # Alle Meshes in der Scene zusammenfügen
        combined_mesh = trimesh.util.concatenate(mesh_data.dump())
    else:
        combined_mesh = mesh_data  # Direkt verwenden, wenn es ein einzelnes Mesh ist
    
    # Abtasten der Oberfläche, um die Punktwolke zu erstellen
    points, _ = trimesh.sample.sample_surface(combined_mesh, num_points)
    
    return points

def visualize_point_cloud(points):
    """
    Visualisiert eine Punktwolke in 3D.
    
    Parameters:
        points (np.ndarray): Punktwolke als Nx3-Array.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=points[:, 2], cmap='viridis')
    ax.set_title("3D Punktwolke")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

def save_point_cloud_to_txt(points, output_file):
    """
    Speichert eine Punktwolke in eine .txt-Datei.

    Parameters:
        points (np.ndarray): Punktwolke als Nx3-Array.
        output_file (str): Pfad zur Ausgabedatei.
    """
    np.savetxt(output_file, points, fmt="%.6f", delimiter=" ", header="x y z", comments='')


file_path = "C:/Users/User/OneDrive - Universität Salzburg/Dokumente/Studium/DataScience/5. Semester/Imaging/Daten_Programm/model/bed/IKEA_FJELLSE_2/model.obj"
output_file = "point_cloud.txt"  # Name der Ausgabedatei
num_points = 10000  # Anzahl der Punkte in der Punktwolke

# Punktwolke generieren
#point_cloud = generate_point_cloud_from_mesh(file_path, num_points)

# Punktwolke visualisieren
#visualize_point_cloud(point_cloud)

# Punktwolke in Datei speichern
#save_point_cloud_to_txt(point_cloud, output_file)

#print(f"Punktwolke erfolgreich in '{output_file}' gespeichert.")
