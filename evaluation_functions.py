from pyemd import emd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
import numpy as np
from scipy.stats import wasserstein_distance

def chamfer_distance(point_cloud1, point_cloud2):
    """
    Berechnet die Chamfer-Distanz zwischen zwei Punktwolken. (normiert wegen np.mean)
    """
    dist_pc1_to_pc2 = np.min(np.linalg.norm(
        point_cloud1[:, np.newaxis, :] - point_cloud2[np.newaxis, :, :], axis=2), axis=1)
    dist_pc2_to_pc1 = np.min(np.linalg.norm(
        point_cloud2[:, np.newaxis, :] - point_cloud1[np.newaxis, :, :], axis=2), axis=1)
    chamfer_dist = np.mean(dist_pc1_to_pc2) + np.mean(dist_pc2_to_pc1)
    return chamfer_dist

def intersection_over_union(point_cloud1, point_cloud2, threshold=0.05):
    """
    Berechnet die Intersection over Union (IoU) für zwei Punktwolken.
    Funktioniert in unserem Fall nicht, weil wir sehr oft überhaupt keine Überschneidungen haben
    """
    # Berechne die Distanzen zwischen den Punkten beider Punktwolken
    distances = np.linalg.norm(point_cloud1[:, np.newaxis, :] - point_cloud2[np.newaxis, :, :], axis=2)
    
    # Intersection: Zähle, wie viele Punkte in point_cloud1 mit mindestens einem Punkt in point_cloud2 eine Distanz unter dem Schwellenwert haben
    intersection = 0
    for i in range(len(point_cloud1)):
        # Prüfe, ob es einen Punkt in point_cloud2 gibt, dessen Distanz zu point_cloud1[i] unter dem Schwellenwert liegt
        if np.min(distances[i]) < threshold:
            intersection += 1
    
    # Berechne die Union
    union = point_cloud1.shape[0] + point_cloud2.shape[0] - intersection
    
    # Berechne die IoU (Intersection over Union)
    iou = intersection / union if union > 0 else 0
    return iou

def earth_movers_distance(point_cloud1, point_cloud2):
    """
    Berechnet die Earth Mover's Distance (EMD) für zwei Punktwolken.
    """
    mass1 = np.ones(point_cloud1.shape[0])
    mass2 = np.ones(point_cloud2.shape[0])

    # Berechnung der Wasserstein-Distanz für jede Dimension (x, y, z)
    emd_x = wasserstein_distance(point_cloud1[:, 0], point_cloud2[:, 0], mass1, mass2)
    emd_y = wasserstein_distance(point_cloud1[:, 1], point_cloud2[:, 1], mass1, mass2)
    emd_z = wasserstein_distance(point_cloud1[:, 2], point_cloud2[:, 2], mass1, mass2)

    # Summieren der Distanzen über alle Dimensionen
    emd = emd_x + emd_y + emd_z
    return emd
def hausdorff_distance(point_cloud1, point_cloud2):
    """
    Berechnet die Hausdorff-Distanz zwischen zwei Punktwolken.
    """
    distances1 = cdist(point_cloud1, point_cloud2, 'euclidean')
    hausdorff_dist = max(np.max(np.min(distances1, axis=1)), np.max(np.min(distances1, axis=0)))
    return hausdorff_dist

def normalize_point_cloud(point_cloud):
    """
    Normalisiert eine Punktwolke auf den Bereich [-1, 1].
    """
    center = np.mean(point_cloud, axis=0)
    point_cloud -= center  
    scale = np.max(np.linalg.norm(point_cloud, axis=1))
    if scale > 0:
        point_cloud /= scale
    return point_cloud
