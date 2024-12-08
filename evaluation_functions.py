from pyemd import emd
from scipy.spatial.distance import cdist
import numpy as np

def chamfer_distance(point_cloud1, point_cloud2):
    """
    Berechnet die Chamfer-Distanz zwischen zwei Punktwolken.
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
    """
    distances = np.linalg.norm(point_cloud1[:, np.newaxis, :] - point_cloud2[np.newaxis, :, :], axis=2)
    intersection = np.sum(distances < threshold)
    union = len(point_cloud1) + len(point_cloud2) - intersection
    iou = intersection / union if union > 0 else 0
    return iou

def earth_movers_distance(point_cloud1, point_cloud2):
    """
    Berechnet die Earth Mover's Distance (EMD) für zwei Punktwolken.
    """
    point_cloud1 = point_cloud1.reshape(-1, 3)
    point_cloud2 = point_cloud2.reshape(-1, 3)
    dist_matrix = np.linalg.norm(point_cloud1[:, np.newaxis, :] - point_cloud2[np.newaxis, :, :], axis=2)
    hist1 = np.ones(len(point_cloud1))  
    hist2 = np.ones(len(point_cloud2))
    emd_distance = emd(hist1, hist2, dist_matrix)
    return emd_distance

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
