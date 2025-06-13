import numpy as np

def triangle_multi_mask(corr_matrix, mask_threshold:float):
    """

    :param corr_matrix:
    :param mask_threshold:
    :return:
    """

    # np.triu = Triangle Upper. Also nur das obere Dreieck wird hier mit dem Ones_like Array gefüllt
    # dtype: macht 1 zu True und 0 zu False des Dreiecks
    triangle_mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Maske um nur Dreieck anzeigen zu lassen
    threshold_mask = np.abs(
        corr_matrix) < 0  # Diese Maske legt die Maske auf alles fest, das zutrifft. Also wenn Wert unter dem Threshold liegt

    multi_mask = np.logical_or(triangle_mask, threshold_mask)

    return multi_mask