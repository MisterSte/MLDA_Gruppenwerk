import numpy as np

def triangle_multi_mask(corr_matrix, mask_threshold: float):
    """Create a triangular mask for correlation matrices with threshold filtering.

    :param corr_matrix: a df.corr() object (pandas DataFrame)
    :param mask_threshold: float threshold. 0 = no filter, 1 = full filter.
    :return: combined mask as numpy boolean array
    """

    triangle_mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    threshold_mask = np.abs(corr_matrix) <= mask_threshold
    multi_mask = np.logical_or(triangle_mask, threshold_mask)

    if not multi_mask.any().any():
        print("No values match filter criterion. Returning unfiltered triangle mask.")
        multi_mask = triangle_mask  # Fallback: nur Dreieck, ohne Filter

    return multi_mask