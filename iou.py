import numpy as np

def iou(a, b):
    """
    Calculate intersect area between two bounding boxes
    @INPUT:
        - a: tupple, list or array contain (xmin, xmax, ymin, ymax)
        - b: tupple, list or array contain (xmin, xmax, ymin, ymax)
    @OUTPUT:
        - p: Percentage of overlap area between two bounding boxes
    """
    # Get bounding box position
    axmin, axmax, aymin, aymax = a[0], a[1], a[2], a[3]
    bxmin, bxmax, bymin, bymax = b[0], b[1], b[2], b[3]
    
    # Calculate area of each bounding box
    a_area = (axmax - axmin)*(aymax - aymin)
    b_area = (bxmax - bxmin)*(bymax - bymin)
    
    # Calculate overlap area
    dx = np.min((axmax, bxmax)) - np.max((axmin, bxmin))
    dy = np.min((aymax, bymax)) - np.max((aymin, bymin))
    
    if (dx <= 0) or (dy <= 0):
        return 0
    else:
        return (dx*dy)/(a_area + b_area - dx*dy)