import numpy as np
from iou import iou

def soft_nms(boxes, score, threshold=0.5):
    """
    Perform Non-Maximal Suppression to remove overlap predicted bounding box
    @INPUT:
        - boxes: predicted bouding box
        - score: score of each box
        - iou_threshold: threshold to decide if two boxes are one
    @OUTPUT:
        - D: suppressed bounding box
        - S: score for each box
    
    References:
    [1] Navaneeth Bodla, Brahat Singh, Rama Chellappa, Larry S. Davis: Improving Object Detection With One Line of Code
    """
    # Initialize output
    D = np.zeros(boxes.shape)
    S = np.zeros(score.shape)
    
    # Get number of boxes
    N = boxes.shape[0]
    
    # Soft-NMS
    for i in range(N):
        # Finding boxes with largest score
        index = np.argmax(score)
        
        # Add that box and score to output
        D[i,:] = boxes[index,:]
        S[i] = score[index]
        
        # Remove that box from boxes and score
        boxes = np.delete(boxes, index, axis=0)
        score = np.delete(score, index)
        
        # Re-calculate box score base on iou
        for j in range(boxes.shape[0]):
            iou_score = iou(D[i,:], boxes[j,:])
            score[j] *= 1-iou_score
    
    # Remove all box with score lower than threshold
    index = np.where(S < threshold)[0]
    D = np.delete(D, index, axis=0)
    S = np.delete(S, index)
    D = np.array(D, dtype=np.int)

    
    return D,S

def hard_nms(boxes, score, overlapThresh=0.8):
    """
    Hard Non-Maximal Suppression
    @INPUT:
        - boxes: predicted bounding box. An array with columns are: [xmin, xmax, ymin, ymax]
        - score: score of each box
        - overlapThreshold: Overlapping area threshold to decide whether or not two boxes are box for the same object
    @OUTPUT:
        - boxes: Suppressed bounding box
    
    References:
    [1] https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    x2 = boxes[:,1]
    y1 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]
            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)
        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)
    # return only the bounding boxes that were picked
    return boxes[pick], score[pick]
