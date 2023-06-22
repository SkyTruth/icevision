from icevision.imports import *
from icevision import BBox, BaseRecord
from icevision.core.mask import RLE
import torch.nn.functional as F


def get_best_score_item(prediction_items: Collection[Dict], background_class_id):
    # fill with dummy if list of prediction_items is empty
    dummy = dict(
        predicted_bbox=BBox.from_xyxy(0, 0, 0, 0),
        score=1.0,
        iou_score=1.0,
        predicted_label_id=background_class_id,
    )
    best_item = max(prediction_items, key=lambda x: x["score"], default=dummy)
    return best_item

    
def get_aligned_pairs(dice_table, groundtruth_dice_thresh):
    # Flatten the tensor and get the sorted indices
    descending_scores, descending_idxs = torch.sort(dice_table.view(-1), descending=True)
    rows, cols = divmod(descending_idxs.numpy(), dice_table.shape[1])
    ordered_idxs = torch.stack((torch.Tensor(rows), torch.Tensor(cols))).t()

    aligned_pairs = []
    for count, (pred_idx, gt_idx) in enumerate(ordered_idxs):
        # If this row or column has been recorded already, skip
        if any(pred_idx == pair[0] or gt_idx == pair[1] for pair in aligned_pairs):
            continue
        # If the scores are now below the threshold, stop
        if descending_scores[count] <= max(groundtruth_dice_thresh, 0):
            break
        # Record the pair (i, j) as aligned (no verdict on True/False Positive yet)
        aligned_pairs.append((int(pred_idx.item()), int(gt_idx.item())))
    return aligned_pairs


def match_records(target: BaseRecord, prediction, groundtruth_dice_thresh, groundtruth_dice_function) -> Tuple[List[Dict], torch.Tensor]:
    """
    Match bounding boxes and labels from the target with corresponding predictions using the Intersection-over-Union (IoU) threshold.
    
    This function computes the pairwise IoU between predictions and targets. It then forms pairs of target and prediction
    where the IoU exceeds the specified threshold. For each such pair, the function creates a dictionary comprising 
    bounding box, label, and score of the prediction, along with the IoU score. The function returns a list of such dictionaries 
    for each target, along with indices of any false positive predictions.
    
    Parameters
    ----------
    target : BaseRecord
        The ground truth record containing bounding boxes and labels.
    prediction
        The prediction dictionary containing bounding boxes, labels, and scores.
    groundtruth_dice_thresh : bool, optional
        The Dice threshold for matching
    groundtruth_dice_function : float, optional
        The dice function for matching
    
    Returns
    -------
    Tuple[List[Dict], torch.Tensor]
        A tuple containing a list of dictionaries for each target, where each dictionary contains the target and its matched 
        predictions, and a tensor of indices of any false positive predictions.
    """
    # Compute pairwise IoU
    dice_table = groundtruth_dice_function(target=target, prediction=prediction)

    aligned_pairs = get_aligned_pairs(dice_table, groundtruth_dice_thresh)
    aligned_preds = set([pair[0] for pair in aligned_pairs])
    aligned_gts = set([pair[1] for pair in aligned_pairs])

    FP_idxs = [i for i in range(dice_table.shape[0]) if i not in aligned_preds]
    FN_idxs = [i for i in range(dice_table.shape[1]) if i not in aligned_gts]

    return aligned_pairs, FP_idxs, FN_idxs    
