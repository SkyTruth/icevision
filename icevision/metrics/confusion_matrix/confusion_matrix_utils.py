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
    
    # Indices of prediction-target pairs exceeding IoU threshold
    pairs_indices = torch.nonzero(dice_table.ge(groundtruth_dice_thresh))
    
    # Indices where a prediction doesn't match any target (considered as false positives)
    false_positive_indices = torch.nonzero(torch.tensor([0 if any(row) else 1 for row in dice_table.ge(groundtruth_dice_thresh)]))

    # Form list of target details and their corresponding matched predictions
    target_list, prediction_list = [], []
    
    for bbox, label, label_id in zip(target.detection.bboxes, target.detection.labels, target.detection.label_ids):
        target_list.append([{'target_bbox': bbox, 'target_label': label, 'target_label_id': label_id}, []])

    for bbox, label_id, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
        prediction_list.append({'predicted_bbox': bbox, 'predicted_label_id': label_id, 'score': score})

    # Attach matched predictions to respective targets
    for pred_id, target_id in pairs_indices:
        matched_prediction = prediction_list[pred_id].copy()
        matched_prediction['dice_score'] = round(dice_table[pred_id, target_id].item(), 4)
        target_list[target_id][1].append(matched_prediction)

    return target_list, false_positive_indices
