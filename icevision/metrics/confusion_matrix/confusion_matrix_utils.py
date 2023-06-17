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


def rle_to_tensor_mask(rle_mask, resize=None):
    mask_tile_size = int(np.sqrt(sum(rle_mask.counts)))
    if mask_tile_size**2 != sum(rle_mask.counts):
        # target tile is not square, and we're not sure of the original dimensions...
        raise NotImplementedError        
    mask_tensor = rle_mask.to_mask(mask_tile_size, mask_tile_size).to_tensor().squeeze()
    if resize and resize != mask_tensor.size():
        mask_tensor = F.interpolate(mask_tensor[None,None,...], size=resize, mode='nearest')[0,0,:]
    return mask_tensor
    

def pairwise_dice(target: BaseRecord, prediction: BaseRecord, soft: bool = False):
    """
    Calculates pairwise Dice scores between target and prediction BaseRecords. Uses bounding box Intersection-over-Union 
    for hard dice calculation and the custom `calculate_soft_dice_coefficient` function for soft dice calculation.

    Args:
        target (BaseRecord): The target BaseRecord, which contains the ground truth data.
        prediction (BaseRecord): The prediction BaseRecord, which contains the predicted data.
        soft (bool, optional): A flag to determine if soft dice calculation should be used. Defaults to False.

    Returns:
        torch.Tensor: A tensor of pairwise Dice scores. The tensor is of shape [len(predictions), len(targets)].
    """
    if not soft:  # hard dice calculation
        # Convert bounding boxes to tensors
        stacked_preds = [bbox.to_tensor() for bbox in prediction.detection.bboxes]
        stacked_preds = torch.stack(stacked_preds) if stacked_preds else torch.empty(0, 4)

        stacked_targets = [bbox.to_tensor() for bbox in target.detection.bboxes]
        stacked_targets = torch.stack(stacked_targets) if stacked_targets else torch.empty(0, 4)
        
        # Calculate Intersection-over-Union for each pair of prediction and target
        res = torchvision.ops.box_iou(stacked_preds, stacked_targets) 
    else:  # soft dice calculation
        # Initialize a zero tensor for results
        res = torch.zeros(len(prediction.detection.mask_array.data), len(target.detection.masks))
        
        # Loop through each mask in the prediction and target
        for i, pred_mask in enumerate(prediction.detection.masks):
            for j, target_mask in enumerate(target.detection.masks):
                # If the mask is in RLE format, convert it to a tensor mask
                pred_mask = rle_to_tensor_mask(pred_mask, prediction.common.img_size) if type(pred_mask)==RLE else pred_mask 
                target_mask = rle_to_tensor_mask(target_mask, target.common.img_size) if type(target_mask)==RLE else target_mask
                
                # Calculate the soft dice coefficient for the pair of masks
                res[i,j] = calculate_soft_dice_coefficient(torch.squeeze(pred_mask), torch.squeeze(target_mask))
    
    # Return the tensor of Dice scores
    return res


def calculate_soft_dice_coefficient(mask1: Tensor, mask2: Tensor) -> Tensor:
    """
    Calculates the Dice Coefficient between two 2D probability mask arrays.

    The Dice Coefficient is a statistical tool that measures the overlap 
    between two masks, often used in image segmentation tasks. 
    A Dice Coefficient of 1 represents perfect overlap.

    Args:
        mask1 (Tensor[X, Y]): First mask with pixel values between 0 and 1
        mask2 (Tensor[X, Y]): Second mask with pixel values between 0 and 1

    Returns:
        float: Dice Coefficient between the two masks
    """
    return 2 * torch.sum(torch.sqrt(torch.mul(mask1, mask2))) / (torch.sum(mask1 + mask2))


def prune_records_by_id(record: BaseRecord, remove_ids):
    record.detection.labels = [l for i, l in enumerate(record.detection.labels) if i not in remove_ids]
    record.detection.masks = [l for i, l in enumerate(record.detection.masks) if i not in remove_ids]
    record.detection.mask_array.data = [l for i, l in enumerate(record.detection.mask_array.data) if i not in remove_ids]
    record.detection.scores = [l for i, l in enumerate(record.detection.scores) if i not in remove_ids]
    record.detection.bboxes = [l for i, l in enumerate(record.detection.bboxes) if i not in remove_ids]
    return record


def apply_global_soft_dice_nms(prediction: BaseRecord, soft_dice_threshold: float = 0.5):
    masks = prediction.detection.masks # extract the predicted masks
    keep_masks = torch.tensor([True] * len(masks)) # create a tensor to keep track of which masks are still relevant (not suppressed)
    remove_ids = []
    
    for i, i_mask in enumerate(masks): # for each mask in the list of masks
        if keep_masks[i]: # if the current mask is not suppressed
            for j, j_mask in enumerate(masks[i+1:]): # for all the subsequent masks to be compared
                if keep_masks[i+j+1]: # if the compared mask is not already suppressed
                    i_mask.to_mask(*prediction.common.img_size).to_tensor() if type(i_mask)==RLE else i_mask
                    j_mask.to_mask(*prediction.common.img_size).to_tensor() if type(j_mask)==RLE else j_mask
                    keep_masks[i+j+1] = calculate_soft_dice_coefficient(torch.squeeze(i_mask), torch.squeeze(j_mask)) <= soft_dice_threshold # update the list of masks to suppress if the similarity is too high, or leave it relevant if the similarity is low
        else:
            remove_ids.append(i)
    
    return prune_records_by_id(prediction, remove_ids) # return the reduced prediction object


def match_records(target: BaseRecord, prediction: BaseRecord, dice_threshold: float = 0.5, use_soft_dice: bool = False) -> Tuple[List[Dict], torch.Tensor]:
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
    prediction : BaseRecord
        The prediction record containing bounding boxes, labels, and scores.
    dice_threshold : float, optional
        The IoU threshold for matching, by default 0.5.
    use_soft_dice : bool, optional
        Whether to compute the soft Dice coefficient for IoU calculation, by default False.
    
    Returns
    -------
    Tuple[List[Dict], torch.Tensor]
        A tuple containing a list of dictionaries for each target, where each dictionary contains the target and its matched 
        predictions, and a tensor of indices of any false positive predictions.
    """

    # Compute pairwise IoU
    iou_table = pairwise_dice(target=target, prediction=prediction, soft=use_soft_dice)
    
    # Indices of prediction-target pairs exceeding IoU threshold
    pairs_indices = torch.nonzero(iou_table.ge(dice_threshold))
    
    # Indices where a prediction doesn't match any target (considered as false positives)
    false_positive_indices = torch.nonzero(torch.tensor([0 if any(row) else 1 for row in iou_table.ge(dice_threshold)]))

    # Form list of target details and their corresponding matched predictions
    target_list, prediction_list = [], []
    
    for bbox, label, label_id in zip(target.detection.bboxes, target.detection.labels, target.detection.label_ids):
        target_list.append([{'target_bbox': bbox, 'target_label': label, 'target_label_id': label_id}, []])

    for bbox, label, label_id, score in zip(prediction.detection.bboxes, prediction.detection.labels, prediction.detection.label_ids, prediction.detection.scores):
        prediction_list.append({'predicted_bbox': bbox, 'predicted_label': label, 'predicted_label_id': label_id, 'score': score})

    # Attach matched predictions to respective targets
    for pred_id, target_id in pairs_indices:
        matched_prediction = prediction_list[pred_id].copy()
        matched_prediction['iou_score'] = round(iou_table[pred_id, target_id].item(), 4)
        target_list[target_id][1].append(matched_prediction)

    return target_list, false_positive_indices
