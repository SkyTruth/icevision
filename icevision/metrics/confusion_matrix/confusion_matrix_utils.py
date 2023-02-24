from icevision.imports import *
from icevision import BBox, BaseRecord
from icevision.core.mask import RLE


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


def pairwise_iou_record_record(target: BaseRecord, prediction: BaseRecord):
    """
    Calculates pairwise iou on prediction and target BaseRecord. Uses torchvision implementation of `box_iou`.
    """
    stacked_preds = [bbox.to_tensor() for bbox in prediction.detection.bboxes]
    stacked_preds = torch.stack(stacked_preds) if stacked_preds else torch.empty(0, 4)

    stacked_targets = [bbox.to_tensor() for bbox in target.detection.bboxes]
    stacked_targets = torch.stack(stacked_targets) if stacked_targets else torch.empty(0, 4)
    return torchvision.ops.box_iou(stacked_preds, stacked_targets)

def mask_similarity(mask1: Tensor, mask2: Tensor) -> Tensor:
    """
    Return similarity index between two 2D pixel confidences masks.

    Args:
        mask1 (Tensor[X, Y]): first masks
        mask2 (Tensor[X, Y]): second masks

    Returns:
        float: the pairwise IoU values for the two masks
    """
    return 2 * torch.sum(torch.sqrt(torch.mul(mask1, mask2))) / (torch.sum(mask1 + mask2))


def pairwise_mask_similarity_record_record(record_1: BaseRecord, record_2: BaseRecord, img_size):
    """
    Calculates pairwise similarity between two BaseRecords. Uses custom implementation of `mask_iou`.
    """
    res = torch.zeros(len(record_2.detection.masks), len(record_1.detection.masks))
    for i, mask_2 in enumerate(record_2.detection.masks):
        for j, mask_1 in enumerate(record_1.detection.masks):
            mask_2 = mask_2.to_mask(*img_size).to_tensor() if type(mask_2)==RLE else mask_2
            mask_1 = mask_1.to_mask(*img_size).to_tensor() if type(mask_1)==RLE else mask_1

            res[i,j] = mask_similarity(torch.squeeze(mask_2).to(device="cuda"), torch.squeeze(mask_1).to(device="cuda"))
    return res # output is tensor of size [len(preds), len(targets)]

def prune_records_by_id(record: BaseRecord, remove_ids):
    record.detection.labels = [l for i, l in enumerate(record.detection.labels) if i not in remove_ids]
    record.detection.masks = [l for i, l in enumerate(record.detection.masks) if i not in remove_ids]
    record.detection.mask_array.data = [l for i, l in enumerate(record.detection.mask_array.data) if i not in remove_ids]
    record.detection.scores = [l for i, l in enumerate(record.detection.scores) if i not in remove_ids]
    record.detection.bboxes = [l for i, l in enumerate(record.detection.bboxes) if i not in remove_ids]
    return record


def apply_interclass_mask_nms(prediction: BaseRecord, nms_threshold: float = 0.5):
    iou_table = pairwise_mask_similarity_record_record(prediction, prediction, img_size=prediction.common.img_size)
    pairs_indices = torch.nonzero(iou_table > nms_threshold)
    remove_ids = [int(j) for i, j in pairs_indices if j>i]
    return prune_records_by_id(prediction, remove_ids)


def match_records(
    target: BaseRecord, prediction: BaseRecord, iou_threshold: float = 0.5, use_mask_similarity: bool = False
) -> Collection:
    """
    matches bboxes, labels from targets with their predictions by iou threshold
    """
    # here we get a tensor of indices that match iou criteria (order is (pred_id, target_id))
    if use_mask_similarity:
        iou_table = pairwise_mask_similarity_record_record(record_1=target, record_2=prediction, img_size=target.common.img_size)
    else:
        iou_table = pairwise_iou_record_record(target=target, prediction=prediction)
    pairs_indices = torch.nonzero(iou_table > iou_threshold)

    # if a prediction has no target ids, then it is background
    false_positive_indices = torch.nonzero(tensor([0 if any(row) else 1 for row in iou_table > iou_threshold ]))

    # creating a list of [target, matching_predictions]
    target_list = [
        [dict(target_bbox=bbox, target_label=label, target_label_id=label_id), []]
        for bbox, label, label_id in zip(
            target.detection.bboxes, target.detection.labels, target.detection.label_ids
        )
    ]
    prediction_list = [
        dict(
            predicted_bbox=bbox,
            predicted_label=label,
            predicted_label_id=label_id,
            score=score,
        )
        for bbox, label, label_id, score in zip(
            prediction.detection.bboxes,
            prediction.detection.labels,
            prediction.detection.label_ids,
            prediction.detection.scores,
        )
    ]

    # appending matches to targets
    for pred_id, target_id in pairs_indices:
        single_prediction = deepcopy(prediction_list[pred_id])
        # python value casting needs rounding cause otherwise there are 0.69999991 values
        iou_score = round(iou_table[pred_id, target_id].item(), 4)
        single_prediction["iou_score"] = iou_score
        # seems like a magic number, but we want to append to the list of target's matching_predictions
        target_list[target_id][1].append(single_prediction)

    return target_list, false_positive_indices
