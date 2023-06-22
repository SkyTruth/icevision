__all__ = ["SimpleConfusionMatrix"]

from icevision.data.prediction import Prediction
from icevision.metrics.metric import Metric
from icevision.imports import *
from icevision.metrics.confusion_matrix.confusion_matrix_utils import *
import PIL
import matplotlib.pyplot as plt
import numpy as np


class SimpleConfusionMatrix(Metric):
    def __init__(
        self,
        groundtruth_dice_thresh: float = 0.5,
        groundtruth_dice_function = None,
        print_summary: bool = True,
        background_class_id: int = 0,
    ):
        Metric.__init__(self)
        self._groundtruth_dice_thresh = groundtruth_dice_thresh
        self._groundtruth_dice_function = groundtruth_dice_function
        self.print_summary = print_summary
        self._background_class_id = background_class_id
        self.target_labels = []
        self.predicted_labels = []
        self.class_map = None
        self.confusion_matrix: sklearn.metrics.confusion_matrix = None

    def _reset(self):
        self.target_labels = []
        self.predicted_labels = []

    def accumulate(self, ds, preds):
        for target, pred in zip(ds, preds):
            self.class_map = target.detection.class_map
            # create matches based on iou

            aligned_pairs, FP_idxs, FN_idxs = match_records(
                target=target,
                prediction=pred,
                groundtruth_dice_thresh=self._groundtruth_dice_thresh,
                groundtruth_dice_function = self._groundtruth_dice_function,
            )

            target_labels, predicted_labels = [], []
            for pred_idx, gt_idx in aligned_pairs:
                predicted_labels.append(int(pred["labels"][pred_idx]))
                target_labels.append(target.detection.label_ids[gt_idx])
                
            for pred_idx in FP_idxs:
                predicted_labels.append(int(pred["labels"][pred_idx]))
                target_labels.append(self._background_class_id)
                
            for gt_idx in FN_idxs:
                predicted_labels.append(self._background_class_id)
                target_labels.append(target.detection.label_ids[gt_idx])

            assert len(predicted_labels) == len(target_labels)
            self.target_labels.extend(target_labels)
            self.predicted_labels.extend(predicted_labels)

    def finalize(self):
        """Convert preds to numpy arrays and calculate the CM"""
        assert len(self.target_labels) == len(self.predicted_labels)
        label_ids = np.arange(self.class_map.num_classes)
        self.confusion_matrix = sklearn.metrics.confusion_matrix(
            y_true=self.target_labels,
            y_pred=self.predicted_labels,
            labels=label_ids,
        )
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support
        p, r, f1, _ = sklearn.metrics.precision_recall_fscore_support(
            y_true=self.target_labels,
            y_pred=self.predicted_labels,
            labels=label_ids,
        )
        # default is no average so p r a nd f1 are initially arrays with vals for each class
        p = {f"{category} Prec": np.round(p[i], 3) for i, category in enumerate(self.class_map.get_classes()) if i != self._background_class_id}
        r = {f"{category} Recall": np.round(r[i], 3) for i, category in enumerate(self.class_map.get_classes()) if i != self._background_class_id}
        f1 = {f"{category} F1": np.round(f1[i], 3) for i, category in enumerate(self.class_map.get_classes()) if i != self._background_class_id}
        if self.print_summary:
            print(self.confusion_matrix)
            print(f1)
            print("Instance Macro-F1 (-background):", np.round(sum(f1.values())/len(f1),3))
        self._reset()
        return {"dummy_value_for_fastai": [f1]}

    def plot(
        self,
        normalize: Optional[str] = None,
        xticks_rotation="vertical",
        values_format: str = None,
        values_size: int = 12,
        cmap: str = "PuBu",
        figsize: int = 11,
        **display_args,
    ):
        """
        A handle to plot the matrix in a jupyter notebook, potentially this could also be passed to save_fig
        """
        if normalize not in ["true", "pred", "all", None]:
            raise ValueError("normalize must be one of {'true', 'pred', " "'all', None}")
        # properly display ints and floats
        if values_format is not None:
            values_format = ".2f" if normalize else "d"

        cm = self._maybe_normalize(self.confusion_matrix, normalize)
        labels_named = self.class_map._id2class
        cm_display = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=labels_named)
        cm_display_plot = cm_display.plot(
            xticks_rotation=xticks_rotation,
            cmap=cmap,
            values_format=values_format,
            **display_args,
        )
        # Labels, title and ticks
        label_font = {"size": "20"}  # Adjust to fit
        cm_display_plot.ax_.set_xlabel("Predicted labels", fontdict=label_font)
        cm_display_plot.ax_.set_ylabel("Observed labels", fontdict=label_font)
        title_font = {"size": "28"}  # Adjust to fit
        cm_display_plot.ax_.set_title("Confusion Matrix", fontdict=title_font)
        cm_display_plot.ax_.tick_params(axis="both", which="major", labelsize=18)  # Adjust to fit
        for labels in cm_display_plot.text_.ravel():
            labels.set_fontsize(values_size)
        figure = cm_display_plot.figure_
        figure.set_size_inches(figsize, figsize)
        figure.tight_layout()
        plt.close()
        return figure

    def _fig2img(self, fig):
        # TODO: start using fi2img from icevision utils
        """Converts matplotlib figure object to PIL Image for easier logging. Writing to buffer is necessary
        to avoid wandb cutting our labels off. Wandb autoconvert doesn't pass the `bbox_inches` parameter so we need
        to do this manually."""
        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches="tight")
        buf.seek(0)
        return PIL.Image.open(buf)

    def _maybe_normalize(self, cm, normalize):
        """This method is copied from sklearn. Only used in plot_confusion_matrix but we want to be able
        to normalize upon plotting."""
        with np.errstate(all="ignore"):
            if normalize == "true":
                cm = cm / cm.sum(axis=1, keepdims=True)
            elif normalize == "pred":
                cm = cm / cm.sum(axis=0, keepdims=True)
            elif normalize == "all":
                cm = cm / cm.sum()
        cm = np.nan_to_num(cm)
        return cm

    def log(self, logger_object) -> None:
        # TODO: Disabled for now, need to design for metric logging for this to work + pl dependency
        # if isinstance(logger_object, pl_loggers.WandbLogger):
        #     fig = self.plot()
        #     image = self._fig2img(fig)
        #     logger_object.experiment.log({"Confusion Matrix": wandb.Image(image)})
        return