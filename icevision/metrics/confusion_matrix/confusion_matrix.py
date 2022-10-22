__all__ = ["SimpleConfusionMatrix"]

from icevision.data.prediction import Prediction
from icevision.metrics.metric import Metric
from icevision.imports import *
from icevision.metrics.confusion_matrix.confusion_matrix_utils import *
import PIL
import matplotlib.pyplot as plt
import numpy as np


class MatchingPolicy(Enum):
    BEST_SCORE = 1
    BEST_IOU = 2


class SimpleConfusionMatrix(Metric):
    def __init__(
        self,
        iou_threshold: float = 0.5,
        policy: MatchingPolicy = MatchingPolicy.BEST_SCORE,
        print_summary: bool = False,
        class_list = [],
    ):
        super(SimpleConfusionMatrix, self).__init__()
        self.print_summary = print_summary
        self.target_labels = []
        self.predicted_labels = []
        self._iou_threshold = iou_threshold
        self._policy = policy
        self.class_map = None
        self.confusion_matrix: sklearn.metrics.confusion_matrix = None
        self.class_list = class_list

    def _reset(self):
        self.target_labels = []
        self.predicted_labels = []

    def accumulate(self, preds: Collection[Prediction], background_class_id=0):
        for pred in preds:
            target_record = pred.ground_truth
            prediction_record = pred.pred
            self.class_map = target_record.detection.class_map
            # create matches based on iou
            matches, false_positive_indices = match_records(
                target=target_record,
                prediction=prediction_record,
                iou_threshold=self._iou_threshold,
            )

            target_labels, predicted_labels = [], []
            # iterate over multiple targets and preds in a record
            # assumes no overlap with multiple preds and single target
            # but mrcnn seems to handle this well with bbox confidence.
            # however bbox conf needs to be set high
            for target_item, prediction_items in matches:
                if self._policy == MatchingPolicy.BEST_SCORE:
                    predicted_item = get_best_score_item( # False negatives are captured/generated in this function
                        prediction_items=prediction_items,
                        background_class_id=background_class_id,
                    )
                elif self._policy == MatchingPolicy.BEST_IOU:
                    raise NotImplementedError
                else:
                    raise RuntimeError(f"policy must be one of {list(MatchingPolicy)}")

                # using label_id because negative examples have ids but not labels
                target_label = target_item["target_label_id"]
                predicted_label = predicted_item["predicted_label_id"]
                target_labels.append(target_label)
                predicted_labels.append(predicted_label)
            # we need to account for false preds on background class, i.e. false positives
            for idx in false_positive_indices:
                target_labels.append(background_class_id)
                predicted_labels.append(prediction_record.detection.label_ids[idx])

            # We need to store the entire list of gts/preds to support various CM logging methods
            assert len(predicted_labels) == len(target_labels)
            self.target_labels.extend(target_labels)
            self.predicted_labels.extend(predicted_labels)

    def finalize(self):
        """Convert preds to numpy arrays and calculate the CM"""
        assert len(self.target_labels) == len(self.predicted_labels)
        label_ids = np.arange(len(self.class_list))
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
        p = {f"{category} Prec": np.round(p[i], 3) for i, category in enumerate(self.class_list)}
        r = {f"{category} Recall": np.round(r[i], 3) for i, category in enumerate(self.class_list)}
        f1 = {f"{category} F1": np.round(f1[i], 3) for i, category in enumerate(self.class_list)}
        if self.print_summary:
            print(self.confusion_matrix)
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
