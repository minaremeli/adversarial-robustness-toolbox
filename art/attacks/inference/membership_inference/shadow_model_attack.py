import logging
from typing import TYPE_CHECKING

import numpy as np
import copy

from typing import Union, Optional, Any
from art.attacks.inference.membership_inference.black_box import MembershipInferenceBlackBox
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class ShadowModelAttack(MembershipInferenceBlackBox):
    attack_params = MembershipInferenceBlackBox.attack_params + [
        "shadow_model",
        "nb_shadow_models",
        "shadow_dataset_size"
    ]
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(self,
                 classifier: Union["CLASSIFIER_TYPE"],
                 shadow_model: Union["CLASSIFIER_TYPE"],
                 nb_shadow_models: int = 1,
                 shadow_dataset_size: int = 1000,
                 input_type: str = "prediction",
                 attack_model_type: str = "nn",
                 attack_model: Optional[Any] = None,
                 ):
        super().__init__(classifier=classifier,
                         input_type=input_type,
                         attack_model_type=attack_model_type,
                         attack_model=attack_model
                         )
        self.nb_shadow_models = nb_shadow_models,
        self.shadow_dataset_size = shadow_dataset_size

        self.shadow_models = []
        for _ in range(nb_shadow_models):
            self.shadow_models.append(copy.deepcopy(shadow_model))

    def fit(  # pylint: disable=W0613
            self, x: np.ndarray, y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, **kwargs
    ):
        self._check_fit_params(x, y, test_x, test_y)

        predictions, labels, membership = [], [], []
        for i, shadow_model in enumerate(self.shadow_models):
            sh_x, sh_y, sh_test_x, sh_test_y = self._train_shadow_model(shadow_model, x, y, test_x, test_y, **kwargs)

            x_1, x_2, y_new = self._create_attack_dataset(self.estimator, sh_x, sh_y, sh_test_x, sh_test_y)
            predictions.extend(x_1)
            labels.extend(x_2)
            membership.extend(y_new)

        predictions, labels, membership = np.array(predictions), np.array(labels), np.array(membership)

        self._fit_attack_model(predictions, labels, membership)

    def _train_shadow_model(self, shadow_model: Union["CLASSIFIER_TYPE"], x: np.ndarray, y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, **kwargs):
        train_length = len(x)
        test_length = len(test_x)

        y = check_and_transform_label_format(y, self.estimator.nb_classes)
        test_y = check_and_transform_label_format(test_y, self.estimator.nb_classes)

        training_idx = np.random.choice(range(train_length), size=self.shadow_dataset_size)
        test_idx = np.random.choice(range(test_length), size=self.shadow_dataset_size)

        shadow_training_data = x[training_idx]
        shadow_training_labels = y[training_idx]
        shadow_test_data = test_x[test_idx]
        shadow_test_labels = test_y[test_idx]

        shadow_model.fit(
            shadow_training_data,
            shadow_training_labels,
            **kwargs
        )

        pred = shadow_model.predict(shadow_test_data)
        acc = np.sum(np.argmax(pred, axis=1) == np.argmax(shadow_test_labels, axis=1)) / self.shadow_dataset_size
        print("Accuracy of shadow model (on test set): %f" % acc)

        return shadow_training_data, shadow_training_labels, shadow_test_data, shadow_test_labels

    def _check_fit_params(self, x: np.ndarray, y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray):
        super()._check_fit_params(x, y, test_x, test_y)

        train_length = len(x)
        test_length = len(test_x)

        if train_length < self.shadow_dataset_size:
            raise ValueError(
                "Size of train dataset (%d) should be at least %d!" % (train_length, self.shadow_dataset_size))
        if test_length < self.shadow_dataset_size:
            raise ValueError(
                "Size of test dataset (%d) should be at least %d!" % (test_length, self.shadow_dataset_size))