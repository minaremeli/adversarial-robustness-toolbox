import logging
from typing import TYPE_CHECKING

import numpy as np

from typing import Union
from art.attacks.attack import InferenceAttack
from art.attacks.inference.membership_inference.black_box import MembershipInferenceBlackBox
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class LabelOnlyTransferAttack(InferenceAttack):
    attack_params = InferenceAttack.attack_params + [
        "classifier",
        "membership_inference"
    ]
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(self,
                 classifier: "CLASSIFIER_TYPE",
                 membership_inference: MembershipInferenceBlackBox,
                 ):
        super().__init__(estimator=classifier)
        self.membership_inference = membership_inference
        self.transferred_membership_inference = membership_inference

    def fit(self, x: np.ndarray, test_x: np.ndarray):
        # query labels from the classifier model
        train_labels = check_and_transform_label_format(np.argmax(self.estimator.predict(x=x), axis=1), self.estimator.nb_classes)
        test_labels = check_and_transform_label_format(np.argmax(self.estimator.predict(x=test_x), axis=1), self.estimator.nb_classes)
        labels = np.concatenate([train_labels, test_labels])
        data = np.concatenate([x, test_x])

        # use labels to train shadow model
        self.membership_inference.estimator.fit(
            data,
            labels,
        )

        # fit the attack model on shadow model
        self.membership_inference.fit(x, train_labels, test_x, test_labels)

    def transfer(self, transfer_model: Union["CLASSIFIER_TYPE"]):
        mia = MembershipInferenceBlackBox(
            classifier=transfer_model,
            input_type=self.membership_inference.input_type,
            attack_model=self.membership_inference.attack_model,
        )
        self.transferred_membership_inference = mia

    def no_transfer(self):
        self.transferred_membership_inference = self.membership_inference

    def infer(self, x: np.ndarray, **kwargs) -> np.ndarray:
        y = np.argmax(self.estimator.predict(x=x), axis=1)

        return self.transferred_membership_inference.infer(x, y)
