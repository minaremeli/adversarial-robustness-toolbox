from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, TYPE_CHECKING

import numpy as np

from art.attacks.attack import InferenceAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class MembershipInferenceBlackBoxLossBased(InferenceAttack):

    attack_params = InferenceAttack.attack_params
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(self, classifier: "CLASSIFIER_TYPE"):
        super().__init__(estimator=classifier)
        self.mean_loss = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray):
        train_y = check_and_transform_label_format(train_y, len(np.unique(train_y)), return_one_hot=True)

        self.mean_loss = self.estimator.compute_loss(train_x, train_y).astype(np.float32).mean()

    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Infer membership in the training set of the target estimator.

        :param x: Input records to attack.
        :param y: True labels for `x`.
        :return: An array holding the inferred membership status, 1 indicates a member and 0 indicates non-member.
        """
        if y is None:
            raise ValueError("MembershipInferenceBlackBoxRuleBased requires true labels `y`.")

        if self.estimator.input_shape is not None:
            if self.estimator.input_shape[0] != x.shape[1]:
                raise ValueError("Shape of x does not match input_shape of classifier")

        y = check_and_transform_label_format(y, len(np.unique(y)), return_one_hot=True)
        if y.shape[0] != x.shape[0]:
            raise ValueError("Number of rows in x and y do not match")

        if self.mean_loss is None:
            raise ValueError("No mean_loss was calculated, call fit() to set this value!")

        loss = self.estimator.compute_loss(x, y).astype(np.float32).reshape(-1, 1)

        return (loss < self.mean_loss).astype(np.int)

