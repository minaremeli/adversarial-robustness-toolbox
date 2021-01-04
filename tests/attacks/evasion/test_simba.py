# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
import pytest

from art.attacks.evasion.simba import SimBA
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import get_labels_np_array

from tests.utils import TestBase
from tests.utils import get_image_classifier_tf, get_image_classifier_kr, get_image_classifier_pt
from tests.attacks.utils import backend_test_classifier_type_check_fail

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 150
    n_test = 2
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.mark.skipMlFramework("mxnet", "scikitlearn")
@pytest.mark.parametrize("targeted", [False, True])
def test_mnist(fix_get_mnist_subset, image_dl_estimator, targeted):
    estimator, _ = image_dl_estimator(from_logits=False)
    
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

    x_test_original = x_test_mnist.copy()

    # set the targeted label
    if targeted:
        y_target = np.zeros(10)
        y_target[8] = 1.0

    df = SimBA(estimator, attack="dct", targeted=targeted)

    x_i = x_test_original[0][None, ...]
    if targeted:
        x_test_adv = df.generate(x_i, y=y_target.reshape(1, 10))
    else:
        x_test_adv = df.generate(x_i)

    for i in range(1, len(x_test_original)):
        x_i = x_test_original[i][None, ...]
        if targeted:
            tmp_x_test_adv = df.generate(x_i, y=y_target.reshape(1, 10))
            x_test_adv = np.concatenate([x_test_adv, tmp_x_test_adv])
        else:
            tmp_x_test_adv = df.generate(x_i)
            x_test_adv = np.concatenate([x_test_adv, tmp_x_test_adv])

    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, x_test_mnist, x_test_adv)
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, 0.0, x_test_adv)

    y_pred = get_labels_np_array(estimator.predict(x_test_adv))
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, y_test_mnist, y_pred)

    accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test_mnist, axis=1)) / x_test_mnist.shape[0]
    logger.info("Accuracy on adversarial examples: %.2f%%", (accuracy * 100))

    # Check that x_test has not been modified by attack and classifier
    np.testing.assert_array_almost_equal(float(np.max(np.abs(x_test_original - x_test_mnist))), 0, decimal=5)


def test_classifier_type_check_fail():
    backend_test_classifier_type_check_fail(SimBA, (BaseEstimator, ClassifierMixin))



