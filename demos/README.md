
# Membership Inference Attacks in ART

When training a Machine Learning model, we often expect our training data to remain private.
Unfortunately, this is not always the case.
To uncover whether or not our models leak information, we might want to subject them to state-of-the art attacks.
**Membership Inference attacks are used to show potential information leakage in ML models.**
The goal of such attack is to infer whether or not a sample was used during training.
Attackers base their decision on the observed output of the model, given a target input. 

### Attacker capabilities
Attackers can be categorized along many axes. 
Here are some of the most important differentiating aspects:

#### Black box vs White box
In the white-box case, the attacker knows the underlying model architecture, while in the black-box case he does not.

#### Access to data
Typically, a Membership Inference attack requires the attacker to have access to a dataset, which he uses to launch the attack.
However, access to data is not necessary, as it may be synthetically generated, or the attack may not need it at all.

#### Access to correct label
It is possible that the attacker only has access to model inputs, without their correct label.

#### Membership information
The attacker may know the membership information about a *subset* of the training data.

### Attack categorization
Since the implemented membership inference attacks are all black-box attacks, we only categorize them based on _Access to data_, _Access to correct label_ and _Membership information_.

#### Newly implemented attacks

|                                                    | Access to data | Access to correct label | Membership information |
|:--------------------------------------------------:|:--------------:|:-----------------------:|:----------------------:|
| [ShadowModelAttack](https://nbviewer.jupyter.org/github/minaremeli/adversarial-robustness-toolbox/blob/main/demos/shadow_model_attack.ipynb) |       yes       |            yes           |           no           |
| [LabelOnlyDecisionBoundary](https://nbviewer.jupyter.org/github/minaremeli/adversarial-robustness-toolbox/blob/main/demos/label_only_membership_inference.ipynb) (unsupervised threshold) |       no       |            no           |           no           |
| [LabelOnlyTransferAttack](https://nbviewer.jupyter.org/github/minaremeli/adversarial-robustness-toolbox/blob/main/demos/label_only_transfer_attack.ipynb)                            |       yes      |            no           |           yes          |

#### Already existing attacks
|                                                    | Access to data | Access to correct label | Membership information |
|:--------------------------------------------------:|:--------------:|:-----------------------:|:----------------------:|
| [MembershipInferenceBlackBox](https://nbviewer.jupyter.org/github/minaremeli/adversarial-robustness-toolbox/blob/main/notebooks/attack_membership_inference.ipynb)                        |       yes      |           yes           |           yes          |
| [MembershipInferenceBlackBoxRuleBased](https://nbviewer.jupyter.org/github/minaremeli/adversarial-robustness-toolbox/blob/main/notebooks/attack_membership_inference.ipynb)               |       yes      |           yes           |           no           |
| [LabelOnlyDecisionBoundary](https://nbviewer.jupyter.org/github/minaremeli/adversarial-robustness-toolbox/blob/main/demos/label_only_membership_inference.ipynb) (supervised threshold)   |       yes      |           yes           |           yes          |

## Using ART on multiple frameworks
Any of the above mentioned attacks can be tried out on multiple different frameworks. ART supports many types of popular ML frameworks (TensorFlow, Keras, PyTorch, MXNet, scikit-learn, XGBoost, LightGBM, CatBoost, GPy, etc.). Typically, before using an attack, one must wrap their classifier in an ART wrapper class.

### Classifiers
Here we show how one can wrap their classifiers using ART through examples on multiple frameworks.
#### scikit-learn
```python
from sklearn.ensemble import RandomForestClassifier
from art.estimators.classification.scikitlearn import ScikitlearnRandomForestClassifier

model = RandomForestClassifier()
# wrap classifier in ART specific wrapper
art_classifier = ScikitlearnRandomForestClassifier(model)
```
ART also supports wrapping the following scikit-learn classifiers: `DecisionTreeClassifier`,  `ExtraTreeClassifier`, `AdaBoostClassifier`, `BaggingClassifier`, `ExtraTreesClassifier`, `GradientBoostingClassifier`, `GaussianNB`, `SVC`.
#### PyTorch
```python
from art.estimators.classification.pytorch import PyTorchClassifier

model = MyPyTorchClassifier() # any kind of PyTorch classifier
# wrap classifier in ART specific wrapper
art_classifier = PyTorchClassifier(
					model = model, 
					loss = ...,
					input_shape = ...,
					nb_classes = ..., # parameters after nb_classes are optional - some attacks will need them some won't
					optimizer = ...,	
					clip_values = ...,
					preprocessing = ...,
					...
				)
```
#### Keras
```python
from art.estimators.classification import KerasClassifier

model = MyKerasModel() # any kind of Keras classifier
# wrap classifier in ART specific wrapper
art_classifier = KerasClassifier(
					model = model, # parameters after model are optional - some attacks will need them some won't
					use_logits = ...,
					clip_values = ...,
					preprocessing = ...,
					...
				)
```
#### Tensorflow
##### TensorFlow v1
```python
import  tensorflow.compat.v1  as  tf
from  art.estimators.classification  import  TensorFlowClassifier

# define the model
input_ph = ... # input placeholder
logits = ... # define output

# wrap classifier in ART specific wrapper
art_classifier = TensorFlowClassifier(
					input_ph = input_ph,
					output = logits, # parameters after output are optional - some attacks will need them some won't
					labels_ph = ...,
					train = ...,
					loss = ...,
					sess = ...,
					...
				)
```
##### TensorFlow v2
```python
from art.estimators.classification import TensorFlowV2Classifier

model = MyKerasModel() # any kind of Keras classifier
# wrap classifier in ART specific wrapper
art_classifier = TensorFlowV2Classifier(
					model = model,
					nb_classes = ...,
					input_shape = ..., # parameters after input_shape are optional - some attacks will need them some won't
					loss_object = ...,
					clip_values = ...,
					preprocessing = ...,
					...
				)
```

#### XGBoost
```python
import xgboost as xgb
from art.estimators.classification import XGBoostClassifier

param = ...
train_data = ...
num_round = ...
model = xgb.train(param, train_data, num_round)
# wrap classifier in ART specific wrapper
 art_classifier = XGBoostClassifier(
					 model = model, # parameters after model are optional - some attacks will need them some won't
					 clip_values = ...,
					 nb_features = ...,
					 nb_classes = ...,
					 ...
				 )
```