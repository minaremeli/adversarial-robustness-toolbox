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

|                                                    | Access to data | Access to correct label | Membership information |
|:--------------------------------------------------:|:--------------:|:-----------------------:|:----------------------:|
| MembershipInferenceBlackBox                        |       yes      |           yes           |           yes          |
| MembershipInferenceBlackBoxRuleBased               |       yes      |           yes           |           no           |
| [LabelOnlyDecisionBoundary](label_only_membership_inference.ipynb) (supervised threshold)   |       yes      |           yes           |           yes          |
| [LabelOnlyDecisionBoundary](label_only_membership_inference.ipynb) (unsupervised threshold) |       no       |            no           |           no           |
| [LabelOnlyTransferAttack](label_only_transfer_attack.ipynb)                            |       yes      |            no           |           yes          |