
# Towards Fairness in Machine Learning with Adversarial Networks

## Overview

This project investigates how machine learning predictions can be made less dependent on sensitive demographic attributes.

Using the UCI Adult Census Income dataset, we build a classifier to predict whether a person's annual income exceeds $50,000. Race and sex are treated as sensitive attributes and are kept separate from the features used by the classifier.

We first train a standard income classifier and evaluate its predictions. We then introduce an adversarial neural network that attempts to infer sensitive attributes from the classifier's predictions. The classifier and adversary are trained competitively: the classifier aims to predict income accurately while making sensitive attributes harder to infer.

We compare the models using prediction performance, the p-rule, prediction distributions, and the adversary's ROC AUC. The final training results reported in the notebook show an adversary ROC AUC of 0.50 and p-rules above 80% for both sensitive attributes.

## Tools

Python · PyTorch · Scikit-learn · Pandas · NumPy · Matplotlib · Seaborn

## Reference

Louppe, G., Kagan, M., & Cranmer, K. (2017). [Learning to Pivot with Adversarial Networks](https://papers.nips.cc/paper/6699-learning-to-pivot-with-adversarial-networks).
