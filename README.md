# DL-Life-Science
Modeling DNA Sequence Determinants of Regulatory Activity.

## Approach
Our approach is based on a combination of a **convolutional neural network** (local approach) and another model that analyses sequences,
such as a **transformer** (global approach), and on this base we separately predict the outputs:

* **is_active** : a binary label (0 or 1) indicating the presence or absence of regulatory activity for a given sequence.
* **rna_dna_ratio** : a continuous variable representing the experimentally determined ratio of RNA to DNA levels associated with the sequence, serving as a
quantitative measure of regulatory strength.
