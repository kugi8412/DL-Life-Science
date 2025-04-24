# DL-Life-Science
 Modeling 🧬 Sequence Determinants of Regulatory Activity.

## Approach
Our approach combines local feature extraction using a convolutional neural network (CNN) with global modelling of long-range dependencies using a transformer. We separately perform prediction of classification and regression tasks on the resulting joint sequence representation.

* **is_active** : a binary label (0 or 1) indicating the presence or absence of regulatory activity for a given sequence.
* **rna_dna_ratio** : a continuous variable representing the experimentally determined ratio of RNA to DNA levels associated with the sequence, serving as a
quantitative measure of regulatory strength.

### Evaluation
If you have a model with the same architecture in file path_to_model, just call the script: 

`python evaluation_script.py <path_to_model> <path_to_test_data>`

path_to_test_data is a tsv file containing two columns:
* id (iunique sequence id)
* sequence (sequence length 271)

The results, on the other hand, will contain 3 columns:
* id (unique sequence id)
* cls (value 0, or 1 corresponding to the is_active column)
* reg (continuous value corresponding to the rna_dna_ratio column)
