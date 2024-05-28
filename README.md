# Natural Language Processing Project
Group project for the NLP subject in our AI degree.

## Files in this repository
- `data`: contains the training and test data, as well as preprocessed data useful for our models.
- `latex`: necessary files used for the creation of the report in latex.
- `models`: last trained checkpoints of the CRF and LSTM models.
- `src`: contains all the python scripts and jupyter notebooks.
	- `preprocesing.py`: functions for data preprocessing, incuding tokenization and lemmatization.
	- `blindNegex.py`: code for the Full Scope NegEx, the baseline model.
	- `negex.py`: implementation of NegEx as proposed in the original paper, using medical terms extracted from the training set.
	- `crf0.py`: code for the first implementation of CRF.
	- `crf.py`: code for the second and official implementation of CRF.
	- `lstm.py`: code for the BiLSTM implementation.
	- `eval.py`: code to evaluate the model predictions
	- `.ipynb` files: python scripts used during the development of this project, most of them to test and demonstrate the funcitonality of their corresponding `.py` files.

## Contributors
- Amritpal Singh
- Oscar Arrocha
- Mustapha El Aichouni
- Eric López
