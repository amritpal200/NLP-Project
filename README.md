# Natural Language Processing Project
Group project for the NLP subject in our AI degree.

## Files in this repository
- `config`: contains global files required by programs, such as config.json with global variables.
- `data`: contains the training and test data, as well as preprocessed data useful for our models.
- `src`: contains all the python scripts and jupyter notebooks.
	- `blindNegex.py`: code for the Full Scope NegEx, the baseline model.
	- `preprocesing.py`: functions for data preprocessing, incuding tokenization and lemmatization.
	- `cutext.py`: function definitions to use CUTEXT.
	- `cutextNegex.py`: implementation of NegEx as proposed in the original paper, using medical terms provided by CUTEXT.
	- `eval.py`: code to evaluate the model predictions
	- `.ipynb` files: python scripts used during the development of this project, to test and demonstrate the funcitonality of their corresponding `.py` files.

## Requirements
- CUTEXT:
	- installation instructions [here](https://github.com/PlanTL-GOB-ES/CUTEXT)
	- also requires [TreeTagger](https://www.cis.lmu.de/~schmid/tools/TreeTagger/)
	- once installed, write the path to your CUTEXT repository in `config.json`
- Common NLP libraries: `nltk`, `spacy`, `langdetect`

## Contributors
- Amritpal Singh
- Oscar Arrocha
- Mustapha El Aichouni
- Eric López
