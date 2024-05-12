import pycrfsuite as crfs
from preprocessing import *
import os

ROOT_DIR = os.path.dirname(os.path.abspath(""))

class CRF:

	def __init__(
			self,
			model_path: str,
			trainer_params: Optional[Dict[str, Any]] = None
	):
		self.model_path = model_path

		if os.path.exists(model_path):
			self.tagger = crfs.Tagger()
			self.tagger.open(model_path)
			self.trainer = None
		else:
			self.tagger = None
			self.trainer = crfs.Trainer()

			if trainer_params is not None:
				self.trainer.set_params(trainer_params)

	def word2features(
			self,
			sent: List[Tuple[str, str]],
			i: int
	):
		"""
		Returns a list of features for a given word in a sentence.
		"""
		word, _ = sent[i]
		features = [
			"bias",
			"word.lower=" + word.lower(),
		]

		# add more features

		return features

	def sent2features(
			self,
			sent: List[Tuple[str, str]]
	):
		"""
		Returns a list of features for each word in a sentence.
		"""
		return [self.word2features(sent, i) for i in range(len(sent))]

	def sent2labels(
			self,
			sent: List[Tuple[str, str]]
	):
		"""
		Returns a list of labels for each word in a sentence.
		"""
		return [label for _, label in sent]

	def sent2tokens(
			self,
			sent: List[Tuple[str, str]]
	):
		"""
		Returns a list of tokens in a sentence.
		"""
		return [token for token, _ in sent]
	
	def train(
			self,
			train_tokens_path: str,
			train_labels_path: str
	):
		"""
		Trains the CRF model on the given training data.
		"""
		if self.trainer is None:
			raise ValueError("Model already trained")
		
		train_data = load_tokens(os.path.join(ROOT_DIR, "data", "training_data_tokens.json"))
		# train_labels = # here load labels (BIO tagging) of same shape as train_data

		# sents = # here merge train_data and train_labels so format is List[List[Tuple[str, str]]]

		X_train = [self.sent2features(sent) for sent in sents]
		y_train = [self.sent2labels(sent) for sent in sents]

		for xseq, yseq in zip(X_train, y_train):
			self.trainer.append(xseq, yseq)

		# train and save model
		self.trainer.train(self.model_path)

		self.tagger = crfs.Tagger()
		self.tagger.open(self.model_path)

	def predict(
			self,
			tokens: List[str]
	) -> List[Tuple[str, str]]:
		"""
		Predicts NER tags.
			Returns a list of (token, label) tuples for the given tokens.
		"""
		if self.tagger is None:
			raise ValueError("Model not trained")

		# convert tokens to List[Tuple[str, str]]
		sent = [(token, "") for token in tokens]
		x = self.sent2features(sent)
		y = self.tagger.tag(x)

		return list(zip(tokens, y))
