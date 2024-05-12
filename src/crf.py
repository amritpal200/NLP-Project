import pycrfsuite as crfs
from preprocessing import *
import os
import nltk

ROOT_DIR = os.path.dirname(os.path.abspath(""))
def put_bio(data, data_tokens, train_data_bio):
	tags = {"NEG": "B-NEG", "NSCO": "I-NEG", "UNC": "B-UNC", "USCO": "I-UNC"}

	for i, tokens_group in enumerate(data_tokens): 
		spans = data[i]["predictions"][0]["result"]
		for span in spans:
			span_tag = span["value"]["labels"][0]
			if span_tag in tags:
				start = span["value"]["start"]
				for j, token_info in enumerate(tokens_group):
					# Check if the last span end is before the current span start
					if not token_info["spans"].any(axis=None) or token_info["spans"][-1][-1] < start:
						continue
					# Process spans within the current token_info
					for index, k in enumerate(token_info["spans"]):
						# This condition assumes perfect alignment
						if k[0] <= start <= k[1]:
							train_data_bio[i][j][index] = tags[span_tag]
							break
					break  # Breaking only if we made a change

def create_bio_tags(data, data_tokens):
	data_bio = [[["O"]*len(sent["tokens"]) for sent in doc] for doc in data_tokens]
	put_bio(data, data_tokens, data_bio)
	return data_bio

class CRF:

	def __init__(
			self,
			model_path: str,
			trainer_params: Optional[Dict[str, Any]] = None,
			verbose: bool = False
	):
		self.model_path = model_path
		self.nlp_es, self.nlp_ca = load_nlps()
		self.nlp_es.disable_pipes('ner', 'parser')
		self.nlp_ca.disable_pipes('ner', 'parser')

		if os.path.exists(model_path):
			self.tagger = crfs.Tagger()
			self.tagger.open(model_path)
			self.trainer = None
		else:
			model_dir = os.path.dirname(model_path)
			if not os.path.exists(model_dir):
				os.makedirs(model_dir)

			self.tagger = None
			self.trainer = crfs.Trainer(verbose=verbose)

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
		word, pos, _ = sent[i]
		features = [
			"bias",
			"word=" + word,
			"pos=" + pos,
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
		return [label for _,_,label in sent]

	def sent2tokens(
			self,
			sent: List[Tuple[str, str]]
	):
		"""
		Returns a list of tokens in a sentence.
		"""
		return [token for token,_,_ in sent]
	
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
		
		train_data = load_tokens(train_tokens_path)
		with open(train_labels_path, "r") as f:
			train_labels = json.load(f)

		sents = []
		for doc_tokens, doc_labels in tqdm(zip(train_data, train_labels), total=len(train_data)):
			sent = []
			for sentence, labels in zip(doc_tokens, doc_labels):
				lang = sentence["lang"]
				if lang == "ca":
					doc = self.nlp_ca(" ".join(sentence["tokens"]))
				else:
					doc = self.nlp_es(" ".join(sentence["tokens"]))
				pos = [token.pos_ for token in doc]
				sent.extend(list(zip(sentence["tokens"], pos, labels)))
			sents.append(sent)

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
			tokens: List[str],
			pos: Optional[List[str]] = None,
			lang: Optional[str] = "es"
	) -> List[Tuple[str, str]]:
		"""
		Predicts NER tags.
			Returns a list of (token, label) tuples for the given tokens.
		"""
		if self.tagger is None:
			raise ValueError("Model not trained")

		# convert tokens to List[Tuple[str, str]]
		if pos is None:
			if lang == "ca":
				doc = self.nlp_ca(" ".join(tokens))
			else:
				doc = self.nlp_es(" ".join(tokens))
			pos = [token.pos_ for token in doc]
		sent = [(token, p, "") for token,p in zip(tokens, pos)]
		x = self.sent2features(sent)
		y = self.tagger.tag(x)

		return y

	# def process
