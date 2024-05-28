import pycrfsuite as crfs
from preprocessing import *
import os
import nltk
import string

ROOT_DIR = os.path.dirname(os.path.abspath(""))

def put_bio(
		data: List[Dict[str, Any]],
		data_tokens: List[List[Dict[str, Any]]],
		train_data_bio: List[List[List[str]]]
):
	tags = {"NEG": ["B-NEG", "I-NEG"], "NSCO": ["B-NSCO", "I-NSCO"],\
		 	"UNC": ["B-UNC", "I-UNC"], "USCO": ["B-USCO", "I-USCO"]}

	for d, doc in enumerate(data_tokens): 
		res_spans = data[d]["predictions"][0]["result"]
		for res_span in res_spans:
			span_tag = res_span["value"]["labels"][0]
			if span_tag in tags:
				start = res_span["value"]["start"]
				end = res_span["value"]["end"]
				for s, sent in enumerate(doc):
					# Check if res_span is in this sentence
					if not sent["spans"].any(axis=None) or sent["spans"][0][0] > end or sent["spans"][-1][-1] < start:
						continue
					# Find the span in the sentence
					found = False
					for i, span in enumerate(sent["spans"]):
						if span[0] >= start and span[1] <= end:
							train_data_bio[d][s][i] = tags[span_tag][found]
							found = True
						elif found: # found and we are out of the span
							break
					break # we found the sentence

def create_bio_tags(
		data: List[Dict[str, Any]],
		data_tokens: List[List[Dict[str, Any]]]
) -> List[List[List[str]]]:
	"""
	Creates BIO tags for the given data.
	"""
	data_bio = [[["O"]*len(sent["tokens"]) for sent in doc] for doc in data_tokens]
	put_bio(data, data_tokens, data_bio)
	return data_bio

def precompute_pos(
		tokens_path: str,
		pos_path: str,
		verbose: bool = False
):
	"""
	Precomputes POS tags for the given tokens and saves them to a file.
	"""
	nlp_es, nlp_ca = load_nlps()
	pos_dir = os.path.dirname(pos_path)
	if not os.path.exists(pos_dir):
		os.makedirs(pos_dir)
	tokens = load_tokens(tokens_path)
	pos = []
	for doc in tqdm(tokens, total=len(tokens), disable=not verbose):
		doc_pos = []
		for sentence in doc:
			lang = sentence["lang"]
			if lang == "ca":
				doc = nlp_ca(" ".join(sentence["tokens"]))
			else:
				doc = nlp_es(" ".join(sentence["tokens"]))
			doc_pos.append([token.pos_ for token in doc])
		pos.append(doc_pos)
	with open(pos_path, "w") as f:
		json.dump(pos, f)

class CRF:

	def __init__(
			self,
			model_path: str,
			trainer_params: Optional[Dict[str, Any]] = None,
			nlps: Optional[Tuple[Any, Any]] = None,
			verbose: bool = False
	):
		self.model_path = model_path
		if nlps is None:
			nlps = load_nlps()
		self.nlp_es, self.nlp_ca = nlps
		# Disable NER and parser for faster processing
		self.nlp_es.disable_pipes('ner', 'parser')
		self.nlp_ca.disable_pipes('ner', 'parser')

		# Set trainer parameters
		other_params = ["padding", "before_lim", "after_lim", "special_words"]
		self.padding = trainer_params.get("padding", False)
		self.before_lim = trainer_params.get("before_lim", 6)
		self.after_lim = trainer_params.get("after_lim", 1)
		self.special_words = trainer_params.get("special_words",\
			["nada", "ni", "nunca", "ningun", "ninguno", "ninguna", "alguna", "apenas", "para_nada", "ni_siquiera"])
		trainer_params = {k:v for k,v in trainer_params.items() if k not in other_params}
		self.verbose = verbose

		# Load model if it exists, otherwise create a new one
		if os.path.exists(model_path):
			self.tagger = crfs.Tagger()
			self.tagger.open(model_path)
			self.trainer = None
		else:
			model_dir = os.path.dirname(model_path)
			if not os.path.exists(model_dir):
				os.makedirs(model_dir)

			self.tagger = None
			self.trainer = crfs.Trainer(verbose=False)

			if trainer_params is not None:
				self.trainer.set_params(trainer_params)

	def word2features(
			self,
			sent: List[Tuple[str, str, str]],
			idx: int
	) -> List[str]:
		"""
		Returns a list of features for a given word in a sentence.
		"""
		word, pos, _ = sent[idx]
		features = [
			"bias",
			"word=" + word,
			"pos=" + pos,
			"is_first_capital=" + str(word[0].isupper()),
			"is_alphanumeric=" + str(word.isalnum()),
			"has_num=" + str(any(char.isdigit() for char in word)),
			"has_cap=" + str(any(char.isupper() for char in word)),
			"has_dash=" + str('-' in word),
			"has_us=" + str('_' in word),
			"has_punctuation=" + str(any(char in string.punctuation for char in word)),
			"is_special=" + str(word.lower() in self.special_words)
		]

		# Adding suffix features for suffix lengths of 2, 3, and 4
		suffix_lengths = [2, 3, 4]
		for length in suffix_lengths:
			if len(word) >= length:
				features.append(f"suffix-{length}=" + word[-length:])

		# Adding prefix features for prefix lengths of 2, 3, and 4
		prefix_lengths = [2, 3, 4]
		for length in prefix_lengths:
			if len(word) >= length:
				features.append(f"prefix-{length}=" + word[:length])

		pad_token = "<PAD>"
		if self.padding:
			# Ensure there are always self.before_lim bigram features, use padding if necessary
			for j in range(1, self.before_lim + 1):  # Generate bigrams up to self.before_lim
				if idx >= j:
					prev_word = sent[idx - j][0]
				else:
					prev_word = pad_token  # Use padding token if there are not enough previous words
				current_bigram = f"2GRAMBEFORE_{j}={prev_word}_{word}"
				features.append(current_bigram)

		else:
			# Adding bigram features for up to self.before_lim words before the current word
			for j in range(max(0, idx-self.before_lim), idx):
				if idx > 0 and j < idx:  # Check to ensure there is a previous word to form a bigram
					prev_word = sent[j][0]
					current_bigram = f"2GRAMBEFORE={prev_word}_{word}"
					features.append(current_bigram)

		# Add bigram features for up to self.after_lim words after the current word
		for j in range(1, self.after_lim + 1):
			if idx + j < len(sent):
				next_word = sent[idx + j][0]
				current_bigram_after = f"2GRAMAFTER_{j}={word}_{next_word}"
				features.append(current_bigram_after)
			elif self.padding:
				next_word = pad_token  # Use padding token if there is no next word
				current_bigram_after = f"2GRAMAFTER_{j}={word}_{next_word}"
				features.append(current_bigram_after)

		# beforepos
		for i in range(1, self.before_lim + 1):
			if idx - i >= 0:
				features.append(f'BEFOREPOS-{i}=' + sent[idx - i][0])
			elif self.padding:
				features.append(f'BEFOREPOS-{i}=START')
		
		# afterpos
		for i in range(1, self.after_lim + 1):
			if idx + i < len(sent):
				features.append(f'AFTERPOS-{i}=' + sent[idx + i][0])
			elif self.padding:
				features.append(f'AFTERPOS-{i}=END')
		
		return features

	def sent2features(
			self,
			sent: List[Tuple[str, str]]
	) -> List[List[str]]:
		"""
		Returns a list of features for each word in a sentence.
		"""
		return [self.word2features(sent, i) for i in range(len(sent))]

	def sent2labels(
			self,
			sent: List[Tuple[str, str]]
	) -> List[str]:
		"""
		Returns a list of labels for each word in a sentence.
		"""
		return [label for _,_,label in sent]

	def sent2tokens(
			self,
			sent: List[Tuple[str, str]]
	) -> List[str]:
		"""
		Returns a list of tokens in a sentence.
		"""
		return [token for token,_,_ in sent]
	
	def train(
			self,
			train_tokens_path: str,
			train_labels_path: str,
			train_pos_path: Optional[str] = None
	) -> None:
		"""
		Trains the CRF model on the given training data.
		"""
		if self.trainer is None:
			raise ValueError("Model already trained")
		
		# Load raw training data
		train_data = load_tokens(train_tokens_path)
		with open(train_labels_path, "r") as f:
			train_labels = json.load(f)
		if train_pos_path is not None:
			with open(train_pos_path, "r") as f:
				train_pos = json.load(f)

		# Process input data into words, POS tags, and labels
		sents = []
		for d, (doc_tokens, doc_labels) in tqdm(enumerate(zip(train_data, train_labels)),\
										  total=len(train_data), disable=not self.verbose):
			sent = []
			for s, (sentence, labels) in enumerate(zip(doc_tokens, doc_labels)):
				if train_pos_path is not None:
					pos = train_pos[d][s]
				else:
					lang = sentence["lang"]
					if lang == "ca":
						doc = self.nlp_ca(" ".join(sentence["tokens"]))
					else:
						doc = self.nlp_es(" ".join(sentence["tokens"]))
					pos = [token.pos_ for token in doc]
				sent.extend(list(zip(sentence["tokens"], pos, labels)))
			sents.append(sent)

		# Create input features and target labels
		X_train = [self.sent2features(sent) for sent in sents]
		y_train = [self.sent2labels(sent) for sent in sents]

		for xseq, yseq in zip(X_train, y_train):
			self.trainer.append(xseq, yseq)

		# train and save model
		if self.verbose: print("Training model...")
		self.trainer.train(self.model_path)
		if self.verbose: print("Model trained")

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

		# POS tagging if not provided
		if pos is None:
			if lang == "ca":
				doc = self.nlp_ca(" ".join(tokens))
			else:
				doc = self.nlp_es(" ".join(tokens))
			pos = [token.pos_ for token in doc]
		sent = [(token, p, "") for token,p in zip(tokens, pos)]
		# Get features and predict labels
		x = self.sent2features(sent)
		y = self.tagger.tag(x)

		return y

	def process(
			self,
			data_path: str,
			tokens_path: str,
			save_path: str,
			pos_path: Optional[str] = None
	) -> None:
		"""
		Processes the given data and saves the results to a file.
		"""
		with open(data_path, "r") as f:
			data = json.load(f)
		data_tokens = load_tokens(tokens_path)
		if pos_path is not None:
			with open(pos_path, "r") as f:
				pos = json.load(f)
		else:
			pos = None
		save_dir = os.path.dirname(save_path)
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		# Predict labels
		preds = []
		for d, doc_tokens in tqdm(enumerate(data_tokens), total=len(data_tokens), disable=not self.verbose):
			doc_results = []
			for i, sentence in enumerate(doc_tokens):
				tokens = sentence["tokens"]
				if pos is not None:
					sent_pos = pos[d][i]
				else:
					sent_pos = None
				lang = sentence["lang"]
				labels = self.predict(tokens, sent_pos, lang)
				doc_results.append(labels)
			preds.append(doc_results)

		# Save preds to formated predictions
		tags = {"B-NEG": "NEG", "I-NEG": "NEG", "B-NSCO": "NSCO", "I-NSCO": "NSCO",\
		  		"B-UNC": "UNC", "I-UNC": "UNC", "B-USCO": "USCO", "I-USCO": "USCO"}
		predictions = []
		for d, doc_preds in enumerate(preds):
			doc_results = []
			for s, sent_preds in enumerate(doc_preds):
				tag = None
				for i, label in enumerate(sent_preds):
					if tags.get(label, ".") != tag:
						if tag is not None:
							doc_results.append({
								"value": {
									"start": int(start),
									"end": int(end),
									"labels": [tag]
								}
							})
						start = None
						end = None
						tag = None
					if label == "O":
						continue
					if start is None:
						start, end = data_tokens[d][s]["spans"][i]
						tag = tags[label]
					else:
						end = data_tokens[d][s]["spans"][i][1]
				if tag is not None:
					doc_results.append({
						"value": {
							"start": int(start),
							"end": int(end),
							"labels": [tag]
						}
					})
			predictions.append([
				{
					"result": doc_results
				}
			])
		
		for d in range(len(data)):
			data[d]["predictions"] = predictions[d]

		with open(save_path, "w") as f:
			json.dump(data, f, indent=4)
