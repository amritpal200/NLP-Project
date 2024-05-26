import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import fasttext
import fasttext.util
from sklearn.preprocessing import OneHotEncoder
from preprocessing import *

def precompute_lemmas(
		tokens_path: str,
		lemmas_path: str,
		verbose: bool = False
):
	"""
	Precomputes lemmas for the given tokens and saves them to a file.
	"""
	nlp_es, nlp_ca = load_nlps()
	lemmas_dir = os.path.dirname(lemmas_path)
	if not os.path.exists(lemmas_dir):
		os.makedirs(lemmas_dir)
	tokens = load_tokens(tokens_path)
	lemmas = lemmatize_corpus(tokens, nlp_es, nlp_ca, verbose)
	lemmas = [[sent["tokens"].tolist() for sent in doc] for doc in lemmas]
	with open(lemmas_path, 'w') as f:
		json.dump(lemmas, f)

def load_fasttext():
	fasttext.util.download_model('es', if_exists='ignore')
	return fasttext.load_model('cc.es.300.bin')

class SlidingWindowDataset(Dataset):
	def __init__(self, data_tokens, data_lemmas, data_pos, data_labels, ft, seq_len=10, padding_value=0):
		self.data_tokens = [sent["tokens"] for doc in data_tokens for sent in doc]
		self.data_tokens = np.concatenate(self.data_tokens)
		self.data_lemmas = [sent for doc in data_lemmas for sent in doc]
		self.data_lemmas = np.concatenate(self.data_lemmas)
		self.data_pos = [sent for doc in data_pos for sent in doc]
		self.data_pos = np.concatenate(self.data_pos)
		self.data_labels = [sent for doc in data_labels for sent in doc]
		self.data_labels = np.concatenate(self.data_labels)

		self.ft = ft
		self.seq_len = seq_len
		self.padding_value = padding_value

		self.pos2idx = {"ADJ": 0, "ADP": 1, "ADV": 2, "AUX": 3, "CCONJ": 4, "DET": 5, "INTJ": 6, "NOUN": 7,\
						"NUM": 8, "PART": 9, "PRON": 10, "PROPN": 11, "PUNCT": 12, "SCONJ": 13, "SYM": 14,\
						"VERB": 15, "X": 16}
		self.label2idx = {"B-NEG": 0, "I-NEG": 1, "B-NSCO": 2, "I-NSCO": 3,\
		  		"B-UNC": 4, "I-UNC": 5, "B-USCO": 6, "I-USCO": 7, "O": 8}
		
		self.pos_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
		self.pos_encoder.fit(np.array(list(self.pos2idx.values())).reshape(-1, 1))

	def pad(self, tokens, lemmas, pos, labels):
		# Pad if shorter than seq_len
		if len(tokens) < self.seq_len:
			tokens = np.pad(tokens, (0, self.seq_len - len(tokens)), 'constant', constant_values=self.padding_value)
			lemmas = np.pad(lemmas, (0, self.seq_len - len(lemmas)), 'constant', constant_values=self.padding_value)
			pos = np.pad(pos, (0, self.seq_len - len(pos)), 'constant', constant_values=self.padding_value)
			labels = np.pad(labels, (0, self.seq_len - len(labels)), 'constant', constant_values=self.padding_value)
		return tokens, lemmas, pos, labels

	def get_vectors(self, tokens, lemmas, pos, labels):
		token_embeddings = np.array([self.ft.get_word_vector(token) for token in tokens])
		lemma_embeddings = np.array([self.ft.get_word_vector(lemma) for lemma in lemmas])
		pos_indices = np.array([self.pos2idx.get(p, 16) for p in pos]).reshape(-1, 1)
		pos_one_hot = self.pos_encoder.transform(pos_indices)
		label_indices = np.array([self.label2idx.get(label, 8) for label in labels])

		x = np.concatenate((token_embeddings, lemma_embeddings, pos_one_hot), axis=1)
		y = label_indices
		return x, y
	
	def getseq(self, start, end):
		tokens = self.data_tokens[start:end]
		lemmas = self.data_lemmas[start:end]
		pos = self.data_pos[start:end]
		labels = self.data_labels[start:end]

		tokens, lemmas, pos, labels = self.pad(tokens, lemmas, pos, labels)

		x, y = self.get_vectors(tokens, lemmas, pos, labels)
		return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class OverlappingWindowDataset(SlidingWindowDataset):
	def __init__(self, data_tokens, data_lemmas, data_pos, data_labels, ft, seq_len=10, padding_value=0):
		super().__init__(data_tokens, data_lemmas, data_pos, data_labels, ft, seq_len, padding_value)

	def __len__(self):
		return len(self.data_tokens)
	
	def __getitem__(self, idx):
		# get sequence around the idx
		start = max(0, idx - self.seq_len//2)
		end = min(len(self.data_tokens), idx + self.seq_len//2)
		return self.getseq(start, end)
	
class NonOverlappingWindowDataset(SlidingWindowDataset):
	def __init__(self, data_tokens, data_lemmas, data_pos, data_labels, ft, seq_len=10, padding_value=0):
		super().__init__(data_tokens, data_lemmas, data_pos, data_labels, ft, seq_len, padding_value)

	def __len__(self):
		return (len(self.data_tokens) + self.seq_len - 1) // self.seq_len

	def __getitem__(self, idx):
		# get sequence around the idx
		start = idx * self.seq_len
		end = start + self.seq_len
		return self.getseq(start, end)

