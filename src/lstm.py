import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
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
	with open(lemmas_path, "w") as f:
		json.dump(lemmas, f)

def load_fasttext():
	fasttext.util.download_model("es", if_exists="ignore")
	return fasttext.load_model("cc.es.300.bin")

def load_data(tokens_path, lemmas_path, pos_path, labels_path, frac=1.0):
	tokens = load_tokens(tokens_path)
	n = int(len(tokens) * frac)
	tokens = tokens[:n]
	with open(lemmas_path, "r", encoding="utf-8") as f:
		lemmas = json.load(f)[:n]
	with open(pos_path, "r", encoding="utf-8") as f:
		pos = json.load(f)[:n]
	with open(labels_path, "r", encoding="utf-8") as f:
		labels = json.load(f)[:n]
	return tokens, lemmas, pos, labels

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
		
		self.pos_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
		self.pos_encoder.fit(np.array(list(self.pos2idx.values())).reshape(-1, 1))

	def pad(self, tokens, lemmas, pos, labels):
		# Pad if shorter than seq_len
		if len(tokens) < self.seq_len:
			tokens = np.pad(tokens, (0, self.seq_len - len(tokens)), "constant", constant_values=self.padding_value)
			lemmas = np.pad(lemmas, (0, self.seq_len - len(lemmas)), "constant", constant_values=self.padding_value)
			pos = np.pad(pos, (0, self.seq_len - len(pos)), "constant", constant_values=self.padding_value)
			labels = np.pad(labels, (0, self.seq_len - len(labels)), "constant", constant_values=self.padding_value)
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

class NegationDetectionModel(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
		super(NegationDetectionModel, self).__init__()
		
		# BiLSTM Layer
		self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
		# Dense Layer
		self.fc = nn.Linear(hidden_dim * 2, output_dim) # hidden_dim * 2 is done because is BIdirectional. Hence, we have the double dimensions
		
	def forward(self, word_embeds):
		lstm_out, _ = self.bilstm(word_embeds)
		out = self.fc(lstm_out)
		return out

class LSTM:
	def __init__(
			self,
			model_path: str,
			device: torch.device,
			hyperparams: Optional[Dict[str, Any]] = None,
			ft: Optional[fasttext.FastText._FastText] = None,
			verbose: bool = False
	):
		self.model_path = model_path
		self.device = device
		self.hyperparams = hyperparams
		self.verbose = verbose

		self.ft = ft
		if ft is None:
			if self.verbose: print("Loading FastText...")
			self.ft = load_fasttext()

		pretrained = os.path.exists(model_path)
		if pretrained:
			checkpoint = torch.load(model_path)
			self.hyperparams = checkpoint["hyperparams"]

		self.model = NegationDetectionModel(
			self.hyperparams.get("input_dim", 617),
			self.hyperparams.get("hidden_dim", 128),
			self.hyperparams.get("num_layers", 2),
			self.hyperparams.get("output_dim", 9)
		).to(device)

		if pretrained:
			self.model.load_state_dict(checkpoint["model_state_dict"])

		self.seq_len = self.hyperparams.get("seq_len", 10)
		self.padding_value = self.hyperparams.get("padding_value", 0)
		self.epochs = self.hyperparams.get("epochs", 10)
		self.batch_size = self.hyperparams.get("batch_size", 32)
		self.lr = self.hyperparams.get("lr", 0.001)
		self.num_workers = self.hyperparams.get("num_workers", 0)
		self.dataset_type = self.hyperparams.get("dataset_type", "overlapping")
	
	def train(
			self,
			train_tokens_path: str,
			train_lemmas_path: str,
			train_pos_path: str,
			train_labels_path: str,
			**kwargs
	) -> List[float]:
		epochs = kwargs.get("epochs", self.epochs)
		batch_size = kwargs.get("batch_size", self.batch_size)
		lr = kwargs.get("lr", self.lr)
		num_workers = kwargs.get("num_workers", self.num_workers)
		frac = kwargs.get("frac", 1.0)

		if self.verbose: print("Preparing training set...")
		train_data = load_data(train_tokens_path, train_lemmas_path, train_pos_path, train_labels_path, frac=frac)
		if self.dataset_type == "overlapping":
			train_dataset = OverlappingWindowDataset(*train_data, self.ft, self.seq_len, self.padding_value)
		elif self.dataset_type == "non-overlapping":
			train_dataset = NonOverlappingWindowDataset(*train_data, self.ft, self.seq_len, self.padding_value)
		train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
		
		criterion = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
		self.model.train()
		
		if self.verbose: print("Training model...")
		losses = []
		for epoch in range(epochs):
			for i, (sequences, targets) in enumerate(train_dataloader):
				# Forward pass
				sequences = sequences.to(self.device)
				targets = targets.to(self.device)

				outputs = self.model(sequences)
				outputs = outputs.view(-1, outputs.shape[-1])
				targets = targets.view(-1).long()
				# Compute loss
				loss = criterion(outputs, targets)
				
				# Backward pass and optimization
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				losses.append(loss.item())
				
				if self.verbose: print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_dataloader)}], Loss: {loss.item()}", end="\r")
			if self.verbose: print()
		return losses
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		self.model.eval()
		with torch.no_grad():
			x = x.to(self.device)
			outputs = self.model(x)
		return outputs
	
	def predict(
			self,
			tokens: List[str],
			lemmas: List[str],
			pos: List[str]
	) -> List[str]:
		labels = ["O"] * len(tokens)
		tokens = {"tokens": tokens}
		if self.dataset_type == "overlapping":
			temp_dataset = OverlappingWindowDataset([[tokens]], [[lemmas]], [[pos]], [[labels]], self.ft, self.seq_len, self.padding_value)
		elif self.dataset_type == "non-overlapping":
			temp_dataset = NonOverlappingWindowDataset([[tokens]], [[lemmas]], [[pos]], [[labels]], self.ft, self.seq_len, self.padding_value)
		temp_dataloader = DataLoader(temp_dataset, batch_size=1, shuffle=False, num_workers=0)
		idx2label = {v: k for k, v in temp_dataset.label2idx.items()}

		predictions = []
		for sequences, _ in temp_dataloader:
			outputs = self.forward(sequences)
			_, predicted = torch.max(outputs, 2)
			predicted = predicted.squeeze().tolist()
			predictions.extend([idx2label[p] for p in predicted])
		return predictions
	
	def evaluate(
			self,
			test_tokens_path: str,
			test_lemmas_path: str,
			test_pos_path: str,
			test_labels_path: str,
			**kwargs
	) -> Tuple[float, float]:
		batch_size = kwargs.get("batch_size", self.batch_size)
		num_workers = kwargs.get("num_workers", self.num_workers)
		frac = kwargs.get("frac", 1.0)

		if self.verbose: print("Preparing test set...")
		test_data = load_data(test_tokens_path, test_lemmas_path, test_pos_path, test_labels_path, frac=frac)
		if self.dataset_type == "overlapping":
			test_dataset = OverlappingWindowDataset(*test_data, self.ft, self.seq_len, self.padding_value)
		elif self.dataset_type == "non-overlapping":
			test_dataset = NonOverlappingWindowDataset(*test_data, self.ft, self.seq_len, self.padding_value)
		test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
		
		criterion = nn.CrossEntropyLoss()

		self.model.eval()
		total_loss = 0
		correct = 0
		total = 0
		
		with torch.no_grad():
			for sequences, targets in tqdm(test_dataloader):
				# Forward pass
				sequences = sequences.to(self.device)
				targets = targets.to(self.device)
				
				outputs = self.model(sequences)
				outputs = outputs.view(-1, outputs.shape[-1])
				targets = targets.view(-1).long()
				
				# Compute loss
				loss = criterion(outputs, targets)
				total_loss += loss.item()
				
				# Compute accuracy
				_, predicted = torch.max(outputs, 1)
				correct += (predicted == targets).sum().item()
				total += targets.size(0)
		
		average_loss = total_loss / len(test_dataloader)
		accuracy = correct / total
		if self.verbose: print(f"Test Loss: {average_loss:.4f}, Test Accuracy: {accuracy:.4f}")
		return average_loss, accuracy
	
	def process(
			self,
			data_path: str,
			tokens_path: str,
			pos_path: str,
			lemmas_path: str,
			save_path: str,
	) -> None:
		raise NotImplementedError

	def save(self) -> None:
		model_dir = os.path.dirname(self.model_path)
		if not os.path.exists(model_dir):
			os.makedirs(model_dir)
		torch.save(
			{
				"model_state_dict": self.model.state_dict(),
				"hyperparams": self.hyperparams
			},
			self.model_path
		)
