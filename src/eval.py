import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from preprocessing import *
import negex, crf
from time import time

ROOT_DIR = os.path.dirname(os.path.abspath(""))

class EvalOfficial:
	def __init__(self):
		pass
	
	def process(self, data):
		final_chars = []
		for i in range(len(data)):
			text = data[i]['data']['text']
			chars = list(text)
			for value in data[i]['predictions'][0]['result']:
				index = (value['value']['start'], value['value']['end'])
				label = value['value']['labels'][0]
				for j in range(index[0], index[1]-1):
					chars[j] = label
			final_chars.extend(chars)
		return final_chars

	def calc(self, pred, groundtruth):
		pred = self.process(pred)
		gt = self.process(groundtruth)
		precision = precision_score(gt, pred, average='micro')
		recall = recall_score(gt, pred, average='micro')
		f1 = f1_score(gt, pred, average='micro')
		return precision, recall, f1

def save_results(
		path: str,
		metrics: dict,
		hyperparameters: dict,
		method: str
) -> None:
	try:
		with open(path, 'r') as file:
			data = json.load(file)
	except FileNotFoundError:
		data = {}

	if method in data:
		# Check if instance with same hyperparameters already exists
		for inst in data[method]:
			if inst['hyperparameters'] == hyperparameters:
				inst['metrics'] = metrics
				break
		else:
			# Create new instance
			instance_dict = {
				'metrics': metrics,
				'hyperparameters': hyperparameters
			}
			data[method].append(instance_dict)
	else:
		instance_dict = {
			'metrics': metrics,
			'hyperparameters': hyperparameters
		}
		data[method] = [instance_dict]

	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'w') as file:
		json.dump(data, file, indent=4)

class EvalModel(EvalOfficial):
	def __init__(
			self,
			save_dir: str,
			results_dir: str,
			data_path: str,
			load_existing_tokens: bool = False,
			**kwargs
	):
		super(EvalModel, self).__init__()
		self.verbose = kwargs.get("verbose", False)
		self.save_dir = save_dir
		self.results_dir = results_dir
		self.data_path = data_path
		self.data = self.load_data(self.data_path)
		if not load_existing_tokens:
			if self.verbose: print("Creating evaluation tokens...")
			self.create_tokens(
				data=self.data,
				file_name="data_tokens.json",
				lemmatize=kwargs.get("lemmatize", False),
				remove_punctuation=kwargs.get("remove_punctuation", True),
				replace_numbers=kwargs.get("replace_numbers", None)
			)
		self.name = "Default"

	def load_data(
			self,
			data_path: str
	) -> List[dict]:
		with open(data_path, 'r', encoding='utf8') as _f:
			data = json.load(_f)
		return data
	
	def create_tokens(
			self,
			data: List[dict],
			file_name: str,
			lemmatize: bool = False,
			remove_punctuation: bool = True,
			replace_numbers: Optional[str] = None
	) -> List[dict]:
		sents = sent_tokenize_corpus(data, verbose=self.verbose)
		nlp_es, nlp_cat = load_nlps()
		tokens = tokenize_corpus(
			sents, nlp_es, nlp_cat,
			remove_punctuation=remove_punctuation,
			replace_numbers=replace_numbers,
			verbose=self.verbose
		)
		if lemmatize:
			tokens = lemmatize_corpus(tokens, nlp_es, nlp_cat, verbose=self.verbose)
		save_tokens(tokens, os.path.join(self.save_dir, file_name))
		return tokens
	
	def load_tokens(self) -> List[dict]:
		return load_tokens(os.path.join(self.save_dir, "data_tokens.json"))
	
	def predict(self, **kwargs) -> None:
		raise NotImplementedError
	
	def save_results(
			self,
			metrics: dict,
			hyperparameters: dict
	) -> None:
		save_results(
			os.path.join(self.results_dir, "results.json"),
			metrics,
			hyperparameters,
			self.name
		)

	def evaluate(self, **kwargs) -> dict:
		if self.verbose: print("Predicting...")
		start_time = time()
		self.predict(**kwargs)
		total_time = time() - start_time
		with open(os.path.join(self.save_dir, "data_predictions.json"), 'r', encoding='utf8') as _f:
			predictions = json.load(_f)
		self.data = self.load_data(self.data_path)
		if self.verbose: print("Calculating metrics...")
		p, r, f1 = self.calc(predictions, self.data)
		metrics = {
			"precision": p,
			"recall": r,
			"f1": f1,
			"time": round(total_time, 4)
		}
		self.save_results(
			metrics,
			kwargs
		)
		return metrics

class EvalNegex(EvalModel):
	def __init__(
			self,
			save_dir: str,
			results_dir: str,
			data_path: str,
			load_existing_tokens: bool = False,
			**kwargs
	):
		super(EvalNegex, self).__init__(save_dir, results_dir, data_path, load_existing_tokens, **kwargs)
		self.name = "Negex"

	def predict(self, **kwargs) -> None:
		tokens = self.load_tokens()
		predictions = negex.process_data(tokens, max_context_size=kwargs.get("max_context_size", 5))
		negex.write_predictions(self.data, predictions, "data_predictions.json", dir=self.save_dir.split("/")[-1])

class EvalCRF(EvalModel):
	def __init__(
			self,
			save_dir: str,
			results_dir: str,
			train_data_path: str,
			eval_data_path: str,
			load_existing_train_tokens: bool = False,
			load_existing_eval_tokens: bool = False,
			**kwargs
	):
		super(EvalCRF, self).__init__(save_dir, results_dir, eval_data_path, load_existing_eval_tokens, **kwargs)
		self.train_data_path = train_data_path
		self.eval_data_path = eval_data_path
		self.train_data = self.load_data(self.train_data_path)
		if not load_existing_train_tokens:
			if self.verbose: print("Creating training tokens...")
			self.create_tokens(
				data=self.train_data,
				file_name="train_data_tokens.json",
				lemmatize=kwargs.get("lemmatize", False),
				remove_punctuation=kwargs.get("remove_punctuation", True),
				replace_numbers=kwargs.get("replace_numbers", None)
			)
		with open(os.path.join(self.save_dir, "train_data_tokens.json"), 'r', encoding='utf8') as _f:
			train_data_bio = crf.create_bio_tags(self.train_data, json.load(_f))
		with open(os.path.join(self.save_dir, "train_data_bio.json"), 'w', encoding='utf8') as _f:
			json.dump(train_data_bio, _f)
		if self.verbose: print("Precomputing training POS tags...")
		crf.precompute_pos(
			tokens_path=os.path.join(self.save_dir, "train_data_tokens.json"),
			pos_path=os.path.join(self.save_dir, "train_data_pos.json"),
			verbose=self.verbose
		)
		if self.verbose: print("Precomputing evaluation POS tags...")
		crf.precompute_pos(
			tokens_path=os.path.join(self.save_dir, "data_tokens.json"),
			pos_path=os.path.join(self.save_dir, "data_pos.json"),
			verbose=self.verbose
		)
		if self.verbose: print("Loading NLP models...")
		self.nlps = load_nlps()
		self.name = "CRF"
		self.model = None
		self.kwargs = kwargs

	def predict(self, **kwargs) -> None:
		self.model.process(
			data_path=self.data_path,
			tokens_path=os.path.join(self.save_dir, "data_tokens.json"),
			save_path=os.path.join(self.save_dir, "data_predictions.json"),
			pos_path=os.path.join(self.save_dir, "data_pos.json"),		
		)

	def evaluate(self, **kwargs) -> dict:
		if self.verbose: print("Instantiating CRF...")
		self.model = crf.CRF(
			model_path=os.path.join(self.save_dir, "crf_0_0"), # TODO: allow multiple models
			trainer_params=kwargs,
			nlps=self.nlps,
			verbose=self.verbose
		)
		if self.verbose: print("Training CRF...")
		self.model.train(
			train_tokens_path=os.path.join(self.save_dir, "train_data_tokens.json"),
			train_labels_path=os.path.join(self.save_dir, "train_data_bio.json"),
			train_pos_path=os.path.join(self.save_dir, "train_data_pos.json")
		)
		hyperparams = kwargs
		hyperparams["lemmatize"] = self.kwargs.get("lemmatize", False)
		hyperparams["remove_punctuation"] = self.kwargs.get("remove_punctuation", True)
		hyperparams["replace_numbers"] = self.kwargs.get("replace_numbers", None)
		return super(EvalCRF, self).evaluate(**hyperparams)

if __name__ == "__main__":
	with open(os.path.join(ROOT_DIR, "data", 'test_data.json'), 'r', encoding='utf8') as _f:
		test_data = json.load(_f)
	metric = EvalOfficial()
	p, r, f1 = metric.calc(test_data, test_data)
	print(f'Precision: {p}, Recall:{r}, F1:{f1}')
