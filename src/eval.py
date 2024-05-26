import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from preprocessing import *
import negex, crf, lstm
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
		if kwargs.get("save_results", True):
			kwargs.pop("save_results", None)
			self.save_results(
				metrics,
				kwargs
			)
		return metrics

class EnhancedEvalModel(EvalModel):
	def __init__(
			self,
			save_dir: str,
			results_dir: str,
			train_data_path: str,
			eval_data_path: str,
			load_existing_train_tokens: bool = False,
			load_existing_eval_tokens: bool = False,
			load_existing_train_pos: bool = False,
			load_existing_eval_pos: bool = False,
			**kwargs
	):
		super(EnhancedEvalModel, self).__init__(save_dir, results_dir, eval_data_path, load_existing_eval_tokens, **kwargs)
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
		train_data_bio = crf.create_bio_tags(self.train_data, load_tokens(os.path.join(self.save_dir, "train_data_tokens.json")))
		with open(os.path.join(self.save_dir, "train_data_bio.json"), 'w', encoding='utf8') as _f:
			json.dump(train_data_bio, _f)
		if not load_existing_train_pos:
			if self.verbose: print("Precomputing training POS tags...")
			crf.precompute_pos(
				tokens_path=os.path.join(self.save_dir, "train_data_tokens.json"),
				pos_path=os.path.join(self.save_dir, "train_data_pos.json"),
				verbose=self.verbose
			)
		if not load_existing_eval_pos:
			if self.verbose: print("Precomputing evaluation POS tags...")
			crf.precompute_pos(
				tokens_path=os.path.join(self.save_dir, "data_tokens.json"),
				pos_path=os.path.join(self.save_dir, "data_pos.json"),
				verbose=self.verbose
			)
		if self.verbose: print("Loading NLP models...")
		self.nlps = load_nlps()
		self.name = "Enhanced"
		self.model = None
		self.kwargs = kwargs

	def grid_search(
			self,
			params_ranges: dict
	) -> List[dict]:
		"""
		Perform grid search over the hyperparameters.
		"""
		combinations = [] # here we will store params and metrics
		keys = list(params_ranges.keys())
		values = list(params_ranges.values())
		indexes = [0] * len(keys)
		total = 1
		for v in values:
			total *= len(v)
		p = 0
		while True:
			print(f"Progress: {p}/{total}", end="\r")
			p += 1
			params = {keys[i]: values[i][indexes[i]] for i in range(len(keys))}
			metrics = self.evaluate(**params)
			combinations.append({"params": params, "metrics": metrics})
			i = 0
			while i < len(keys):
				indexes[i] += 1
				if indexes[i] == len(values[i]):
					indexes[i] = 0
					i += 1
				else:
					break
			if i == len(keys):
				break
		print(f"Progress: {total}/{total}")
		return sorted(combinations, key=lambda x: x["metrics"]["f1"], reverse=True)
	
	def cross_validation(
			self,
			n_splits: int,
			file_names: List[str] = [
				"train_data_tokens.json",
				"train_data_bio.json",
				"train_data_pos.json",
				"data_tokens.json",
				"data_pos.json"
			],
			**kwargs
	) -> dict:
		"""
		Perform cross-validation over the data.
		"""
		# save original dataoriginal_train_data_path
		original_train_data_path = self.train_data_path
		original_eval_data_path = self.eval_data_path
		original_train_data = self.train_data.copy()
		original_data = []
		file_names = file_names.extend(kwargs.get("extra_files", [])).extend(kwargs.get("extra_files", []))
		for file_name in file_names:
			with open(os.path.join(self.save_dir, file_name), 'r', encoding='utf8') as _f:
				original_data.append(json.load(_f))

		# split, save and evaluate
		total_docs = len(self.train_data)
		split_size = total_docs // n_splits
		split_idxs = [i * split_size for i in range(n_splits)] + [total_docs]
		results = []
		for i in range(n_splits):
			self.train_data_path = os.path.join(self.save_dir, "train_data.json")
			self.eval_data_path = os.path.join(self.save_dir, "data.json")
			self.data_path = self.eval_data_path

			with open(self.train_data_path, 'w', encoding='utf8') as _f:
				json.dump(original_train_data[:split_idxs[i]] + original_train_data[split_idxs[i+1]:], _f)
			with open(self.data_path, 'w', encoding='utf8') as _f:
				json.dump(original_train_data[split_idxs[i]:split_idxs[i+1]], _f)
				
			for file_name, data in zip(file_names, original_data):
				with open(os.path.join(self.save_dir, file_name), 'w', encoding='utf8') as _f:
					if file_name.startswith("train"):
						json.dump(data[:split_idxs[i]] + data[split_idxs[i+1]:], _f)
					else:
						json.dump(data[split_idxs[i]:split_idxs[i+1]], _f)

			results.append(self.evaluate(**kwargs, save_results=False))
		
		# return data to original state
		os.remove(self.train_data_path)
		os.remove(self.eval_data_path)
		self.train_data_path = original_train_data_path
		self.eval_data_path = original_eval_data_path
		self.data_path = self.eval_data_path

		for file_name, data in zip(file_names, original_data):
			with open(os.path.join(self.save_dir, file_name), 'w', encoding='utf8') as _f:
				json.dump(data, _f)

		cv_results = {
			"results": results,
			"avg_precision": sum([r["precision"] for r in results]) / n_splits,
			"avg_recall": sum([r["recall"] for r in results]) / n_splits,
			"avg_f1": sum([r["f1"] for r in results]) / n_splits,
			"avg_time": sum([r["time"] for r in results]) / n_splits
		}
		self.save_results(
			{
				"precision": cv_results["avg_precision"],
				"recall": cv_results["avg_recall"],
				"f1": cv_results["avg_f1"],
				"time": cv_results["avg_time"]
			},
			kwargs
		)
		return cv_results

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

class EvalCRF(EnhancedEvalModel):
	def __init__(
			self,
			save_dir: str,
			results_dir: str,
			train_data_path: str,
			eval_data_path: str,
			load_existing_train_tokens: bool = False,
			load_existing_eval_tokens: bool = False,
			load_existing_train_pos: bool = False,
			load_existing_eval_pos: bool = False,
			**kwargs
	):
		super(EvalCRF, self).__init__(save_dir, results_dir, train_data_path, eval_data_path, load_existing_train_tokens,\
								load_existing_eval_tokens, load_existing_train_pos, load_existing_eval_pos, **kwargs)
		self.name = "CRF"

	def predict(self, **kwargs) -> None:
		self.model.process(
			data_path=self.data_path,
			tokens_path=os.path.join(self.save_dir, "data_tokens.json"),
			save_path=os.path.join(self.save_dir, "data_predictions.json"),
			pos_path=os.path.join(self.save_dir, "data_pos.json"),		
		)

	def evaluate(self, **kwargs) -> dict:
		if self.verbose: print("Instantiating CRF...")
		# delete previous model
		if os.path.exists(os.path.join(self.save_dir, "crf_0_0.crfsuite")):
			os.remove(os.path.join(self.save_dir, "crf_0_0.crfsuite"))
		self.model = crf.CRF(
			model_path=os.path.join(self.save_dir, "crf_0_0.crfsuite"), # TODO: allow multiple models
			trainer_params={k: v for k, v in kwargs.items() if k != "save_results"},
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

class EvalLSTM(EnhancedEvalModel):
	def __init__(
			self,
			save_dir: str,
			results_dir: str,
			train_data_path: str,
			eval_data_path: str,
			load_existing_train_tokens: bool = False,
			load_existing_eval_tokens: bool = False,
			load_existing_train_pos: bool = False,
			load_existing_eval_pos: bool = False,
			load_existing_train_lemmas: bool = False,
			load_existing_eval_lemmas: bool = False,
			**kwargs
	):
		super(EvalLSTM, self).__init__(save_dir, results_dir, train_data_path, eval_data_path, load_existing_train_tokens,\
								load_existing_eval_tokens, load_existing_train_pos, load_existing_eval_pos, **kwargs)
		if not load_existing_train_lemmas:
			if self.verbose: print("Precomputing training lemmas...")
			lstm.precompute_lemmas(
				tokens_path=os.path.join(self.save_dir, "train_data_tokens.json"),
				lemmas_path=os.path.join(self.save_dir, "train_data_lemmas.json"),
				verbose=self.verbose
			)
		if not load_existing_eval_lemmas:
			if self.verbose: print("Precomputing evaluation lemmas...")
			lstm.precompute_lemmas(
				tokens_path=os.path.join(self.save_dir, "data_tokens.json"),
				lemmas_path=os.path.join(self.save_dir, "data_lemmas.json"),
				verbose=self.verbose
			)



if __name__ == "__main__":
	with open(os.path.join(ROOT_DIR, "data", 'test_data.json'), 'r', encoding='utf8') as _f:
		test_data = json.load(_f)
	metric = EvalOfficial()
	p, r, f1 = metric.calc(test_data, test_data)
	print(f'Precision: {p}, Recall:{r}, F1:{f1}')
