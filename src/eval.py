import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from preprocessing import *
import negex
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

class EvalNegex(EvalOfficial):
	def __init__(
			self,
			save_dir: str,
			results_dir: str,
			data_path: str,
			load_existing: bool = False
	):
		super(EvalNegex, self).__init__()
		self.save_dir = save_dir
		self.results_dir = results_dir
		self.data_path = data_path
		self.data = self.load_test_data()
		if not load_existing:
			self.create_tokens()

	def load_test_data(self) -> List[dict]:
		with open(self.data_path, 'r', encoding='utf8') as _f:
			test_data = json.load(_f)
		return test_data
	
	def create_tokens(
			self,
			lemmatize: bool = False
	) -> List[dict]:
		sents = sent_tokenize_corpus(self.data)
		nlp_es, nlp_cat = load_nlps()
		tokens = tokenize_corpus(sents, nlp_es, nlp_cat)
		if lemmatize:
			tokens = lemmatize_corpus(tokens, nlp_es, nlp_cat)
		save_tokens(tokens, os.path.join(self.save_dir, "data_tokens.json"))
		return tokens
	
	def load_tokens(self) -> List[dict]:
		return load_tokens(os.path.join(self.save_dir, "data_tokens.json"))
	
	def predict(self, **kwargs) -> None:
		tokens = self.load_tokens()
		predictions = negex.process_data(tokens, max_context_size=kwargs.get("max_context_size", 5))
		negex.write_predictions(self.data, predictions, "data_predictions.json", dir=self.save_dir.split("/")[-1])
	
	def save_results(
			self,
			metrics: dict,
			hyperparameters: dict
	) -> None:
		save_results(
			os.path.join(self.results_dir, "results.json"),
			metrics,
			hyperparameters,
			"Negex"
		)

	def evaluate(self, **kwargs) -> dict:
		start_time = time()
		self.predict(**kwargs)
		total_time = time() - start_time
		with open(os.path.join(self.save_dir, "data_predictions.json"), 'r', encoding='utf8') as _f:
			predictions = json.load(_f)
		self.data = self.load_test_data()
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


if __name__ == "__main__":
	with open(os.path.join(ROOT_DIR, "data", 'test_data.json'), 'r', encoding='utf8') as _f:
		test_data = json.load(_f)
	metric = EvalOfficial()
	p, r, f1 = metric.calc(test_data, test_data)
	print(f'Precision: {p}, Recall:{r}, F1:{f1}')
