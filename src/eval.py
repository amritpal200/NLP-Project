import json
import random
import numpy as np
import nltk
import os
from sklearn.metrics import precision_score, recall_score, f1_score

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


if __name__ == "__main__":
	with open(os.path.join(ROOT_DIR, "data", 'test_data.json'), 'r', encoding='utf8') as _f:
		test_data = json.load(_f)
	metric = EvalOfficial()
	p, r, f1 = metric.calc(test_data, test_data)
	print(f'Precision: {p}, Recall:{r}, F1:{f1}')
