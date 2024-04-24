import json
import os
import pprint
from langdetect import detect, DetectorFactory
import nltk
import spacy
from spacy.tokenizer import Tokenizer
import re
from tqdm import tqdm
import numpy as np
from typing import Any, Literal, List, Dict, Union, Tuple, Optional
from unidecode import unidecode
from copy import deepcopy

def load_nlps() -> Any:
	"""
	Load spacy models for Spanish and Catalan.
		Returns both models in a tuple.

	Remember to download them first with:
	- $ python -m spacy download es_core_news_sm
	- $ python -m spacy download ca_core_news_sm
	"""
	nlp_es = spacy.load("es_core_news_sm")
	nlp_ca = spacy.load("ca_core_news_sm")
	return nlp_es, nlp_ca

def detect_lang(
		text: str
) -> Literal["es", "ca"]:
	"""
	Detect the language of a text.
		Returns either "es" or "ca".
	"""
	DetectorFactory.seed = 42
	try:
		lang = detect(text)
		if lang not in ["es", "ca"]:
			return "es"
		return lang
	except:
		return "es"

def sent_tokenize(
		text: str
) -> List[Dict[str, Any]]:
	"""
	Tokenize a text into sentences.
		Returns a list of dictionaries with keys "text" and "span".
	"""
	sents = nltk.sent_tokenize(text, language="spanish") # assuming for catalan will be similar
	text_cpy = text
	spans = []
	l = 0
	for sent in sents:
		pos = text_cpy.find(sent)
		span = (pos, pos + len(sent))
		span = (0, span[1]) # keep from start in case space was removed during tokenization
		sent = text_cpy[span[0]:span[1]]
		text_cpy = text_cpy[span[1]:]
		span = (l + span[0], l + span[1])
		spans.append({"text":sent, "span":span})
		l = span[1]
	if text_cpy:
		last_sent, last_span = spans[-1]["text"], spans[-1]["span"]
		last_span = (last_span[0], len(text))
		last_sent = last_sent + text_cpy
		spans[-1] = {"text":last_sent, "span":last_span}
	return spans

def sent_tokenize_corpus(
		corpus: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
	"""
	Tokenize a corpus into sentences.
		Returns a list of lists of dictionaries with keys "text", "span" and "lang".
	"""
	corpus_sents = []
	for d in tqdm(corpus):
		text = d["data"]["text"]
		spans = sent_tokenize(text)
		for i, s in enumerate(spans):
			sent, span = s["text"], s["span"]
			assert sent == text[span[0]:span[1]], "Error in span"
			lang = detect_lang(sent)
			spans[i]["lang"] = lang
		corpus_sents.append(spans)
	return corpus_sents

def tokenize_corpus(
		corpus_sents: List[List[Dict[str, Any]]],
		nlp_es: Any,
		nlp_ca: Any,
		remove_asterisk: bool = True,
		remove_punctuation: bool = True,
		remove_spaces: bool = True,
		replace_numbers: Optional[str] = None,
) -> List[List[Dict[str, np.ndarray, str]]]:
	"""
	Tokenize a corpus into tokens.
	Parameters:
	- corpus_sents: list of lists of dictionaries with keys "text", "span" and "lang".
	- nlp_es: spacy model for Spanish.
	- nlp_ca: spacy model for Catalan.
	- remove_asterisk: whether to remove asterisks or not.
	- remove_punctuation: whether to remove punctuation or not.
	- remove_spaces: whether to remove spaces or not.
	- replace_numbers: string to replace numbers (integers, floats, dates...) with.
		If None, numbers are kept.
	
	Returns a list of lists of dictionaries with keys "tokens", "spans" and "lang".
	"""
	tokens = []
	for d in tqdm(corpus_sents):
		d_tokens = []
		for sent in d:
			text = sent["text"]
			span = sent["span"]
			lang = sent["lang"]
			if lang == "es":
				doc = nlp_es.tokenizer(text)
			else:
				doc = nlp_ca.tokenizer(text)
			_tokens = []
			spans = []
			asterisk_len = 0
			asterisk_pos = 0
			for token in doc:
				word = token.text
				# handle asterisks
				if word == "*":
					if remove_asterisk: continue
					else:
						if asterisk_len == 0:
							asterisk_pos = token.idx
						asterisk_len += 1
						continue
				else:
					if asterisk_len > 0:
						_tokens.append("<HIDDEN>")
						spans.append([span[0] + asterisk_pos, span[0] + asterisk_pos + asterisk_len])
						asterisk_len = 0
				# handle numbers
				if re.fullmatch(r'(\d+[.,:/-]?)+', word) and replace_numbers:
					word = replace_numbers
				# remove punctuation
				elif remove_punctuation:
					word = re.sub(r'[^\w\s]|[ºª]', '', word)
				# remove spaces
				if remove_spaces:
					word = word.strip()
				# unidecode in case of weird characters
				word = unidecode(word)
				# append token
				if word:
					_tokens.append(word)
					spans.append([span[0] + token.idx, span[0] + token.idx + len(token.text)])
			sent_tokens = {"tokens": np.array(_tokens), "spans": np.array(spans), "lang": lang}
			d_tokens.append(sent_tokens)
		tokens.append(d_tokens)
	return tokens

def lemmatize_corpus(
		tokens: List[List[Dict[str, np.ndarray, str]]],
		nlp_es: Any,
		nlp_ca: Any,
) -> List[List[Dict[str, np.ndarray, str]]]:
	"""
	Lemmatize a corpus.
		Returns a list of lists of dictionaries with keys "lemmas", "spans" and "lang".
	"""
	def custom_tokenizer(nlp):
		return Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)

	# print("Preparing tokenizers...")
	nlp_es_custom = deepcopy(nlp_es)
	nlp_es_custom.tokenizer = custom_tokenizer(nlp_es_custom)
	nlp_ca_custom = deepcopy(nlp_ca)
	nlp_ca_custom.tokenizer = custom_tokenizer(nlp_ca_custom)

	# print("Lemmatizing...")
	tokens_lemmatized = []
	for d in tqdm(tokens):
		d_tokens = []
		for sent in d:
			lang = sent["lang"]
			# lemmatization works better with the whole sentence
			if lang == "es":
				doc = nlp_es_custom(" ".join(sent["tokens"]))
			else:
				doc = nlp_ca_custom(" ".join(sent["tokens"]))
			lemmas = [token.lemma_ for token in doc]
			assert len(lemmas) == len(sent["tokens"]), f"\n{lemmas}\n{sent['tokens']}"
			sent_tokens = {"tokens": np.array(lemmas), "spans": sent["spans"], "lang": lang}
			d_tokens.append(sent_tokens)
		tokens_lemmatized.append(d_tokens)
	
	return tokens_lemmatized

def save_tokens(
		tokens: List[List[Dict[str, np.ndarray, str]]],
		path: str,
) -> None:
	"""
	Save tokens to a JSON file.
	"""
	tokens = [
		{
			"tokens": sent["tokens"].tolist(),
			"spans": sent["spans"].tolist(),
			"lang": sent["lang"]
		}
		for d in tokens for sent in d
	]
	with open(path, 'w') as f:
		json.dump(tokens, f)

def load_tokens(
		path
) -> List[Dict[str, Union[np.ndarray, str]]]:
	"""
	Load tokens from a JSON file.
	"""
	with open(path, 'r') as f:
		tokens = json.load(f)
	tokens = [
		{
			"tokens": np.array(sent["tokens"]),
			"spans": np.array(sent["spans"]),
			"lang": sent["lang"]
		}
		for sent in tokens
	]
	return tokens
