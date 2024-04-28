from preprocessing import *
import blindNegex
import os
import cutext

def write_tokens_txt(
		tokens: np.ndarray,
		path: str
) -> None:
	directory = os.path.dirname(path)
	if not os.path.exists(directory):
		os.makedirs(directory)
	with open(path, "w") as f:
		f.write(" ".join(tokens))

def read_terms_txt(
		path: str
) -> List[str]:
	with open(path, "r") as f:
		terms = [line.split("|")[0] for line in f]
	return terms

def get_medical_terms(
		tokens: np.ndarray,
		cutext_path: Optional[str] = None
) -> List[List[str]]:
	write_tokens_txt(tokens, os.path.join(ROOT_DIR, "temp", "cutext_in.txt"))
	cutext.process_input(cutext_path)
	terms = read_terms_txt(os.path.join(ROOT_DIR, "temp", "hsimpli.txt"))
	terms = [term.split(" ") for term in terms]
	return terms

def prepare_negation_speculation(
		negation_speculation_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	negations = blindNegex.read_negations(negation_speculation_path)
	words, tags = zip(*negations.items())
	negation_words = np.array([word for i, word in enumerate(words) if tags[i] in ("[PREN]", "[POST]")])
	speculation_words = np.array([word for i, word in enumerate(words) if tags[i] in ("[PREP]", "[POSP]")])
	maps = {"[PREN]": 1, "[POST]": -1, "[PREP]": 1, "[POSP]": -1}
	tags = np.array([maps[tag] for tag in tags if tag in maps])
	negation_dirs = tags[np.where(np.isin(words, negation_words))[0]]
	speculation_dirs = tags[np.where(np.isin(words, speculation_words))[0]]
	return negation_words, negation_dirs, speculation_words, speculation_dirs

def get_indices(
		tokens: np.ndarray,
		terms: List[List[str]],
		negation_words: np.ndarray,
		negation_dirs: np.ndarray,
		speculation_words: np.ndarray,
		speculation_dirs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	neg_bin = np.isin(tokens, negation_words)
	neg_idx = np.where(neg_bin)[0]
	neg_dir = negation_dirs[neg_idx]
	unc_bin = np.isin(tokens, speculation_words)
	unc_idx = np.where(unc_bin)[0]
	unc_dir = speculation_dirs[unc_idx]
	terms_bin = np.zeros_like(tokens, dtype=int)
	for term in terms:
		for i in range(len(tokens) - len(term) + 1):
			if np.array_equal(tokens[i:i+len(term)], term):
				terms_bin[i:i+len(term)] = 1
	terms_idx = np.where(terms_bin)[0]
	return neg_idx, neg_dir, unc_idx, unc_dir, terms_idx, terms_bin

def _find_regions(data):
	"""Find the start and end indices of contiguous regions of 1s"""
	diff = np.diff(data)
	starts = np.where(diff == 1)[0] + 1
	if data[0] == 1:
		starts = np.r_[0, starts]
	ends = np.where(diff == -1)[0] + 1
	if data[-1] == 1:
		ends = np.r_[ends, len(data)]
	return list(zip(starts, ends))

def _combine_regions(a, b):
	"""Combine regions from two binary arrays based on overlap criteria"""
	a_regions = _find_regions(a)
	b_regions = _find_regions(b)
	result = np.zeros_like(a, dtype=int)

	for a_start, a_end in a_regions:
		for b_start, b_end in b_regions:
			if (a_start < b_end and a_end > b_start):
				start = min(a_start, b_start)
				end = max(a_end, b_end)
				result[start:end] = 1
	return result

def get_contexts_bin(
		terms_bin: np.ndarray,
		terms_idx: np.ndarray,
		neg_idx: np.ndarray,
		neg_dir: np.ndarray
) -> List[np.ndarray]:
	terms_idx = np.repeat(terms_idx[None, :], len(neg_idx), axis=0)
	context_dis = terms_idx - neg_idx[:, None]
	context_bin = np.sign(context_dis) == neg_dir[:, None]
	near_bin = np.abs(context_dis) <= 5
	context_near_bin = context_bin & near_bin
	context_near = [terms_idx[i][context_near_bin[i]] for i in range(len(neg_idx))]
	context_end = np.array([context_near[i].max() if neg_dir[i] else context_near[i].min() for i in range(len(neg_idx))])
	context_idx = [np.arange(neg_idx[i]+neg_dir[i], context_end[i]+neg_dir[i], neg_dir[i]) for i in range(len(neg_idx))]
	context_bin = [np.zeros_like(terms_bin) for _ in context_idx]
	[np.put(ar, indices, 1) for ar, indices in zip(context_bin, context_idx)]
	context_bin = [_combine_regions(terms_bin, context_bin[i].astype(int)) for i in range(len(neg_idx))]
	return context_bin

def get_contexts_span(
		spans: np.ndarray,
		contexts_bin: List[np.ndarray]
) -> List[np.ndarray]:
	tokens_span = [spans[np.where(indices)[0]] for indices in contexts_bin]
	contexts_span = [[span[0][0], span[-1][1]] for span in tokens_span]
	return contexts_span
