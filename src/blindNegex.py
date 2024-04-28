import re
import json

def read_negations(file_path):
	negations = {}
	with open(file_path, 'r') as file:
		for line in file:
			negation, tag = line.strip().split('\t\t')
			negations[negation] = tag
	return negations

def fordward_scope(tagged_sentence, start_index):
	end_index = start_index
	tagger_checker = tagged_sentence[end_index]
	
	# Loop until "[" is found or end of tagged_sentence
	while tagger_checker != "[" and end_index != len(tagged_sentence)-1:
		end_index += 1
		tagger_checker = tagged_sentence[end_index]
	
	possible_tag = tagger_checker
	
	if possible_tag == "[":
		i = 0
		# Loop until "]" is found
		while tagger_checker != "]":
			i += 1
			tagger_checker = tagged_sentence[end_index + i]
			possible_tag = possible_tag + tagger_checker
		
		# Check if the possible tag matches the pattern
		if (re.match(r"\[(PREN|PREP|POST|POSP|USCOPE|NSCOPE)\]", possible_tag)):
			end_index = end_index - 1
		else:
			# Recursive call until end_index == len(tagged_sentence)-1
			end_index = fordward_scope(tagged_sentence, end_index+1)
	
	return end_index

def backward_scope(tagged_sentence, end_index):
	start_index = end_index
	tagger_checker = tagged_sentence[start_index]
	# Loop until "]" is found or beginning of tagged_sentence
	while tagger_checker != "]" and start_index != 0:
		start_index -= 1
		tagger_checker = tagged_sentence[start_index]
	possible_tag = tagger_checker
	if possible_tag == "]":
		i = 0
		# Loop until "]" is found
		while tagger_checker != "[":
			i += 1
			tagger_checker = tagged_sentence[start_index - i]
			possible_tag = possible_tag + tagger_checker
		
		# Check if the possible tag matches the pattern
		if re.match(r"\[(PREN|PREP|POST|POSP|USCOPE|NSCOPE)\]", possible_tag[::-1]):
			start_index = start_index + 1
		else:
			# Recursive call until start_index == 0
			start_index = backward_scope(tagged_sentence, start_index-1)
	
	return start_index

def tag_negations(sentence, negations):
	tagged_sentence = sentence
	negations_of_the_sentence=dict()
	for negation in negations:
		pattern = r'\b' + re.escape(negation) + r'\b'
		if re.search(pattern, tagged_sentence):
			tag = negations[negation]
			tagged_negation=tag+''+negation+''+tag
			tagged_sentence=re.sub(r'\b' + re.escape(negation) + r'\b', tagged_negation, tagged_sentence)
			negations_of_the_sentence[negation]=tag
	
	for negation in negations_of_the_sentence:
		pattern = r'\b' + re.escape(negation) + r'\b'
		tag = negations[negation]
		tagged_negation=tag+''+negation+''+tag
		
		# Find all occurrences of the negation in the tagged_sentence
		negation_occurrences = re.finditer(pattern, tagged_sentence)

		# Iterate over each occurrence of the negation
		i=0
		for match in negation_occurrences:
			if tag == '[PREN]':
				scope_tag = '[NSCOPE]'
				start_index = match.start() + len(tagged_negation) - len(tag) + (len(scope_tag)*2*i)
				end_index = fordward_scope(tagged_sentence, start_index)
			elif tag == '[PREP]':
				scope_tag = '[USCOPE]'
				start_index = match.start() + len(tagged_negation) - len(tag) + (len(scope_tag)*2*i)
				end_index = fordward_scope(tagged_sentence, start_index)
			elif tag == '[POST]':
				scope_tag = '[NSCOPE]'
				end_index = match.start() - len(tag) + (len(scope_tag)*2*i)
				start_index = backward_scope(tagged_sentence, end_index)
			elif tag == '[POSP]':
				scope_tag = '[USCOPE]'
				end_index = match.start() - len(tag) + (len(scope_tag)*2*i)
				start_index = backward_scope(tagged_sentence, end_index)
		
			tagged_sentence=tagged_sentence.replace(tagged_sentence[start_index:end_index], scope_tag+''+tagged_sentence[start_index:end_index]+''+scope_tag)
			i+=1
	return tagged_sentence

def process_text(data, negations, output_file):
    copy_json_object = data
    
    # Reset predictions result list for each data item
    for i in range(len(copy_json_object)):
        copy_json_object[i]["predictions"][0]["result"] = []
    
    # Iterate over each data item
    for i in range(len(data)):
        text = data[i]["data"]["text"]
        result = tag_negations(text, negations)
        copy_json_object[i]["predictions"][0]["result"].extend(result)
    
    # Write updated data to output file
    with open(output_file, "w") as json_file:
        json.dump(copy_json_object, json_file)
