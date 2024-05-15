import re
import json
from tqdm import tqdm

# Function to read negations from a file and store them in a dictionary
def read_negations(file_path):
    negations = {}
    with open(file_path, 'r') as file:
        for line in file:
            negation, tag = line.strip().split('\t\t')
            negations[negation] = tag
    return negations

# Function to create a new tag object
def new_tag(start, end, labels, id):
    return {'value': {'start': start, 'end': end, 'labels': labels}, 'id': 'ent'+str(id), 'from_name': 'label', 'to_name': 'text', 'type': 'labels'}

# Function to find the end index of the forward scope of a tagged text
def fordward_scope(tagged_text, start_index):
    end_index = start_index
    tagger_checker = tagged_text[end_index]
    
    # Loop until "." is found or end of tagged_text
    while tagger_checker !="." and end_index != len(tagged_text)-1:
        end_index += 1
        tagger_checker = tagged_text[end_index]
    
    return end_index - 1

# Function to find the start index of the backward scope of a tagged text
def backward_scope(tagged_text, end_index):
    start_index = end_index
    tagger_checker = tagged_text[start_index]
    
    # Loop until "." is found or beginning of tagged_sentence
    while tagger_checker != "." and start_index != 0:
        start_index -= 1
        tagger_checker = tagged_text[start_index]

        if tagger_checker == ".":
            start_index += 1
    
    return start_index

# Function to tag negations in the text
def tag_negations(text, negations):
    result = []
    negations_of_the_text = dict()
    
    # Iterate over each negation in the negations dictionary
    for negation in negations:
        pattern = r'\b' + re.escape(negation) + r'\b'
        if re.search(pattern, text):
            tag = negations[negation]
            negations_of_the_text[negation] = tag
            
    i = 0
    
    # Iterate over each negation found in the text
    for negation in negations_of_the_text:
        pattern = r'\b' + re.escape(negation) + r'\b'
        tag = negations[negation]
        
        # Find all occurrences of the negation in the text
        negation_occurrences = re.finditer(pattern, text)
        
        # Iterate over each occurrence of the negation
        for match in negation_occurrences:
            # Add a new tag object for the negation occurrence
            if tag[-2] == "P":
                result.append(new_tag(match.start(), match.end(), ['UNC'], i))
            else:
                result.append(new_tag(match.start(), match.end(), ['NEG'], i))
            i += 1
        
        # Find all occurrences of the negation in the text
        negation_occurrences = re.finditer(pattern, text)
        
        # Iterate over each occurrence of the negation for scope tagging
        for match in negation_occurrences:
            # Determine the scope tag based on the negation tag
            if tag == '[PREN]':
                scope_tag = '[NSCO]'
                start_index = match.end()
                end_index = fordward_scope(text, start_index)
            elif tag == '[PREP]':
                scope_tag = '[USCO]'
                start_index = match.end()
                end_index = fordward_scope(text, start_index)
            elif tag == '[POST]':
                scope_tag = '[NSCO]'
                end_index = match.start()
                start_index = backward_scope(text, end_index)
            elif tag == '[POSP]':
                scope_tag = '[USCO]'
                end_index = match.start()
                start_index = backward_scope(text, end_index)
            
            # Add a new tag object for the scope of the negation
            result.append(new_tag(start_index, end_index, [scope_tag], i))
            i += 1
    
    return result

# Function to process text data
def process_text(data, negations, output_file):
    copy_json_object = data
    
    # Reset predictions result list for each data item
    for i in range(len(copy_json_object)):
        copy_json_object[i]["predictions"][0]["result"] = []
    
    # Iterate over each data item
    for i in tqdm(range(len(data))):
        text = data[i]["data"]["text"]
        result = tag_negations(text, negations)
        copy_json_object[i]["predictions"][0]["result"].extend(result)
    
    # Write updated data to output file
    with open(output_file, "w") as json_file:
        json.dump(copy_json_object, json_file)
