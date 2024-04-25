import re

def sortRules (ruleList):
    """Return sorted list of rules.
    
    Rules should be in a tab-delimited format: 'rule\t\t[four letter negation tag]'
    Sorts list of rules descending based on length of the rule, 
    splits each rule into components, converts pattern to regular expression,
    and appends it to the end of the rule. """
    ruleList.sort(key=len, reverse=True)  # Sort rules by length in descending order
    sortedList = []  # Initialize an empty list to store sorted rules
    for rule in ruleList:
        s = rule.strip().split('\t')  # Split each rule by tabs
        splitTrig = s[0].split()  # Split the trigger term into individual words
        trig = r'\s+'.join(splitTrig)  # Join trigger words with regex for whitespace
        pattern = r'\b(' + trig + r')\b'  # Create regex pattern for whole word match
        s.append(re.compile(pattern, re.IGNORECASE))  # Compile regex pattern
        sortedList.append(s)  # Append rule with compiled regex to sorted list
    return sortedList

class negTagger(object):
    '''Take a sentence and tag negation terms and negated phrases.
    
    Keyword arguments:
    sentence -- string to be tagged
    phrases  -- list of phrases to check for negation
    rules    -- list of negation trigger terms from the sortRules function
    negP     -- tag 'possible' terms as well (default = True)    '''
    def __init__(self, sentence = '', phrases = None, rules = None, 
                 negP = True):
        # Initialize attributes
        self.__sentence = sentence
        self.__phrases = phrases
        self.__rules = rules
        self.__negTaggedSentence = ''
        self.__scopesToReturn = []
        self.__negationFlag = None
        
        filler = '_'  # Placeholder for space
        
        # Loop through each rule and process the sentence
        for rule in self.__rules:
            reformatRule = re.sub(r'\s+', filler, rule[0].strip())  # Replace spaces with filler in rule
            self.__sentence = rule[3].sub(' ' + rule[2].strip()
                                          + reformatRule
                                          + rule[2].strip() + ' ', self.__sentence)
        
        # Loop through each phrase and replace in the sentence
        for phrase in self.__phrases:
            phrase = re.sub(r'([.^$*+?{\\|()[\]])', r'\\\1', phrase)  # Escape special characters
            splitPhrase = phrase.split()  # Split phrase into individual words
            joiner = r'\W+'  # Regex for non-word characters
            joinedPattern = r'\b' + joiner.join(splitPhrase) +  r'\b'  # Create regex pattern
            reP = re.compile(joinedPattern, re.IGNORECASE)  # Compile regex pattern
            m = reP.search(self.__sentence)  # Search for pattern in sentence
            if m:
                # Replace matched phrase with tagged phrase
                self.__sentence = self.__sentence.replace(m.group(0), '[PHRASE]'
                                                          + re.sub(r'\s+', filler, m.group(0).strip())
                                                          + '[PHRASE]')
                
    
#        Exchanges the [PHRASE] ... [PHRASE] tags for [NEGATED] ... [NEGATED] 
#        based on PREN, POST rules and if negPoss is set to True then based on 
#        PREP and POSP, as well.
#        Because PRENEGATION [PREN} is checked first it takes precedent over
#        POSTNEGATION [POST]. Similarly POSTNEGATION [POST] takes precedent over
#        POSSIBLE PRENEGATION [PREP] and [PREP] takes precedent over POSSIBLE 
#        POSTNEGATION [POSP].
              
        # Initialize flags for various types of negations
        overlapFlag = 0
        prenFlag = 0
        postFlag = 0
        prePossibleFlag = 0
        postPossibleFlag = 0
        
        sentenceTokens = self.__sentence.split()  # Tokenize sentence
        sentencePortion = ''
        aScopes = []  # List to store negation scopes
        
        sb = []  # List to build sentence
        
        #check for [PREN]
        for i in range(len(sentenceTokens)):
            if sentenceTokens[i][:6] == '[PREN]':
                prenFlag = 1
                overlapFlag = 0

            if sentenceTokens[i][:6] in ['[CONJ]', '[PSEU]', '[POST]', '[PREP]', '[POSP]']:
                overlapFlag = 1
            
            if i+1 < len(sentenceTokens):
                if sentenceTokens[i+1][:6] == '[PREN]':
                    overlapFlag = 1
                    if sentencePortion.strip():
                        aScopes.append(sentencePortion.strip())
                    sentencePortion = ''
            
            if prenFlag == 1 and overlapFlag == 0:
                sentenceTokens[i] = sentenceTokens[i].replace('[PHRASE]', '[NEGATED]')
                sentencePortion = sentencePortion + ' ' + sentenceTokens[i]
            
            sb.append(sentenceTokens[i])
        
        if sentencePortion.strip():
            aScopes.append(sentencePortion.strip())
        
        sentencePortion = ''
        sb.reverse()
        sentenceTokens = sb
        
        sb2 = []# List to build sentence for post-negation

        # Check for [POST]
        for i in range(len(sentenceTokens)):
            if sentenceTokens[i][:6] == '[POST]':
                postFlag = 1
                overlapFlag = 0

            if sentenceTokens[i][:6] in ['[CONJ]', '[PSEU]', '[PREN]', '[PREP]', '[POSP]']:
                overlapFlag = 1
            
            if i+1 < len(sentenceTokens):
                if sentenceTokens[i+1][:6] == '[POST]':
                    overlapFlag = 1
                    if sentencePortion.strip():
                        aScopes.append(sentencePortion.strip())
                    sentencePortion = ''
            
            if postFlag == 1 and overlapFlag == 0:
                sentenceTokens[i] = sentenceTokens[i].replace('[PHRASE]', '[NEGATED]')
                sentencePortion = sentenceTokens[i] + ' ' + sentencePortion
            
            sb2.insert(0, sentenceTokens[i])
        
        if sentencePortion.strip():
            aScopes.append(sentencePortion.strip())
        
        sentencePortion = ''
        self.__negTaggedSentence = ' '.join(sb2)  # Join sentence parts
        
        # If negP is True, check for possible negations
        if negP:
            sentenceTokens = sb2

            sb3 = [] # List to build sentence for possible pre-negation
            
            # Check for [PREP]
            for i in range(len(sentenceTokens)):
                if sentenceTokens[i][:6] == '[PREP]':
                    prePossibleFlag = 1
                    overlapFlag = 0

                if sentenceTokens[i][:6] in ['[CONJ]', '[PSEU]', '[POST]', '[PREN]', '[POSP]']:
                    overlapFlag = 1
            
                if i+1 < len(sentenceTokens):
                    if sentenceTokens[i+1][:6] == '[PREP]':
                        overlapFlag = 1
                        if sentencePortion.strip():
                            aScopes.append(sentencePortion.strip())
                        sentencePortion = ''
            
                if prePossibleFlag == 1 and overlapFlag == 0:
                    sentenceTokens[i] = sentenceTokens[i].replace('[PHRASE]', '[POSSIBLE]')
                    sentencePortion = sentencePortion + ' ' + sentenceTokens[i]
            
                sb3 = sb3 + ' ' + sentenceTokens[i]
        
            if sentencePortion.strip():
                aScopes.append(sentencePortion.strip())
            
            sentencePortion = ''
            sb3.reverse()
            sentenceTokens = sb3 
            sb4 = []
            # Check for [POSP]
            for i in range(len(sentenceTokens)):
                if sentenceTokens[i][:6] == '[POSP]':
                    postPossibleFlag = 1
                    overlapFlag = 0

                if sentenceTokens[i][:6] in ['[CONJ]', '[PSEU]', '[PREN]', '[PREP]', '[POST]']:
                    overlapFlag = 1
            
                if i+1 < len(sentenceTokens):
                    if sentenceTokens[i+1][:6] == '[POSP]':
                        overlapFlag = 1
                        if sentencePortion.strip():
                            aScopes.append(sentencePortion.strip())
                        sentencePortion = ''
            
                if postPossibleFlag == 1 and overlapFlag == 0:
                    sentenceTokens[i] = sentenceTokens[i].replace('[PHRASE]', '[POSSIBLE]')
                    sentencePortion = sentenceTokens[i] + ' ' + sentencePortion
            
                sb4.insert(0, sentenceTokens[i])
        
            if sentencePortion.strip():
                aScopes.append(sentencePortion.strip())
            
            self.__negTaggedSentence = ' '.join(sb4)
            
        if '[NEGATED]' in self.__negTaggedSentence:
            self.__negationFlag = 'negated'
        elif '[POSSIBLE]' in self.__negTaggedSentence:
            self.__negationFlag = 'possible'
        else:
            self.__negationFlag = 'affirmed'
        
        self.__negTaggedSentence = self.__negTaggedSentence.replace(filler, ' ')
        
        for line in aScopes:
            tokensToReturn = []
            thisLineTokens = line.split()
            for token in thisLineTokens:
                if token[:6] not in ['[PREN]', '[PREP]', '[POST]', '[POSP]']:
                    tokensToReturn.append(token)
            self.__scopesToReturn.append(' '.join(tokensToReturn))

    def getNegTaggedSentence(self):
        return self.__negTaggedSentence
    def getNegationFlag(self):
        return self.__negationFlag
    def getScopes(self):
        return self.__scopesToReturn
    
    def __str__(self):
        text = self.__negTaggedSentence
        text += '\t' + self.__negationFlag
        text += '\t' + '\t'.join(self.__scopesToReturn)
