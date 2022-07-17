from dbdeep import configure as con
from kss import split_sentences

import time      
import re

class Processing():  
    def __init__(self):
        None
        
    def get_token_position(self, sentence_org, tag_list):
        content_ = sentence_org
        position = 0
        loc_list = []
        for word, pos in tag_list:
            loc = content_.find(word)
            if loc != -1:
                position += loc
                content_ = content_[loc:]
                start = position
                end = position + len(word)
                org_word = sentence_org[start:end]
            else:
                start = 0
                end = 0
                org_word = word
            loc_list.append((org_word, pos, (start, end)))
        return loc_list
        
    def language_detector(self, sentence):
        len_ko = len(re.sub("[^가-힇]", "", sentence))
        len_en = len(re.sub("[^a-zA-Z]", "", sentence))
        return "ko" if len_ko >= len_en else "en"

    def iteration_remover(self, sentence, replace_char="."):
        pattern_list = [r'(.)\1{5,}', r'(..)\1{5,}', r'(...)\1{5,}']
        for pattern in pattern_list:
            matcher= re.compile(pattern)
            iteration_term_list = [match.group() for match in matcher.finditer(sentence)]
            for iteration_term in iteration_term_list:
                sentence = sentence.replace(iteration_term, iteration_term[:pattern.count(".")] + replace_char*(len(iteration_term)-pattern.count(".")))
        return sentence
    
    def replacer(self, sentence):
        patterns = [
            (r'won\'t', 'will not'),
            (r'can\'t', 'cannot'),
            (r'i\'m', 'i am'),
            (r'ain\'t', 'is not'),
            (r'(\w+)\'ll', '\g<1> will'),
            (r'(\w+)n\'t', '\g<1> not'),
            (r'(\w+)\'ve', '\g<1> have'),
            (r'(\w+)\'s', '\g<1> is'),
            (r'(\w+)\'re', '\g<1> are'),
            (r'(\w+)\'d', '\g<1> would'),
        ]
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
        for (pattern, repl) in self.patterns:
            sentence = re.sub(pattern, repl, sentence)
        return sentence