## AGAI_Assignment 1
## Author: Amy Moffatt

# This file will contain the logic for processing bigrams. You are responsible for editing this file. You are free to use LLM based code assistants such as GitHub Copilot to help you transform the notebook code from Module 1 and make it available for your API.

import spacy
import random
from collections import defaultdict

nlp = spacy.load("en_core_web_lg")

class BigramModel:
    def __init__(self, corpus):
        self.bigrams = defaultdict(list)
        for sentence in corpus:
            words = sentence.lower().split()
            for i in range(len(words) - 1):
                self.bigrams[words[i]].append(words[i + 1])

    def generate_text(self, start_word, length):
        word = start_word.lower()
        result = [word]
        for _ in range(length - 1):
            next_words = self.bigrams.get(word, [])
            if not next_words:
                break
            word = random.choice(next_words)
            result.append(word)
        return " ".join(result)

    def get_embedding(self, word):
        return nlp(word).vector.tolist()

    def calculate_similarity(self, word1, word2):
        return nlp(word1).similarity(nlp(word2))

