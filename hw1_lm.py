########################################
## CS447 Natural Language Processing  ##
##           Homework 1               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Develop a smoothed n-gram language model and evaluate it on a corpus
##
import os.path
import sys
import random
import math
from operator import itemgetter
from collections import defaultdict
#----------------------------------------
#  Data input 
#----------------------------------------

# Read a text file into a corpus (list of sentences (which in turn are lists of words))
# (taken from nested section of HW0)
def readFileToCorpus(f):
    """ Reads in the text file f which contains one sentence per line.
    """
    if os.path.isfile(f):
        file = open(f, "r") # open the input file in read-only mode
        i = 0 # this is just a counter to keep track of the sentence numbers
        corpus = [] # this will become a list of sentences
        print("Reading file ", f)
        for line in file:
            i += 1
            sentence = line.split() # split the line into a list of words
            #append this lis as an element to the list of sentences
            corpus.append(sentence)
            if i % 1000 == 0:
    	#print a status message: str(i) turns int i into a string
    	#so we can concatenate it
                sys.stderr.write("Reading sentence " + str(i) + "\n")
        #endif
    #endfor
        return corpus
    else:
    #ideally we would throw an exception here, but this will suffice
        print("Error: corpus file ", f, " does not exist")
        sys.exit() # exit the script
    #endif
#enddef


# Preprocess the corpus to help avoid sess the corpus to help avoid sparsity
def preprocess(corpus):
    #find all the rare words
    freqDict = defaultdict(int)
    for sen in corpus:
	    for word in sen:
	       freqDict[word] += 1
	#endfor
    #endfor

    #replace rare words with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if freqDict[word] < 2:
                sen[i] = UNK
	    #endif
	#endfor
    #endfor

    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor
    
    return corpus
#enddef

def preprocessTest(vocab, corpus):
    #replace test words that were unseen in the training with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if word not in vocab:
                sen[i] = UNK
	    #endif
	#endfor
    #endfor
    
    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor

    return corpus
#enddef

# Constants 
UNK = "UNK"     # Unknown word token
start = "<s>"   # Start-of-sentence token
end = "</s>"    # End-of-sentence-token


#--------------------------------------------------------------
# Language models and data structures
#--------------------------------------------------------------

# Parent class for the three language models you need to implement
class LanguageModel:
    # Initialize and train the model (ie, estimate the model's underlying probability
    # distribution from the training corpus)
    def __init__(self, corpus):

        self.uni_word_count= defaultdict(int)
        self.bi_word_count = defaultdict(int)
        self.N = 0

        prev = start

        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.uni_word_count[word] += 1
                self.bi_word_count[word + " " + prev] += 1
                prev = word
                self.N += 1
        
        print("""Your task is to implement five kinds of n-gram language models:
      a) an (unsmoothed) unigram model (UnigramModel)
      b) a unigram model smoothed using Laplace smoothing (SmoothedUnigramModel)
      c) an unsmoothed bigram model (BigramModel)
      """)
    #enddef
    
    # Generate a sentence by drawing words according to the 
    # model's probability distribution
    # Note: think about how to set the length of the sentence 
    #in a principled way
    def generateSentence(self):
        print("Implement the generateSentence method in each subclass")
        return "mary had a little lamb ."
    #emddef

    # Given a sentence (sen), return the probability of 
    # that sentence under the model
    def getSentenceProbability(self, sen):
        print("Implement the getSentenceProbability method in each subclass")
        return 0.0
    #enddef

    # Given a corpus, calculate and return its perplexity 
    #(normalized inverse log probability)
    def getCorpusPerplexity(self, corpus):
        print("Implement the getCorpusPerplexity method")
        return 0.0
    #enddef

    # Given a file (filename) and the number of sentences, generate a list
    # of sentences and write each to file along with its model probability.
    # Note: you shouldn't need to change this method
    def generateSentencesToFile(self, numberOfSentences, filename):
        filePointer = open(filename, 'w+')
        for i in range(0,numberOfSentences):
            sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)
            stringGenerated = str(prob) + " " + " ".join(sen) 
            
	#endfor
    #enddef
#endclass

# Unigram language model
class UnigramModel(LanguageModel):
    def __init__(self, corpus):
        self.uni_word_count= defaultdict(int)
        self.N = 0

        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.uni_word_count[word] += 1
                self.N += 1
        print("Subtask: implement the unsmoothed unigram language model")
    #endddef

    #define probability function for unigram model
    def prob(self, word):
        return self.uni_word_count[word]/self.N
    #enddef
    
    #define draw word function
    def draw(self):
        r = random.random()
        for word in self.uni_word_count.keys():
            r -= self.prob(word)
            if r <= 0:
                return word
    
    #define generating sentence function
    def generateSentence(self):
        sentence = []
        cur = start

        while cur != end:
            sentence.append(cur)
            cur = self.draw()
        
        sentence.append(end)
        return sentence

    #define sentence prob function
    def getSentenceProbability(self,sen):
        p = 0
        for word in sen:
            if word == start:
                continue
            if self.prob(word) == 0:
                return 0
            p += math.log(self.prob(word))
        return math.exp(p)
    
    #define preplexity funciton
    def getCorpusPerplexity(self, corpus):
        perplexity = 0
        count = 0
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                if self.prob(word) == 0:
                    return math.inf
                perplexity += math.log(self.prob(word))
                count += 1
        
        perplexity = math.exp(-perplexity / count)
        return perplexity

#endclass

#Smoothed unigram language model (use laplace for smoothing)
class SmoothedUnigramModel(LanguageModel):
    def __init__(self, corpus):
        self.uni_word_count= defaultdict(int)
        self.N = 0

        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.uni_word_count[word] += 1
                self.N += 1
        print("Subtask: implement the smoothed unigram language model")
    #endddef

    def prob(self, word):
        return (self.uni_word_count[word] + 1)/(self.N + len(self.uni_word_count))

    def draw(self):
        r = random.random()
        for word in self.uni_word_count.keys():
            r -= self.prob(word)
            if r <= 0:
                return word
    
    def generateSentence(self):
        sentence = []
        cur = start

        while cur != end:
            sentence.append(cur)
            cur = self.draw()
        
        sentence.append(end)
        return sentence
    
    def getSentenceProbability(self,sen):
        p = 0
        for word in sen:
            if word == start:
                continue
            p += math.log(self.prob(word))
        return math.exp(p)
    
        #define preplexity funciton
    def getCorpusPerplexity(self, corpus):
        perplexity = 0
        count = 0
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                if self.prob(word) == 0:
                    return math.inf
                perplexity += math.log(self.prob(word))
                count += 1
        
        perplexity = math.exp(-perplexity / count)
        return perplexity
#endclass

# Unsmoothed bigram language model
class BigramModel(LanguageModel):
    def __init__(self, corpus):
        self.uni_word_count= defaultdict(int)
        self.bi_word_count = defaultdict(int)
        self.N = 0

        prev = start

        for sen in corpus:
            for word in sen:
                self.uni_word_count[word] += 1
                self.bi_word_count[word + " " + prev] += 1
                prev = word
                self.N += 1
        print("Subtask: implement the unsmoothed bigram language model")
    #endddef

    #define prob function
    def prob(self, word, prev):
        return self.bi_word_count[word + " " + prev]/self.uni_word_count[prev]
    #enddef

    #define draw word function
    def draw(self, prev):
        r = random.random()
        for word in self.uni_word_count.keys():
            r -= self.prob(word, prev)
            if r <= 0:
                return word

    #define generating sentence function
    def generateSentence(self):
        sentence = []
        cur = start
        prev = start

        while cur != end:
            sentence.append(cur)
            prev = cur
            cur = self.draw(cur)
        
        sentence.append(end)
        return sentence
    
    #define get sentence prob funtion
    def getSentenceProbability(self,sen):
        p = 0
        prev = start

        for word in sen:
            if word == start:
                continue
            if self.prob(word,prev) == 0:
                return 0
            p += math.log(self.prob(word,prev))
            prev = word
        
        return math.exp(p)
    
        #define preplexity funciton
    def getCorpusPerplexity(self, corpus):
        perplexity = 0
        count = 0
        for sen in corpus:
            prev = start
            for word in sen:
                if word == start:
                    continue
                if self.prob(word,prev) == 0:
                    return math.inf
                perplexity += math.log(self.prob(word,prev))
                prev = word
                count += 1
        
        perplexity = math.exp(-perplexity / count)
        return perplexity
#endclass

#-------------------------------------------
# The main routine
#-------------------------------------------
if __name__ == "__main__":
    #read your corpora
    trainCorpus = readFileToCorpus('train.txt')
    trainCorpus = preprocess(trainCorpus)
    uni =   UnigramModel(trainCorpus)
    smooth = SmoothedUnigramModel(trainCorpus)
    bi = BigramModel(trainCorpus)

    uni.generateSentencesToFile(20,'unigram_output.txt')
    smooth.generateSentencesToFile(20,'smooth_unigram_output.txt')
    bi.generateSentencesToFile(20,'bigram_output.txt')
    
    posTestCorpus = readFileToCorpus('pos_test.txt')
    negTestCorpus = readFileToCorpus('neg_test.txt')

    vocab = set()
    for sen in trainCorpus:
        for word in sen:
            vocab.add(word)
    # Please write the code to create the vocab over here before the function preprocessTest
    print("""Task 0: create a vocabulary(collection of word types) for the train corpus""")
    posTestCorpus = preprocessTest(vocab, posTestCorpus)
    negTestCorpus = preprocessTest(vocab, negTestCorpus)

    print ("POSTIVE")
    print (uni.getCorpusPerplexity(posTestCorpus))
    print (smooth.getCorpusPerplexity(posTestCorpus))
    print (bi.getCorpusPerplexity(posTestCorpus))

    print ("NEGTIVE")
    print (uni.getCorpusPerplexity(negTestCorpus))
    print (smooth.getCorpusPerplexity(negTestCorpus))
    print (bi.getCorpusPerplexity(negTestCorpus))


