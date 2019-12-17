#!/usr/bin/python
#coding = utf-8

"""
    This file defines the class of N-gram model and creates the N-gram dictionary.
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class NGram(object):
    def __init__(self, n):
        # n is the order of n-gram language model
        self.n = n
        self.n_gram = {}
        self.numTotal = 0

    '''
        The function of scaning the sentence and extracting the ngram and updating their frequence.
        @param sentence list{str}
        @return none
    '''

    def scan(self, sentence, outputfile):
        for line in sentence:
            self.ngram(line.split())
        try:
            fip = open(outputfile, "a", encoding='utf-8')
        except BaseException:
            print >> sys.stderr, "failed to open data.uni"

        for i in self.n_gram:
            fip.write("%s/%d\n" % (i, self.n_gram[i]))
        # fip.close()

    '''
        The function of dividing the ngram and caluclating the number of the ngram.
        Results are saved the param 'self.n_gram'.
        @param words list{str}
        @return none
    '''

    def ngram(self, words):
        # unigram
        if self.n == 1:
            for word in words:
                word = word + " "
                if word not in self.n_gram:
                    self.n_gram[word] = 1
                    self.numTotal = self.numTotal + 1
                else:
                    self.n_gram[word] = self.n_gram[word] + 1

        # n >= 2
        if self.n >= 2:

            stri = ''
            for i in range(0, len(words) - self.n + 1):
                for j in range(i, i + self.n):
                    stri = stri + words[j] + " "
                if stri not in self.n_gram:
                    self.n_gram[stri] = 1
                    self.numTotal = self.numTotal + 1
                else:
                    self.n_gram[stri] = self.n_gram[stri] + 1

                stri = ''


'''
    The function of main function that can be used.
    @param   filename   the path of train file
    @param   outputfile   the path of result file
    @return   the list saved the number of all the n-gram
'''


def mainFuncton(filename, outputfile, n):
    num_ngram = []
    with open(filename, 'r', encoding='utf-8') as f_in:
        sentence = []
        lines = f_in.readlines()
        for line in lines:
            if len(line.strip()) != 0:
                sentence.append(line.strip())
    for i in range(1, n + 1):
        ngramTemp = NGram(i)
        ngramTemp.scan(sentence, outputfile)
        print(str(i) + "-gram's number: " + str(ngramTemp.numTotal))
        num_ngram.append(ngramTemp.numTotal)

    return num_ngram


if __name__ == "__main__":
    n = sys.argv[1]
    filename = sys.argv[2]
    outputfile = sys.argv[3]
    n = int(n)
    mainFuncton(filename, outputfile, n)
