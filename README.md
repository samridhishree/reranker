# Discriminative Reranker for a Probabilistic Parser

This repository contains an implementation of a parsing reranker trained with two learning algorithms: __Primal SVM__ and __Perceptron.__ The base parser produces a set of candidate parses for each sentence, with associated log probabilities that define the initial ranking of these parses. A second model attempts to improve on this initial ranking, using additional features extracted from the trees in the training set.

For details on the results and implementation please see the writeup: Result_Writeup.pdf
