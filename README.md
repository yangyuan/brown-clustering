# Brown Clustering
Brown clustering in Python

The code will be released after Nov 25 for pedagogical reasons.

## Design
This project follows the both the original paper and the fixed window size optimization from another thesis.
* http://aclweb.org/anthology/J/J92/J92-4003.pdf
* http://people.csail.mit.edu/pliang/papers/meng-thesis.pdf

During reading the paper from MIT, I noticed that it contains some mistakes. To be specific, the equation 4.5 is wrong.
At the same time, the original paper dose not use fixed window size optimization (of course). So I derived the correct
equations of merging two cluster or appending a cluster, and use the terms from original paper as much as possible.

## Complexity
Assuming the corpus contains `n` words and `v` unique words. and the `m` is the fixed window size. This implementation
has `O(v*m^2+n)` complexity.

## Data Processing
This implementation only require you to provide corpus, as a list of sentences. Laplace smoothing and 2-gram will be
applied as a part of the algorithm. If you don't need any smoothing you can pass 0 as smoothing parameter `alpha`.

## Evaluation
This project contains multiple implementation, and one does not contain any optimization. This project contains some
unit tests which can make sure that different implementations will give exact same result.

## Dependencies
* `numpy` is used for `ndarray`
* `nltk` is used for `ngram`

I was not hesitated to use them considering that both of them are extremely popular. And if you don't like them, it
would be a very easy task to replace `ndarray` and `ngram` with some simple python code.


## The Same Page
### What is n
`n` is not number of words in corpus. Actually, n is the number of 2-grams. so `n` increased after Laplace smoothing.
### pl(i), pr(j) and p(i), p(j) and p(i, j)
In original paper `pl(i)=SUM(p(i, j) for each j)`. If you have start and end symbols in clusters, it's not difficult to 
find out that `pl(i) = pr(i) = c(i)/n`. So instead of sum them, we can directly keep 1-gram information and use that to
calculate `p(i) = pl(i) = pr(i)`.
### Laplace Smoothing
In order to make above true even after smoothing, we should add double the lambda for 1-gram. Wrong smoothing will end
with wrong clustering result.
### Likelihood, Quality and Average Mutual Information 
`Likelihood(C) = Quality(C) = AverageMutualInformation(C) + some constant`, both papers contains the idea of calculating
 AMI instead of Likelihood or Quality. In this project, `benefit` is the delta value you gain when you merge two clusters.

