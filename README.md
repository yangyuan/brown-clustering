# Brown Clustering
Brown clustering in Python

The code will be released after Nov 25 for pedagogical reasons.

## Design
This project follows the both the original paper and the fixed window size optimization from another thesis.
* http://aclweb.org/anthology/J/J92/J92-4003.pdf
* http://people.csail.mit.edu/pliang/papers/meng-thesis.pdf

During reading the paper from MIT, I noticed that it contains some mistakes. To be specific, the equation 4.5 is wrong.
At the same time, the original paper dose not use fixed window size optimization (of course). So I derived the correct
equations of merging two cluster or append cluster, and use the terms from original paper as much as possible.

## Complexity
Assuming the corpus contains `n` words and `v` unique words. and the `m` is the fixed window size. This implementation
has `O(v*m^2+n)` complexity.

## Data Processing
This implementation only require you to provide corpus, as a list of sentences. Laplace smoothing and 2-gram will be
applied as a part of the algorithm. If you don't need any smoothing you can pass 0 as smoothing parameter `alpha`.

## Evaluation
This project contains multiple implementation, and one does not contain any optimization. This project contains some
unit tests which can make sure that different implementations will give exact same result .