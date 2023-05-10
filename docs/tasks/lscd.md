# Lexical Semantic Change Detection (LSCD)

Lexical Semantic Change Detection (LSCD) can be seen as the combination of (at least) three lexical semantic tasks [[1]](#paper1): (i) measurement of semantic proximity between word usages, (ii) clustering of the usages based on their semantic proximity, and (iii) estimation of semantic change labels from the obtained clusterings. Task (i) and (ii) corresponds to the lexicographic process of deriving word senses [[2]](#paper2), while task (iii) measures LSC based on the derived word senses. The tasks need to be solved sequentially, in the order given above, as each is dependent on the output of the previous task, e.g., word usages can only be clustered once their semantic proximity has been estimated.

### Reference

<a name="paper1">[1]</a>
Dominik Schlechtweg. 2023. [Human and Computational Measurement of Lexical Semantic Change](http://dx.doi.org/10.18419/opus-12833). PhD thesis. University of Stuttgart.

<a name="paper2">[2]</a>
Adam Kilgarriff. 2007. [Word Senses](https://doi.org/10.1007/978-1-4020-4809-8_2), chapter 2. Springer.
