# Word-in-Context (WiC)

The Word-in-Context task is to determine if two words occurring in two text fragments have the same or different meanings. Usually two occurrences of the same word probably in different forms are given. However, there are datasets with examples consisting of occurrences of two different words that are similar in one of their meanings [[1]](#paper1), [[2]](#paper2), and in the cross-lingual setup these two words and the corresponding text fragments are in different languages [[3]](#paper3). The WiC task is often framed as a binary classification task. For example, the WiC [[4]](#paper4) and MCL-WiC [[3]](#paper3) datasets contain binary labels and employ accuracy as the main evaluation metric. Alternatively, USim [[5]](#paper5), SCWS [[1]](#paper1) and CoSimLex [[2]](#paper2) were labeled with non-binary similarity scores and promote a graded formulation of the task. In this formulation, a WiC model shall produce scores that are similar to the human scores, or at least rank the pairs of uses similarly. Spearman's and Pearson's correlation coefficients are employed as evaluation metrics. During annotation of most LSCD dataset human annotators were essentially solving the graded WiC task, i.e. annotated the similarity of two uses of the same word on a scale. This provides data for evaluation of WiC models that may serve as a part of LSCD models.

In diachronic LSCD datasets two word uses constituting an example may be extracted from two documents belonging to distant time periods making those uses very different even when the target word has the same meaning. This might be challenging for models trained on traditional WiC datasets, which often contain examples from the same time period. We analyze how sensitive WiC models are to this shift in time period by comparing their performance on pairs of uses extracted from the old, the new or both corpora (EARLIER, LATER, COMPARE pairs).

1. Von Hassel replied that he had such faith in the **plane** that he had no hesitation about allowing his only son to become a Starfighter pilot.
2. This point, where the rays pass through the perspective **plane**, is called the seat of their representation.

### Reference

<a name="paper1">[1]</a>
Eric Huang, Richard Socher, Christopher Manning, and Andrew Ng. 2012. [Improving Word Representations via Global Context and Multiple Word Prototypes](https://aclanthology.org/P12-1092) In Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 873–882, Jeju Island, Korea. Association for Computational Linguistics

<a name="paper2">[2]</a>
Carlos Santos Armendariz, Matthew Purver, Matej Ulčar, Senja Pollak, Nikola Ljubešić, and Mark Granroth-Wilding. 2020. [CoSimLex: A Resource for Evaluating Graded Word Similarity in Context](https://aclanthology.org/2020.lrec-1.720) In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 5878–5886, Marseille, France. European Language Resources Association.

<a name="paper3">[3]</a>
Federico Martelli, Najla Kalach, Gabriele Tola, and Roberto Navigli. 2021. [SemEval-2021 Task 2: Multilingual and Cross-lingual Word-in-Context Disambiguation (MCL-WiC)](https://aclanthology.org/2021.semeval-1.3) . In Proceedings of the 15th International Workshop on Semantic Evaluation (SemEval-2021), pages 24–36, Online. Association for Computational Linguistics.

<a name="paper4">[4]</a>
Mohammad Taher Pilehvar and Jose Camacho-Collados. 2019. [WiC: the Word-in-Context Dataset for Evaluating Context-Sensitive Meaning Representations](https://aclanthology.org/N19-1128). In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 1267–1273, Minneapolis, Minnesota. Association for Computational Linguistics.

<a name="paper5">[5]</a>
Katrin Erk, Diana McCarthy, and Nicholas Gaylord. 2013. [Measuring word meaning in context.]( https://doi.org/10.1162/COLI_a_00142) Computational Linguistics, 39(3):511–554.
