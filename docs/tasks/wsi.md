# Word Sense Induction (WSI)

Word Sense Induction (WSI) task is to infer which senses a given target word has from its occurrences in an unlabeled corpus. It is usually framed as a clustering task where a model shall cluster a given set of occurrences of the same target word probably in different grammatical forms into clusters corresponding to the senses of this word. Unlike the more popular Word Sense Disambiguation task, in WSI no sense inventory is given to the model and the number of senses of the target word is not known as well. Popular WSI datasets include SemEval 2010 Task 14 [[1]](#paper1) and SemEval 2013 Task 13 [[2]](#paper2). The latter dataset contains examples with several senses assigned to a single word occurrence, thus, requiring soft clustering approaches.

### Reference

<a name="paper1">[1]</a>
Suresh Manandhar and Ioannis Klapaftis. 2009. [SemEval-2010 Task 14: Evaluation Setting for Word Sense Induction & Disambiguation Systems](https://aclanthology.org/W09-2419). In Proceedings of the Workshop on Semantic Evaluations: Recent Achievements and Future Directions (SEW-2009), pages 117–122, Boulder, Colorado. Association for Computational Linguistics.

<a name="paper2">[2]</a>
David Jurgens and Ioannis Klapaftis. 2013. [SemEval-2013 Task 13: Word Sense Induction for Graded and Non-Graded Senses](https://aclanthology.org/S13-2049). _In Second Joint Conference on Lexical and Computational Semantics (*SEM), Volume 2: Proceedings of the Seventh International Workshop on Semantic Evaluation (SemEval 2013)_, pages 290–299, Atlanta, Georgia, USA. Association for Computational Linguistics.
