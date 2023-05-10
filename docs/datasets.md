# Datasets

The following table shows all the data set we integrated into benchmark.

|    Data set   | LGS |  n  |   N/V/A  | &#124;U&#124; | AN | JUD |          Task          | t<sub>1</sub> | t<sub>2</sub> | Reference | Version |
|:-------------:|:---:|:---:|:--------:|:-------------:|:--:|:---:|:----------------------:|:-------------:|:-------------:|:---------:|:--------:|
|      DWUG     | DE  | 48  | 32/14/2  |      178      | 8  | 37k | WiC, WSI, LSCD (B,G,C) |   1800–1899   |   1946–1990  | [Schlechtweg et al. (2021)](#paper1) | 2.2.0 |
|      DWUG     | EN  | 40  |  36/4/0  |      189      | 9  | 29k | WiC, WSI, LSCD (B,G,C) |   1810–1860   | 1960–2010  | [Schlechtweg et al. (2021)](#paper1) | 2.0.1 |
|      DWUG     | SV  | 40  |  31/6/3  |      168      | 5  | 20k | WiC, WSI, LSCD (B,G,C) |   1790–1830   | 1895–1903  | [Schlechtweg et al. (2021)](#paper1) | 2.0.1 |
|      DWUG     | ES  | 100 | 51/24/25 |      40       | 12 | 62k | WiC, WSI, LSCD (B,G,C) |   1810–1906   | 1994–2020  | [Zamora-Reina et al. (2022)](#paper2) | 4.0.0 |
|    DiscoWUG   | DE  | 75  | 39/16/20 |      49       | 8  | 24k | WiC, WSI, LSCD (B,G,C) |   1800–1899   | 1946–1990  | [Kurtyigit et al. (2021)](#paper3) | 1.1.1 |
|     RefWUG    | DE  | 22  |  15/1/6  |      19       | 5  | 4k  | WiC, WSI, LSCD (B,G,C) |   1750–1800   | 1850–1900  | ?  | 1.1.0 |
| NorDiaChange1 | NO  | 40  |  40/0/0  |      21       | 3  | 14k | WiC, WSI, LSCD (B,G,C) |   1929–1965   | 1970–2013  | [Kutuzov et al. (2022)](#paper4) | 1.0.0 |
| NorDiaChange2 | NO  | 40  |  40/0/0  |      21       | 3  | 15k | WiC, WSI, LSCD (B,G,C) |   1980–1990   | 2012–2019  | [Kutuzov et al. (2022)](#paper4) | 1.0.0 |
|     DURel     | DE  | 22  |  15/1/6  |      104      | 5  | 6k  |     WiC, LSCD (C)      |   1750–1800   | 1850–1900 | [Schlechtweg et al. (2018)](#paper5) | 3.0.0 |
|     SURel     | DE  | 22  |  19/3/0  |      104      | 4  | 5k  |     WiC, LSCD (C)      |   general   | domain | [Hätty et al. (2019)](#paper6) | 3.0.0 |
|  RuSemShift1  | RU  | 71  |  65/6/0  |      119      | 5  | 21k |     WiC, LSCD (C)      |   1682–1916   | 1918–1990 | [Rodina and Kutuzov (2020)](#paper7) | 2.0.0 |
|  RuSemShift2  | RU  | 69  | 57/12/0  |      105      | 5  | 18k |     WiC, LSCD (C)      |   1918–1990   | 1991–2016 | [Rodina and Kutuzov (2020)](#paper7) | 2.0.0 |
| RuShiftEval1  | RU  | 111 | 111/0/0  |      60       | 3  | 10k |     WiC, LSCD (C)      |   1682–1916   | 1918–1990 | [Kutuzov and Pivovarova (2021)](#paper8) | 2.0.0 |
| RuShiftEval2  | RU  | 111 | 111/0/0  |      60       | 3  | 10k |     WiC, LSCD (C)      |   1918–1990   | 1991–2016 | [Kutuzov and Pivovarova (2021)](#paper8) | 2.0.0 |
| RuShiftEval3  | RU  | 111 | 111/0/0  |      60       | 3  | 10k |     WiC, LSCD (C)      |   1682–1916   | 1991–2016 | [Kutuzov and Pivovarova (2021)](#paper8) | 2.0.0 |

LGS = language, n = number of target words, N/V/A = number of nouns/verbs/adjectives, |U| = average number usages per word, AN = number of annotators, JUD = total number of judged usage pairs, Task = possible evaluation tasks, t<sub>1</sub>, t<sub>2</sub> = time period 1/2, Reference = data set reference paper, Version = version used for experiments.

## WUGs

Word Usage Graphs (WUGs) is the graphs displaying the relation between the usage of words. Each usage of word is a node which is connected by the weighted edges. The edges dipict the human-annotated semantic proximity of use pairs. The WUG data set can be found on the [WUGsite](https://www.ims.uni-stuttgart.de/en/research/resources/experiment-data/wugs/).

### Reference

<a name="paper1">[1]</a>
Dominik Schlechtweg, Nina Tahmasebi, Simon Hengchen, Haim Dubossarsky, Barbara McGillivray. 2021. [DWUG: A large Resource of Diachronic Word Usage Graphs in Four Languages](https://aclanthology.org/2021.emnlp-main.567/). In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing.

<a name="paper2">[2]</a>
Frank D. Zamora-Reina, Felipe Bravo-Marquez, Dominik Schlechtweg. 2022. [LSCDiscovery: A shared task on semantic change discovery and detection in Spanish](https://aclanthology.org/2022.lchange-1.16/). In Proceedings of the 3rd International Workshop on Computational Approaches to Historical Language Change.

<a name="paper3">[3]</a>
Sinan Kurtyigit, Maike Park, Dominik Schlechtweg, Jonas Kuhn, Sabine Schulte im Walde. 2021. [Lexical Semantic Change Discovery](https://aclanthology.org/2021.acl-long.543/). In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers).

<a name="paper4">[4]</a>
Andrey Kutuzov, Samia Touileb, Petter Mæhlum, Tita Enstad, and Alexandra Wittemann. 2022. [NorDiaChange: Diachronic Semantic Change Dataset for Norwegian](https://aclanthology.org/2022.lrec-1.274/). In Proceedings of the Thirteenth Language Resources and Evaluation Conference.

<a name="paper5">[5]</a>
Dominik Schlechtweg, Sabine Schulte im Walde, and Stefanie Eckmann. 2018. [Diachronic Usage Relat
edness (DURel): A framework for the annotation of lexical semantic change.](https://aclanthology.org/N18-2027) In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 169–174.

<a name="paper6">[6]</a>
Anna Hätty, Dominik Schlechtweg, and Sabine Schulte im Walde. 2019. [SURel: A Gold Standard for Incorporating Meaning Shifts into Term Extraction](https://aclanthology.org/S19-1001). In Proceedings of the 8th Joint Conference on Lexical and Computational Semantics, pages 1–8.

<a name="paper7">[7]</a>
Julia Rodina and Andrey Kutuzov. 2020. [RuSemShift: a dataset of historical lexical semantic change in Russian](https://aclanthology.org/2020.coling-main.90). In Proceedings of the 28th International Conference on Computational Linguistics (COLING 2020). Association for Computational Linguistics.

<a name="paper8">[8]</a>
Andrey Kutuzov and Lidia Pivovarova. 2021. Rushifteval: a shared task on semantic shift detection for russian. _Komp’yuternaya Lingvistika i Intellektual’nye Tekhnologii: Dialog conference_.
