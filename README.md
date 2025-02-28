## UIUC_BioNLP@BioLaySumm2024
The official repository of the Paper: 
[UIUC_BioNLP at BioLaySumm: An Extract-then-Summarize Approach Augmented with Wikipedia Knowledge for Biomedical Lay Summarization](https://aclanthology.org/2024.bionlp-1.11) (You et al., BioNLP-WS 2024)

## System Framework
In this work, we compared the performance of fine-tuning both GPT-3.5 and [PubMed LED Large](https://huggingface.co/patrickvonplaten/led-large-16384-pubmed) models under each dataset.
<p align="center"><img width="70%" src="diagram.png" /></p>

## Resource Downloading
- Download the [NLM structured section labels](https://wayback.archive-it.org/7867/20241213200411/https://lhncbc.nlm.nih.gov/ii/areas/structured-abstracts/downloads/Structured-Abstracts-Labels-102615.txt)
- Download the shared task datasets: [train and val](https://www.codabench.org/datasets/download/149ce7f2-b498-49be-93be-44a1d439f72d/), [test data](https://github.com/TGoldsack1/Corpora_for_Lay_Summarisation).

## Citation

Please cite the below paper if you intent to use the code for your research.

```
@inproceedings{you-etal-2024-uiuc,
    title = "{UIUC}{\_}{B}io{NLP} at {B}io{L}ay{S}umm: An Extract-then-Summarize Approach Augmented with {W}ikipedia Knowledge for Biomedical Lay Summarization",
    author = "You, Zhiwen  and
      Radhakrishna, Shruthan  and
      Ming, Shufan  and
      Kilicoglu, Halil",
    editor = "Demner-Fushman, Dina  and
      Ananiadou, Sophia  and
      Miwa, Makoto  and
      Roberts, Kirk  and
      Tsujii, Junichi",
    booktitle = "Proceedings of the 23rd Workshop on Biomedical Natural Language Processing",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.bionlp-1.11",
    pages = "132--143",
}

```
