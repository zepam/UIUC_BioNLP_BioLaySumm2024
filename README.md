## UIUC_BioNLP@BioLaySumm2024
The official repository of the Paper: 
[UIUC_BioNLP at BioLaySumm: An Extract-then-Summarize Approach Augmented with Wikipedia Knowledge for Biomedical Lay Summarization](https://aclanthology.org/2024.bionlp-1.11) (You et al., BioNLP-WS 2024)

## System Framework
In this work, we compared the performance of fine-tuning both GPT-3.5 and [PubMed LED Large](https://huggingface.co/patrickvonplaten/led-large-16384-pubmed) models under each dataset.
<p align="center"><img width="70%" src="diagram.png" /></p>

## Resource Downloading
- Download the [NLM structured section labels](https://wayback.archive-it.org/7867/20241213200411/https://lhncbc.nlm.nih.gov/ii/areas/structured-abstracts/downloads/Structured-Abstracts-Labels-102615.txt)
- Download the shared task datasets: [train and val](https://www.codabench.org/datasets/download/149ce7f2-b498-49be-93be-44a1d439f72d/), [test data](https://github.com/TGoldsack1/Corpora_for_Lay_Summarisation).

## Usage
Please first change the data path in [utils.py](https://github.com/zhiwenyou103/UIUC_BioNLP_BioLaySumm2024/blob/main/utils.py#L67)

### Constractive Dataset Creation
Change your input data path [here](https://github.com/zhiwenyou103/UIUC_BioNLP_BioLaySumm2024/blob/main/contrastive_dataset_creation.py#L37) before running the script.
```bash
# You may define these hyper-parameters on your own
python contrastive_dataset_creation.py \
        --device cuda:0 \
        --chunk-size 600 \
        --pos-threshold 0.9 \
        --neg-threshold 0.01
```

### Generate Lay Language Summaries
Modify the pre-trained models [here](https://github.com/zhiwenyou103/UIUC_BioNLP_BioLaySumm2024/blob/main/evaluation.py#L30) before running the script: `python evaluation.py`.

### Run Model Fine-tuning
Modify the input data path [here](https://github.com/zhiwenyou103/UIUC_BioNLP_BioLaySumm2024/blob/main/fine_tune_elife.py#L173-L181) for elife and [here](https://github.com/zhiwenyou103/UIUC_BioNLP_BioLaySumm2024/blob/main/fine_tune_plos.py#L67-L76) for PLOS.
Fine-tune the PubMed LED large model for each dataset: `python fine_tune_elife.py` for eLife and `python fine_tune_plos.py` for PLOS.


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
