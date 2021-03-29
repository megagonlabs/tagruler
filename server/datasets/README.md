# Using Your Own Data

We'll release some preprocessing code soon! Until then, you need to replicate the following file structure:

```
.
+-- datasets
|   +-- your_dataset_here
|   |   +-- processed.bert  
|   |   +-- processed.csv  
|   |   +-- processed.elmo
|   |   +-- processed.nlp
|   |   +-- processed.sbert
|   +-- example_dataset_1
|   +-- example_dataset_2
```

Where each file contains preprocessed data that follows the following schema:

`processed.csv` (csv format)
|    | text                                                                                                                                                                                                                                                                                                                                       | labels                                                                                                                                        | split |
|----|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|-------|
| 85 | angioedema due to ace inhibitors : common and inadequately diagnosed . the estimated incidence of angioedema during angiotensin - converting enzyme ( ace ) inhibitor treatment is between 1 and 7 per thousand patients . this potentially serious adverse effect is often preceded by minor manifestations that may serve as a warning . | I-DI,O,O,I-CH,I-CH,O,O,O,O,O,O,O,O,O,O,I-DI,O,I-CH,I-CH,I-CH,I-CH,I-CH,I-CH,I-CH,I-CH,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O | train |

'split' is one of 'train', 'dev', 'test', 'valid', where the latter three have labels (for train, labels can be empty).

`processed.bert` ([npy](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format) format)\
A Numpy array of 2-D Numpy arrays. Each 2-D Numpy array is an array of BERT representations of tokens in a text data sample.
```
array([array([[ 0.024, -0.004, ..., -0.002, 0.061 ],
              [ 0.059, -0.004, ..., -0.003, 0.044 ],
               ...,
              [ 0.048,  0.006, ...,  0.011, -0.016]], dtype=float32),
        ...,
        array([[-0.039, 0.090, ..., -0.002, -0.002 ],
                ...,
               [-0.019, 0.027, ..., -0.011,  0.045 ]], dtype=float32)], dtype=object)

```

`processed.elmo` ([npy](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format) format)\
A Numpy array of 2-D Numpy arrays. Each 2-D Numpy array is an array of ELMo representations of tokens in a text data sample.
```
array([array([[ 0.024, -0.004, ..., -0.002, 0.061 ],
              [ 0.059, -0.004, ..., -0.003, 0.044 ],
               ...,
              [ 0.048,  0.006, ...,  0.011, -0.016]], dtype=float32),
        ...,
        array([[-0.039, 0.090, ..., -0.002, -0.002 ],
                ...,
               [-0.019, 0.027, ..., -0.011,  0.045 ]], dtype=float32)], dtype=object)

```

`processed.sbert` ([npy](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format) format)\
A 2-D Numpy array which contains Sentence-BERT representations of text data samples. The shape of this array should be (N, V) where N is the number of text samples and V is the length of the Sentence-BERT representation.
```
array([[ 0.039,  0.011,  0.063, ..., -0.007, -0.004],
       [-0.047, -0.048,  0.023, ..., -0.026, -0.054],
       ...,
       [-0.025, -0.024,  0.054, ..., -0.017, -0.048]], dtype=float32)
```

`processed.nlp`\
\\TODO


# Data Attribution


## BC5CDR
Li, Jiao & Sun, Yueping & Johnson, Robin & Sciaky, Daniela & Wei, Chih-Hsuan & Leaman, Robert & Davis, Allan Peter & Mattingly, Carolyn & Wiegers, Thomas & lu, Zhiyong. (2016). 

BioCreative V CDR task corpus: a resource for chemical disease relation extraction. Database. 2016. baw068. 10.1093/database/baw068. Community-run, formal evaluations and manually annotated text corpora are critically important for advancing biomedical text-mining research. Recently in BioCreative V, a new challenge was organized for the tasks of disease named entity recognition (DNER) and chemical-induced disease (CID) relation extraction. Given the nature of both tasks, a test collection is required to contain both disease/chemical annotations and relation annotations in the same set of articles. Despite previous efforts in biomedical corpus construction, none was found to be sufficient for the task. Thus, we developed our own corpus called BC5CDR during the challenge by inviting a team of Medical Subject Headings (MeSH) indexers for disease/chemical entity annotation and Comparative Toxicogenomics Database (CTD) curators for CID relation annotation. To ensure high annotation quality and productivity, detailed annotation guidelines and automatic annotation tools were provided. The resulting BC5CDR corpus consists of 1500 PubMed articles with 4409 annotated chemicals, 5818 diseases and 3116 chemical-disease interactions. Each entity annotation includes both the mention text spans and normalized concept identifiers, using MeSH as the controlled vocabulary. To ensure accuracy, the entities were first captured independently by two annotators followed by a consensus annotation: The average inter-annotator agreement (IAA) scores were 87.49% and 96.05% for the disease and chemicals, respectively, in the test set according to the Jaccard similarity coefficient. Our corpus was successfully used for the BioCreative V challenge tasks and should serve as a valuable resource for the text-mining research community.

Database URL: http://www.biocreative.org/tasks/biocreative-v/track-3-cdr/

