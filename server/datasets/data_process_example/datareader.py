from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.word_tokenizer import SpacyWordSplitter, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from tqdm.auto import tqdm
from typing import Iterator, List, Dict
from xml.etree import ElementTree
import pandas as pd

@DatasetReader.register('text')
class TextDatasetReader(DatasetReader):
    """
    DatasetReader for Laptop Reviews corpus available at
    http://alt.qcri.org/semeval2014/task4/.
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, doc_id: str, tokens: List[Token], tags: List[str] = None) -> Instance:
        tokens_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": tokens_field}

        if tags:
            tags_field = SequenceLabelField(labels=tags, sequence_field=tokens_field)
            fields["tags"] = tags_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        #splitter = JustSpacesWordSplitter()
        splitter = SpacyWordSplitter('en_core_web_sm', True, True, True)
        tokenizer = WordTokenizer(word_splitter=splitter)
        df = pd.read_csv(file_path)
        #TODO: fix based on correct index
        for i,text in tqdm(enumerate(df['text'])):
            # Tokenizes the sentence
            tokens = tokenizer.tokenize(text)
            space_split = text.split(' ')
            tokens_merged = []
            ii=0
            jj=0
            while ii<len(tokens) and jj<len(space_split):
                if space_split[jj].startswith(tokens[ii].text):
                    tokens_merged.append(tokens[ii])
                    ii+=1
                    jj+=1
                else:
                    tokens_merged[-1] = Token(tokens_merged[-1].text + tokens[ii].text,
                          tokens_merged[-1].idx,
                          tokens_merged[-1].lemma_,
                          tokens_merged[-1].pos_,
                          tokens_merged[-1].tag_,
                          tokens_merged[-1].dep_,
                          tokens_merged[-1].ent_type_)
                    ii+=1
            tokens = tokens_merged

            # Assigns tags based on annotations
            labels = df['labels'][i]
            tags = labels.split(',')

            #TODO: figure out why B- doesn't work and fix if it works
            def mapBtoI(t):
                """
                We assume span labels are in the IOB format (https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging))
                We only support token-level annotation for now, assuming all tags start with "I-". This function converts tags with "B-" into "I-"
                """
                if len(t) == 0:  return t
                elif t[0] == 'B':
                    return 'I' + t[1:]
                else:
                    return t

            tags = list(map(mapBtoI,tags))
            yield self.text_to_instance(i, tokens, tags)

    def yield_from_id(self, file_path, i):

        #splitter = JustSpacesWordSplitter()
        splitter = SpacyWordSplitter('en_core_web_sm', True, True, True)
        tokenizer = WordTokenizer(word_splitter=splitter)
        df = pd.read_csv(file_path)
        #TODO: fix based on correct index
        text = df.at[i, 'text']
        
        # Tokenizes the sentence
        tokens = tokenizer.tokenize(text)
        space_split = text.split(' ')
        tokens_merged = []
        ii=0
        jj=0
        while ii<len(tokens) and jj<len(space_split):
            if space_split[jj].startswith(tokens[ii].text):
                tokens_merged.append(tokens[ii])
                ii+=1
                jj+=1
            else:
                tokens_merged[-1] = Token(tokens_merged[-1].text + tokens[ii].text,
                      tokens_merged[-1].idx,
                      tokens_merged[-1].lemma_,
                      tokens_merged[-1].pos_,
                      tokens_merged[-1].tag_,
                      tokens_merged[-1].dep_,
                      tokens_merged[-1].ent_type_)
                ii+=1
        tokens = tokens_merged

        # Assigns tags based on annotations
        labels = df['labels'][i]
        tags = labels.split(',')

        #TODO: figure out why B- doesn't work and fix if it works
        def mapBtoI(t):
            """
            We assume span labels are in the IOB format (https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging))
            We only support token-level annotation for now, assuming all tags start with "I-". This function converts tags with "B-" into "I-"
            """
            if len(t) == 0:  return t
            elif t[0] == 'B':
                return 'I' + t[1:]
            else:
                return t

        tags = list(map(mapBtoI,tags))
        
        return [(tok, tag) for (tok, tag) in zip(tokens, tags) if tag !='O']

