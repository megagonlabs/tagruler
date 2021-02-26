"""Translate a dictionary explaining a labeling rule into a function
"""
import re
from wiser.rules import TaggingRule # A TaggingRule instance is defined for a tagging LF.
from snorkel.labeling import LabelingFunction
from synthesizer.gll import *
import numpy as np


def raw_stringify(s):
    """From a regex create a regular expression that finds occurences of the string as entire words
    
    Args:
        s (string): the string to look for
    
    Returns:
        string: a regular expession that looks for this string surrounded by non word characters
    """
    return "(?:(?<=\W)|(?<=^))({})(?=\W|$)".format(re.escape(s))


def find_indices(cond_dict: dict, text: str):
    """Find all instances of this condition in the text
    """
    v = cond_dict.get("type")
    k = cond_dict.get("string")
    case_sensitive = True if cond_dict.get("case_sensitive") else False

    if v == KeyType[NER]:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == k:
                return [(doc[ent.start].idx, doc[ent.end-1].idx + len(doc[ent.end-1].text))]
        return []
    elif v == KeyType[POS] or v == KeyType[DEP] or v == KeyType[ELMO_SIMILAR] or v == KeyType[BERT_SIMILAR]:
        return [(0,0)]
    elif case_sensitive:
        return [(m.start(), m.end()) for m in re.finditer(k, text)]
    else:
        return [(m.start(), m.end()) for m in re.finditer(k, text, re.IGNORECASE)]


def make_lf(instance, concepts, emb_dict, ui_label_dict):
    def apply_instance(self, instance):
        """
        An ``Instance`` is a collection of :class:`~allennlp.data.fields.field.Field` objects,
        specifying the inputs and outputs to
        some model.  We don't make a distinction between inputs and outputs here, though - all
        operations are done on all fields, and when we return arrays, we return them as dictionaries
        keyed by field name.  A model can then decide which fields it wants to use as inputs as which
        as outputs.

        The ``Fields`` in an ``Instance`` can start out either indexed or un-indexed.  During the data
        processing pipeline, all fields will be indexed, after which multiple instances can be combined
        into a ``Batch`` and then converted into padded arrays.

        Parameters
        ----------
        instance : An ``Instance`` is a collection of :class:`~allennlp.data.fields.field.Field` objects.
            instance['fields'] : ``Dict[str, Field]``
        ex:
            instance['fields']['tags'] = ['ABS', 'I-OP', 'ABS', 'ABS']
        """
        label = self.lf_dict.get(LABEL)
        direction = bool(self.lf_dict.get(DIRECTION))
        conn = self.lf_dict.get(CONNECTIVE)
        conds = self.lf_dict.get(CONDS)
        types = [cond.get("TYPE_") for cond in conds]
        strs_ = [cond.get("string") for cond in conds]
        #labels = np.array([np.array(['ABS'] * len(instance['tokens']), dtype=object)] * len(conds))

        type_ = conds[0].get("TYPE_")
        str_ = conds[0].get("string")
        is_pos = conds[0].get("positive")
        labels = np.array(['ABS'] * len(instance['tokens']), dtype=object)
        if type_ == BERT_SIMILAR or type_ == ELMO_SIMILAR:
            emb_type = 'bert' if type_ == BERT_SIMILAR else 'elmo'
            emb_thres = BERT_THRESHOLD if type_ == BERT_SIMILAR else ELMO_THRESHOLD
            sent_id = self.lf_dict.get(SENT_ID)
            tok_id = self.lf_dict.get(TOK_ID)
            if type(tok_id) == list:
                target_emb = self.emb_dict.emb_dict[emb_type][sent_id][tok_id]
                emb = instance[emb_type]
                cos_scores = target_emb @ emb.T
                if is_pos:
                    similar_inds = (cos_scores > emb_thres)
                else:
                    similar_inds = (cos_scores <= emb_thres)
                similar_ind = similar_inds[0][:-len(tok_id)+1]
                for i in range(1,len(tok_id)):
                    similar_ind = (similar_ind) & (similar_inds[i][i:len(similar_inds[i])-len(tok_id)+1+i])
                for i in range(len(tok_id)):
                    labels[i:len(labels)-len(tok_id)+1+i][similar_ind] = ui_label_dict[label]
            else:
                # target_emb is the emb vec corresponding to the similarity target word
                target_emb = self.emb_dict.emb_dict[emb_type][sent_id][tok_id]
                # emb is an N x M matrix containing token embeddings in a sentence
                emb = instance[emb_type]
                cos_scores = np.dot(emb, target_emb)
                if is_pos:
                    labels[cos_scores > emb_thres] = ui_label_dict[label]
                else:
                    labels[cos_scores <= emb_thres] = ui_label_dict[label]
                # try:
                # except:
                #     print('error')
        elif type_ == POS:
            for i, pos in enumerate([token.pos_ for token in instance['tokens']]):
                if (pos in str_ and is_pos) or (pos not in str_ and not is_pos):
                    labels[i] = ui_label_dict[label]
        elif type_ == DEP:
            for i, dep in enumerate([token.dep_ for token in instance['tokens']]):
                if (dep in str_ and is_pos) or (dep not in str_ and not is_pos):
                    labels[i] = ui_label_dict[label]
        elif type_ == NER:
            for i, ner in enumerate([token.ent_type_ for token in instance['tokens']]):
                if (ner in str_ and is_pos) or (ner not in str_ and not is_pos):
                    labels[i] = ui_label_dict[label]
        return list(labels)


    def lf_init(self):
        pass


    LF_class = type(instance['name'], (TaggingRule,), {"__init__":lf_init, "lf_dict":instance, "apply_instance":apply_instance, "emb_dict":emb_dict, "name":instance['name']} )
    return LF_class()
