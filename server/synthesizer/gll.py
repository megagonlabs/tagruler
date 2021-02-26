from enum import Enum
import json
import os


class RelationshipType(Enum):
    SET = 0
    UNDIRECTED = 1
    DIRECTED = 2


DELIMITER = '#'
CONNECTIVE = 'Connective'
CONTEXT = 'Context'
DIRECTION = 'Direction'
CONDS = 'Conditions'
WEIGHT = 'Weight'
LABEL = 'Label'
ID = 'ID'

AND = 'AND' # all conditions occuring anywhere in the text
OR = 'OR' # any of the conditions occuring

TOKEN = 'TOKEN'
REGEXP = 'REGEXP'
CONCEPT = 'CONCEPT'
NER = 'NER'
POS = 'POS'
DEP = 'DEP'
ELMO_SIMILAR = 'ELMO_SIMILAR'
BERT_SIMILAR = 'BERT_SIMILAR'
SIM = 'SIM'
SIMILAR_CONTEXTS = [BERT_SIMILAR, ELMO_SIMILAR]

# Give a text explanation of each type of condition
condition_explanations = {
    TOKEN: "Match this text exactly",
    REGEXP: "Match this regular expression",
    CONCEPT: "Match any condition in this concept",
    NER: "Named entity recognition",
    POS: "Part of speech matching",
    DEP: "Dependency parsing",
    ELMO_SIMILAR: "Match similar tokens based on ELMO embedding",
    BERT_SIMILAR: "Match similar tokens based on BERT embedding",
    SIM: "Match token similarity based on defined embedding"
}

SENTENCE = "SENTENCE"
SENT_ID = "SENT_ID"
TOK_ID = "TOK_ID"

CONTEXT_CONTENT = "CONTEXT_CONTENT"

ConnectiveType = {OR: 0, AND: 1}

ContextType = {SENTENCE: 1}

KeyType = {TOKEN: 0, CONCEPT: 1, NER: 2, REGEXP: 3, POS: 4, DEP:5, ELMO_SIMILAR: 6, BERT_SIMILAR: 7, SIM: 8}

ELMO_THRESHOLD = 0.4
BERT_THRESHOLD = 0.5

MV = 'MV' # Majority voting
SNORKEL = 'SNORKEL'
NB = 'NB' # Naive Bayes
HMM_ = 'HMM' # Hidden Markov Model
FS = 'FS' # FlyingSquid

ACTIVE_SAMPLE = 'ENSEMBLE'  #ENTROPY: entropy-based uncertainty sampling, FP: False-positive sampling, ENSEMBLE: ENSEMBLE of ENTROPY and FP

class ConceptWrapper:
    filename = 'concept_collection.json'
    def __init__(self, d={}):
        self.dict = d

    def __str__(self):
        return self.name

    def __len__(self):
        return len(self.dict.keys())

    def get_dict(self):
        return self.dict

    def add_element(self, name: str, elements):
        # make sure there are no duplicates in elements:
        def unique_hash(elt):
            string_id = "#".join([elt.get("string"), str(elt.get("case_sensitive")), str(elt.get("type"))])
            elt["key"] = string_id
            return string_id
        elt_hashes = {unique_hash(elt): elt for elt in elements}
        self.dict[name] = list(elt_hashes.values())

    def get_elements(self, name: str):
        return self.dict[name]

    def delete_element(self, name: str, elements):
        assert name in self.dict.keys()
        for element in elements:
            self.dict[name].remove(element)

    def delete_concept(self, name: str):
        assert name in self.dict.keys()
        del self.dict[name]

    def save(self, dirname):
        with open(os.path.join(dirname, self.filename), "w+") as file:
            json.dump(self.dict, file)

    def load(self, dirname):
        with open(os.path.join(dirname, self.filename), "r") as file:
            self.dict = json.load(file)


class Token:

    """user defined token, usually a word 
    
    Attributes:
        concept_name (string): if the token is assigned to a concept, the name of the concept
        ner_type (string): spacy-defined Named Entity code (for example, ORDINAL for "first")
        sent_idx (int): Which sentence in the document the token belongs to
        text (string): the text of the token
    """
    
    def __init__(self, T, is_pos=True):
        self.text = T
        self.sent_idx = -1
        self.concept_name = None
        self.ner_type = None
        self.pos_type = None
        self.dep_rel = None
        self.tok_id = None
        self.is_positive = is_pos

    def __str__(self):
        return self.text

    def assign_concept(self, c):
        self.concept_name = c

    def assign_sent_idx(self, i):
        self.sent_idx = i

    def assign_ner_type(self, ner):
        self.ner_type = ner

    def assign_pos_type(self, pos):
        self.pos_type = pos

    #TODO: incorporate SPO triple relation
    def assign_dep_rel(self, dep):
        self.dep_rel = dep

    def assign_tok_id(self, tokid):
        self.tok_id = tokid

    def assign_span_label(self, span_label):
        self.span_label = span_label


class Relationship:

    """user defined relationshpi between a set of Tokens. 
    Can be OR, AND (directional or non-directional)
    
    Attributes:
        list (Token): List of annotated spans
        same_sentence (bool): Whether or not the conditions must occur in the same sentence
        type (int): A code describing the relationship type (AND, OR, etc.)
    """
    
    def __init__(self, type: RelationshipType):
        self.list = [] # a list of Token
        self.rel_list = []
        self.type = type
        self.context = None

    def __str__(self):
        pass

    # add the Token to relationship, depending on the rel_code
    # rel_code: None SET
    # 0-100: UNDIRECTED
    # negative (<-100): positive (>100) DIRECTED: negative -> positive
    def add(self, token: Token, rel_code):
        if rel_code is None:
            assert self.type == RelationshipType.SET
            self.list.append(token)
        elif abs(rel_code) < 100:
            assert self.type == RelationshipType.UNDIRECTED
            self.list.append(token)
            self.rel_list.append(rel_code)
        else:
            assert self.type == RelationshipType.DIRECTED
            self.rel_list.append(rel_code)
            if rel_code < 0:
                self.list.insert(0, token)
            else:
                self.list.insert(1, token)

    # return a list of instances
    def get_instances(self, concepts, sent_id, label_dict):
        instances = []
        if len(self.list) == 0:
            return instances

        for crnt_token in self.list:
            if crnt_token.span_label is not None:
                # if label_dict[crnt_token.span_label] != 'O':
                for sim_type in SIMILAR_CONTEXTS:
                    crnt_instance = {CONDS: []}
                    crnt_cond_token = Condition(crnt_token.text.lower(), sim_type, case_sensitive=False, is_pos=crnt_token.is_positive)
                    crnt_instance[CONDS].append(crnt_cond_token)
                    crnt_instance[LABEL] = crnt_token.span_label
                    crnt_instance[SENT_ID] = sent_id
                    crnt_instance[TOK_ID] = crnt_token.tok_id
                    crnt_instance['Target'] = crnt_token.text
                    instances.append(crnt_instance)
                if crnt_token.ner_type is not None:
                    crnt_instance = {CONDS: []}
                    crnt_cond_ner = Condition(crnt_token.ner_type, NER, is_pos=crnt_token.is_positive)
                    crnt_instance[CONDS].append(crnt_cond_ner)
                    crnt_instance[LABEL] = crnt_token.span_label
                    crnt_instance[SENT_ID] = None
                    crnt_instance['Target'] = crnt_token.text
                    instances.append(crnt_instance)
                if crnt_token.pos_type is not None:
                    crnt_instance = {CONDS: []}
                    crnt_cond_pos = Condition(crnt_token.pos_type, POS, is_pos=crnt_token.is_positive)
                    crnt_instance[CONDS].append(crnt_cond_pos)
                    crnt_instance[LABEL] = crnt_token.span_label
                    crnt_instance[SENT_ID] = None
                    crnt_instance['Target'] = crnt_token.text
                    instances.append(crnt_instance)
                if crnt_token.dep_rel is not None:
                    crnt_instance = {CONDS: []}
                    crnt_cond_dep = Condition(crnt_token.dep_rel, DEP, is_pos=crnt_token.is_positive)
                    crnt_instance[CONDS].append(crnt_cond_dep)
                    crnt_instance[LABEL] = crnt_token.span_label
                    crnt_instance[SENT_ID] = None
                    crnt_instance['Target'] = crnt_token.text
                    instances.append(crnt_instance)

        # Multi-token rule
        if len(self.rel_list) > 1:
            for sim_type in SIMILAR_CONTEXTS:
                crnt_instance = {CONDS:[]}
                # TODO Improve multi-token rule generation
                for crnt_token in self.list:
                    crnt_cond = Condition(crnt_token.text.lower(), sim_type, is_pos=crnt_token.is_positive)
                    crnt_instance[CONDS].append(crnt_cond)
                crnt_instance[LABEL] = crnt_token.span_label
                crnt_instance[SENT_ID] = sent_id
                crnt_instance[TOK_ID] = [crnt_token.tok_id for crnt_token in self.list]
                crnt_instance['Target'] = ' '.join([t.text for t in self.list])
                instances.append(crnt_instance)

        if self.type == RelationshipType.SET:
            for crnt_instance in instances:
                crnt_instance[CONNECTIVE] = ConnectiveType[OR]
                crnt_instance["CONNECTIVE_"] = OR
        else:
            for crnt_instance in instances:
                crnt_instance[CONNECTIVE] = ConnectiveType[AND]
                crnt_instance["CONNECTIVE_"] = AND
        if self.type == RelationshipType.DIRECTED:
            for crnt_instance in instances:
                crnt_instance[DIRECTION] = True
        else:
            for crnt_instance in instances:
                crnt_instance[DIRECTION] = False

        return instances



# for classification problem, this is the label class
class Label:
    def __init__(self, name):
        self.task_name = name
        self.dict = {-1:"ABSTAIN"}
        self.inv_dict = {"ABSTAIN":-1}
        self.count = 0

    def add_label(self, key: int, lname: str):
        self.dict[key] = lname
        self.inv_dict[lname] = key
        self.count += 1

    def to_int(self, lname: str):
        return self.dict[lname]

    def to_name(self, i: int):
        for k, v in self.dict.items():
            if i == v:
                return k

    def change_name(self, name):
        self.task_name = name

    def to_dict(self):
        crnt_keys = list(self.dict.keys())
        crnt_keys.remove(-1)
        return {key:self.dict[key] for key in crnt_keys}

class Embeddings:
    def __init__(self, df):
        self.emb_dict = {}
        self.emb_dict['elmo'] = df['elmo'].copy()
        self.emb_dict['bert'] = df['bert'].copy()

def Condition(string, type_str, case_sensitive=False, is_pos=True):
    return {
        "string": string,
        "type": KeyType[type_str],
        "TYPE_": type_str,
        "case_sensitive": case_sensitive,
        "positive": is_pos,
        "explanation": condition_explanations[type_str]
    }
