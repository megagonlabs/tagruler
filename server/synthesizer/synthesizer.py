from synthesizer.parser import parse, simple_parse
from synthesizer.gll import *


class Synthesizer:

    token_list = []
    rel_code_list = []

    def __init__(self, t_origin, annots, t_label, de, cs: dict, sent_id, label_dict):
        self.origin_text = t_origin
        self.annotations = annots
        self.delimiter = de
        self.concepts = cs
        self.label = t_label
        self.instances = []
        self.sent_id = sent_id
        self.label_dict= label_dict

    def print(self):
        print(self.instances)

    def run(self):
        """
        Based on one label, suggest LFs

        Returns:
            List(Dict): labeling functions represented as dicts with fields:
                'Conditions', 'Connective', 'Direction', 'Label', and 'Weight'
        """
        self.token_list, self.rel_code_list = \
            parse(self.annotations, self.origin_text, self.delimiter, self.concepts)

        assert len(self.token_list) == len(self.rel_code_list)

        relationship_set = None
        relationship_undirected = dict()
        relationship_directed = dict()

        for i in range(len(self.rel_code_list)):
            crnt_rel_code = self.rel_code_list[i]
            crnt_token = self.token_list[i]

            if crnt_rel_code is None:
                if relationship_set is None:
                    relationship_set = Relationship(RelationshipType.SET)
                relationship_set.add(crnt_token, crnt_rel_code)

            else:
                if abs(crnt_rel_code) < 100:
                    # no direction relationship
                    if crnt_rel_code not in relationship_undirected:
                        relationship_undirected[crnt_rel_code] = Relationship(RelationshipType.UNDIRECTED)

                    relationship_undirected[crnt_rel_code].add(crnt_token, crnt_rel_code)

                else:
                    # directed relationship
                    abs_code = abs(crnt_rel_code)
                    if abs_code not in relationship_directed:
                        relationship_directed[abs_code] = Relationship(RelationshipType.DIRECTED)

                    relationship_directed[abs_code].add(crnt_token, crnt_rel_code)

        # for each relationship, generate instances
        if relationship_set is not None: # TODO: utilize this code for merging rules later
            self.instances.extend(relationship_set.get_instances(self.concepts, self.sent_id, self.label_dict))
        for k, v in relationship_undirected.items():
            self.instances.extend(v.get_instances(self.concepts, self.sent_id, self.label_dict))

        # add label to each instance
        for i, crnt_instance in enumerate(self.instances):
            # remove repeated conditions
            conditions = crnt_instance[CONDS]
            #crnt_instance[CONDS] = [dict(t) for t in {tuple(d.items()) for d in conditions}]

        # sort instances based on weight
        calc_weight(self.instances, self.concepts)
        self.instances.sort(key=lambda x: x[WEIGHT], reverse=True)
        return self.instances#[:20]

    def single_condition(self):
        extended_instances = []
        crnt_instance = self.instances[0]
        if len(crnt_instance[CONDS]) == 1:
            single_cond = crnt_instance[CONDS][0]
            crnt_text = list(single_cond.keys())[0]
            crnt_type = single_cond[crnt_text]

            # only process when the highlighted is token
            if crnt_type == KeyType[TOKEN]:
                # pipeline to process the crnt_text
                # remove stopwords and punct
                token_list = simple_parse(crnt_text, self.concepts)

                if len(token_list) == 0:
                    return

                # relationship set
                relationship_set = Relationship(RelationshipType.SET)
                for crnt_token in token_list:
                    relationship_set.add(crnt_token, None)
                extended_instances.extend( relationship_set.get_instances(self.concepts) )
        else:
            for condition in crnt_instance[CONDS]:
                new_inst = crnt_instance.copy()
                new_inst[CONDS] = [condition]
                extended_instances.extend([new_inst])
            self.instances = extended_instances



def calc_weight(instances, concepts):
    # TODO better weight calc
    for crnt_instance in instances:
        crnt_weight = 1
        for crnt_cond in crnt_instance[CONDS]:
            k = crnt_cond.get("string")
            v = crnt_cond.get("type")

            KeyType = {TOKEN: 0, CONCEPT: 1, NER: 2, REGEXP: 3, POS: 4, DEP: 5, ELMO_SIMILAR: 6, BERT_SIMILAR: 7,
                       SIM: 8}

            if v == KeyType[POS]:
                crnt_weight += 2
            elif v == KeyType[DEP]:
                crnt_weight += 1
        crnt_instance[WEIGHT] = crnt_weight


def test_synthesizer():
    text_origin = "the room is clean."
    annotations = [{"id":5, "start_offset": 5, "end_offset":9}, {"id":12, "start_offset": 12, "end_offset":17}]
    label = 1.0
    de = '#'
    concepts = dict()

    concepts['Hotel'] = ['room']

    crnt_syn = Synthesizer(text_origin, annotations, label, de, concepts)
    


#test_synthesizer()
