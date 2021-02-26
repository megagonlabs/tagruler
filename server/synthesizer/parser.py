import re
import spacy
from synthesizer.gll import Token
from typing import List


nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)


def parse(annotations: List[dict], origin_text: str, delimiter: str, concepts: dict):
    token_list = []
    rel_code_list = []

    origin_doc = nlp(origin_text)

    sentences = list(origin_doc.sents)
    mul_tok_link_key = -1
    for annotation in annotations:
        crnt_token_texts = origin_text[annotation["start_offset"]:annotation["end_offset"]].split(' ')
        crnt_tokens = []
        if len(crnt_token_texts) > 1:
            mul_tok_link_key += 1
        for crnt_token_text in crnt_token_texts:
            # crnt_token: text
            #crnt_token_text = origin_text[annotation["start_offset"]:annotation["end_offset"]]
            # initialize a new Token class
            crnt_token = Token(crnt_token_text, annotation['isPositive'])
            token_list.append(crnt_token)

            # Find the concept from annotation
            crnt_concept_name = None
            if ("label" in annotation) and (annotation["label"]):
                crnt_concept_name = annotation["label"]

            crnt_token.assign_concept(crnt_concept_name)

            # Find the relationship from annotation
            """
            crnt_rel_code = None
            if "link" in annotation:
                if not annotation["link"] is None:
                    crnt_rel_code = int(annotation["link"])
            """

            crnt_rel_code = None
            if len(crnt_token_texts) > 1:
                crnt_rel_code = mul_tok_link_key
            rel_code_list.append(crnt_rel_code)

            # Find the index of current annotation
            flag: bool = False
            # print(annotated_text)
            for crnt_sent in sentences:
                sent_start = origin_doc[crnt_sent.start].idx
                sent_end = origin_doc[crnt_sent.end-1].idx + len(origin_doc[crnt_sent.end-1])
                if (annotation["start_offset"] >= sent_start) and (annotation["end_offset"]<=sent_end):
                    crnt_token.assign_sent_idx(sentences.index(crnt_sent))
                    flag = True
                    break
            if not flag:
                print("No sentence found for the annotation: \"{}\"\nsentences: {}".format(annotation, sentences))

            # Find the named entity of current annotation
            #TODO if this is too slow, this can be done O(n) out of the loop
            crnt_token.assign_span_label(annotation['spanLabel'])
            crnt_tokens.append(crnt_token)
        offset = 0
        for crnt_token in crnt_tokens:
            for i,tk in enumerate(origin_doc):
                #TODO handle cases where selected span is not a token
                if tk.idx <= annotation['start_offset'] + offset and tk.idx+len(tk.text) >= annotation['start_offset'] + offset:
                    if len(tk.ent_type_)>0: crnt_token.assign_ner_type(tk.ent_type_)
                    crnt_token.assign_pos_type(tk.pos_)
                    crnt_token.assign_dep_rel(tk.dep_)
                    crnt_token.assign_tok_id(i)
                    offset += (len(crnt_token.text) + 1) #TODO what is there's double spaces?
                    break


    # Match existing concepts
    augment_concept(token_list, concepts)

    return token_list, rel_code_list


def augment_concept(token_list, concepts: dict):
    for crnt_token in token_list:
        if crnt_token.concept_name is not None:
            continue

        for key in concepts.keys():
            if crnt_token.text in concepts[key]:
                crnt_token.assign_concept(key)
                break


# remove stop word and punct
def simple_parse(text: str, concepts: dict):
    token_list = []
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    #print(tokens)

    if len(doc) == len(tokens):
        # early return
        return token_list

    ner_dict = dict()

    # merge multiple tokens if falling into one ent.text
    for ent in doc.ents:
        matched_text = []
        for i in range(len(tokens)):
            if tokens[i] in ent.text:
                matched_text.append(tokens[i])

        if len(matched_text) > 1:
            new_text = ""
            for crnt_text in matched_text:
                new_text += crnt_text
                tokens.remove(crnt_text)

            tokens.append(ent.text)
            ner_dict[ent.text] = ent.label_

    for crnt_text in tokens:
        crnt_token = Token(crnt_text)
        if crnt_text in ner_dict.keys():
            crnt_token.assign_ner_type(ner_dict[crnt_text])
        token_list.append(crnt_token)

    augment_concept(token_list, concepts)

    return token_list

