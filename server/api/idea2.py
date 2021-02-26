import copy
import json
import os
import pandas as pd
import shutil
import spacy
import sys

#from google.oauth2 import service_account
#from googleapiclient import discovery


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# from data.preparer import load_restaurant_dataset,load_meg_dataset
from datetime import datetime
from flask import send_from_directory
from project import Project
from project import add_file
from project import initialize_from_task
from synthesizer.gll import CONCEPT
from synthesizer.gll import ConceptWrapper
from synthesizer.gll import ConnectiveType
from synthesizer.gll import DELIMITER
from synthesizer.gll import ID
from synthesizer.gll import KeyType
from synthesizer.gll import Label
from synthesizer.parser import nlp
from synthesizer.synthesizer import Synthesizer
from verifier.DB import LF_DB
from verifier.DB import interactionDB
from verifier.modeler import Modeler
from verifier.translator import find_indices
import threading


# MODE can be "review"
MODE = "reviews"
USER_ID = "testing"

# fetch object with for concepts, modeler, and label types

# IMPORTANT
# Do you want to initialize the project without using the UI? (Good for development)
# Use this snippet

project = initialize_from_task('reviews')

# Do you want to use the UI to upload data (optional), select data for the task, and define labels?
# Use this snippet

#project = Project()

stat_history = pd.DataFrame()
stats = {}
all_concepts = ConceptWrapper()
#############   DATASETS   #############

def assign_dataset(data):
    """Record which dataset this project relies on"""
    global project
    dataset_uuid = data.get('dataset_uuid')
    project = initialize_from_task(dataset_uuid)
    project.assign_dataset(dataset_uuid)

def post_data(file, dataset_uuid, label_col="label", text_col="text"):
    """Add file to a new or existing dataset
    
    Args:
        file (file): csv file
        dataset_uuid (str): ID of the dataset
        label_col (str, optional): name of the column with label data
        text_col (str, optional): name of the column with text data
    
    Returns:
        list(str): The available datasets
    """
    add_file(file, dataset_uuid, label_col, text_col)
    return get_datasets()

def listdir(dirname):
    return sorted([f for f in os.listdir(dirname) if not f.startswith('.')], key=lambda f: f.lower())

def get_datasets():
    """List available datasets"""
    return listdir('datasets')

def get_dataset(dataset_uuid):
    """List files in a given dataset"""
    return listdir(os.path.join('datasets', dataset_uuid))

###########  LAUNCH PROJECT  ###########

def launch():
    """Initialize the modeler
    """
    # launch in a new thread
    workThread = threading.Thread(target=project.launch)
    workThread.start()

    if project.name:
        global MODE
        MODE = project.name

def progress():
    """Get progress of the project launch (data preprocessing and model launching)
    
    Returns:
        float: a number from 0 to 1
    """
    return project.progress()

#############     LABELS    #############

def get_labels():
    l = project.labels.to_dict()
    return {k:v for k,v in l.items()}

def post_labels(new_labels: dict):
    for label in new_labels['labels']:
        project.labels.add_label(label['key'], label['name'])
    if new_labels['name'] != None:
        project.set_name(new_labels['name'])

############# GLL OPERATORS ############

def get_connective():
    return ConnectiveType


def get_keytype():
    return KeyType

###########     CONCEPTS   #############

def get_all_concepts():
    return project.concepts.get_dict()

def create_concept(concept):
    if concept["name"] not in project.concepts.get_dict():
        project.concepts.add_element(name=concept['name'], elements=concept['tokens'])
        update_stats({"concept": concept['name'], "len": len(concept['tokens'])}, "create_concept")

def get_concept(cname):
    return project.concepts.get_elements(cname)

def update_tokens(cname, tokens):
    modeler = project.modeler
    project.concepts.add_element(name=cname, elements=tokens)
    update_stats({"concept": cname, "len": len(tokens)}, "update_concept")
    if modeler.count > 0:
        modeler.apply_lfs()
        modeler.fit_label_model()
        global stats
        stats.update(modeler.get_label_model_stats())

def delete_concept(cname):
    project.concepts.delete_concept(cname)
    update_stats({"concept": cname}, "delete_concept")

##### ANNOTATIONS  ############

def make_annotations(cond_dict: dict, text: str, concept=0):
    """Given a text and a condition, return all annotations over the text for that condition
    
    Args:
        k (string): a key describing the type of span to look for
        v (int): Describes the type of the key (ex: a token or regular expression)
        text (string): Text to annotate
        concept (string, optional): Concept to associate the annotation with, or 0 for no concept.
    
    Returns:
        list(dict): all possible annotations on the text for this condition
    """
    annotations = []
    for start, end in find_indices(cond_dict, text):
        annotations.append({
            "id": start,
            "start_offset": start, 
            "end_offset": end, 
            "label": concept,
            "link": None,
            "origin": cond_dict["string"]
        })
    return annotations

def NER(text):
    """Find all spacy-recognized named entities in the text
    
    Args:
        text (string)
    
    Yields:
        dict: Description of the named entity
    """
    with nlp.disable_pipes("tagger", "parser"):
        doc = nlp(text)
        for ent in doc.ents:
             yield {
                "id": '{}_{}'.format(ent.start, ent.label),
                "start_offset": doc[ent.start].idx, 
                "end_offset": doc[ent.end-1].idx + len(doc[ent.end-1].text), 
                "label": ent.label_,
                "explanation": spacy.explain(ent.label_)
            }

def filter_annotations(annotations):
    """Make sure annotations are in order and non-overlapping
    
    Args:
        annotations (list(dict))
    
    Returns:
       (list(dict)): a subset of the provided annotations
    """
    annotations = sorted(annotations, key=lambda x: (x["start_offset"], -x["end_offset"]))
    filtered_ann = []
    left_idx = 0
    for ann in annotations:
        if ann["start_offset"] >= left_idx:
            filtered_ann.append(ann)
            left_idx = ann["end_offset"]
    return filtered_ann

######  INTERACTIONS  ######

SHOW_EXPLANATION_TEXT = False
tutorial_index = 0
if SHOW_EXPLANATION_TEXT:
    tutorial_index = -1

def next_text(annotate_concepts=True, annotate_NER=True):
    modeler = project.modeler

    global tutorial_index, stat_history
    result = None
    
    if result is None:
        result = modeler.next_text()
    
    if annotate_concepts:
        annotations = []
        for concept, tokens in get_all_concepts().items():
            for token in tokens:
                annotations.extend(make_annotations(token, result["text"], concept))
        annotations = sorted(annotations, key=lambda x: (x["start_offset"], -x["end_offset"]))
        result["annotations"] = []
        left_idx = 0
        result["annotations"] = filter_annotations(annotations)
    if annotate_NER:
        result["NER"] = list(NER(result["text"]))
    result["index"] = interactionDB.add(result)
    update_stats(result, "next_text")
    return result

def lf_to_hash(lf_dict):
    def conditions_to_string(cond_list):
        conds = []
        for x in cond_list:
            conds.append("".join([str(val) for val in x.values()]))
        return "-".join(sorted(conds))
    lf_hash = ""
    for key, value in sorted(lf_dict.items()):
        if key == "Conditions":
            lf_hash += conditions_to_string(value)
        else:
            if key in ["Connective", "Direction", "Label", "Context"]:
                lf_hash += "_" + str(value)
    return lf_hash

def submit_interaction(interaction):
    index = interactionDB.add(interaction)
    text = interaction["text"]
    annotations = interaction["annotations"]
    label = int(interaction["label"])
    sent_id = project.text_inv_dict[interaction['text']]

    crnt_syn = Synthesizer(text, annotations, label, DELIMITER, project.concepts.get_dict(), sent_id, project.modeler.ui_label_dict)
    crnt_instances = crnt_syn.run()

    result = {}
    for i in range(len(crnt_instances)):
        crnt_instances[i]["interaction_idx"] = index
        # TODO LFID should be unique given set of conditions, etc
        lf_hash = lf_to_hash(crnt_instances[i])
        crnt_instances[i]["name"] = lf_hash
        result[lf_hash] = crnt_instances[i]
    update_stats({**interaction, "len": len(annotations)}, "submit_interaction")
    update_stats({"suggested_lf_ids": list(result.keys())}, "suggest_lfs")
    return result


def submit_instances(lf_dicts):
    update_stats({"lf_ids": list(lf_dicts.keys())},"submit_lfs")

    modeler = project.modeler
    new_lfs = LF_DB.add_lfs(lf_dicts, all_concepts, project.emb_dict, modeler.ui_label_dict)
    modeler.add_lfs(new_lfs,lf_dicts)
    modeler.fit_label_model()
    global stats
    stats.update(modeler.get_label_model_stats())
    return get_lf_stats()


def get_interaction_idx(idx):
    update_stats({"index": idx}, "get_interaction_idx")
    return interactionDB.get(idx)

def delete_lfs(lf_ids):
    modeler = project.modeler
    for lf_id in lf_ids:
        LF_DB.deactivate(lf_id)
    lf_count = modeler.remove_lfs(lf_ids)
    update_stats({"lf_ids": lf_ids}, "delete_lfs")

    global stats, stat_history

    if lf_count > 0:
        modeler = project.modeler
        modeler.fit_label_model()

        stats.update(modeler.get_label_model_stats())

def get_stats():
    """return statistics over the development set"""
    update_stats({**stats, "data": "dev"}, "stats")
    return stats

def get_lf_stats():
    """return lf-specific statistics over the training and dev sets"""
    modeler = project.modeler
    stats_df = modeler.analyze_lfs()
    stats = {}
    if stats_df is not None:
        stats = json.loads(stats_df.to_json(orient="index"))
    res = LF_DB.update(stats)
    return res


def get_logreg_stats():
    modeler = project.modeler
    stats = modeler.train()
    update_stats({**stats, "data": "test"}, "train_model")
    return stats

def get_lf_label_examples(lf_id):
    modeler = project.modeler
    examples = modeler.lf_examples(lf_id)
    mistakes = modeler.lf_mistakes(lf_id)
    concepts = all_concepts.get_dict()
    update_stats({"lf_id": lf_id, "examples": len(examples), "mistakes": len(mistakes)}, "get_lf_label_examples")
    for result in [examples, mistakes]:
        for ex in result:
            try: 
                annotations = ex['annotations']
            except KeyError:
                annotations = []
            for cond in LF_DB.get(lf_id)["Conditions"]:
                k = cond["string"]
                v = cond["type"]
                if v == KeyType[CONCEPT]:
                    # match a concept
                    assert k in concepts.keys()
                    for token_dict in concepts[k]:
                        annotations.extend(make_annotations(token_dict, ex["text"], k))
                else:
                    annotations.extend(make_annotations(cond, ex["text"], 0))
            ex['annotations'] = filter_annotations(annotations)
    return {"examples": examples, "mistakes": mistakes}

####### Model Persistence #######

models_path = '.'

def download_model():
    """Save model to zip file and send to user's browser
    """
    dirname = os.path.join('datasets',project.dataset_uuid)
    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass
    modeler = project.modeler
    modeler.save(dirname)
    LF_DB.save(dirname)
    interactionDB.save(dirname)
    project.concepts.save(dirname)

    shutil.make_archive(dirname, 'zip', base_dir=dirname)
    return send_from_directory('..', dirname + '.zip', as_attachment=True)

def upload_model(data):

    update_stats({**stats, "data": "dev"}, "final_stats")
    dirname =  os.path.join(models_path, USER_ID + '/' + MODE)
    try:
        os.mkdir(USER_ID)
    except FileExistsError:
        pass
    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass
    modeler = project.modeler
    modeler.save(dirname)

    LF_DB.save(dirname)

    interactionDB.save(dirname)

    all_concepts.save(dirname)

    global stat_history
    stat_history["time_delta"] = stat_history["time"] - stat_history["time"].iloc[0]
    stat_history["tool"] = "Ruler"
    stat_history["task"] = MODE
    stat_history["user"] = USER_ID
    stat_history.to_csv(os.path.join(dirname, "statistics_history.csv"))

    zipfile = USER_ID + '_' + MODE

    shutil.make_archive(zipfile, 'zip', base_dir=dirname)

    print("Files saved to directory " + dirname)
    print('zipped to ' + zipfile)

def load_model(dirname: str):
    project.concepts.load(dirname)
    print(project.concepts.get_dict())

    LF_DB.load(dirname, project.concepts)

    interactionDB.load(dirname)
    global stat_history
    stat_history = pd.read_csv(os.path.join(dirname, "statistics_history.csv"))
    modeler = project.modeler
    modeler.load(dirname)
    modeler.apply_lfs()


def update_stats(new_stats_dict: dict, action: str):
    pass