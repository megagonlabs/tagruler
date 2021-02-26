"""Class that tracks the dataset, labels, concepts, and model of a project"""
import numpy as np
import os
import pandas as pd
import sys
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from synthesizer.gll import ConceptWrapper
from synthesizer.gll import Label
from synthesizer.gll import Embeddings
from tqdm import tqdm
from verifier.modeler import Modeler
from werkzeug.utils import secure_filename
# from data.datareader import RestaurantsDatasetReader

tqdm.pandas(desc="model launch progress")

DATASETS_PATH = 'datasets/'

class Project:
    def __init__(self, concepts={}, name=None, dataset_uuid=None, labels=None):
        self.set_status("Initializing")
        self.concepts = ConceptWrapper()
        for concept, elements in concepts.items():
            self.concepts.add_element(concept, elements)
        self.name = name
        self.labels = Label(name)
        if labels is not None:
            self.add_labels(labels)
        self.dataset_uuid = dataset_uuid
        self.launch_progress = 0

    def progress(self):
        return self.launch_progress

    def ready(self):
        for item in ['concepts', 'dataset_uuid', 'labels']:
            if self.__dict__[item] is None:
                print("{} is missing".format(item))
                return False
        return True

    def set_name(self, name):
        """(Optional) name this project
        
        Args:
            name (string)
        """
        self.name = name
        self.labels.change_name(name)


    def assign_dataset(self, dataset_uuid):
        """Assign a dataset to label
        This dataset should already be uploaded
        
        Args:
            dataset_uuid (string): the ID of this dataset
        """
        assert os.path.exists(os.path.join(DATASETS_PATH, dataset_uuid))
        self.dataset_uuid = dataset_uuid
        if self.name is None:
            self.set_name(dataset_uuid)

    def add_labels(self, labels: dict):
        """Define the names of the labels for the project
        
        Args:
            labels (list): Description
        """
        self.status = "Creating labels"
        for lname, value in labels.items():
            self.labels.add_label(lname, value)

    def process_data(self, df, processed_file_path):
        # preprocess data
        df = self.preprocess_dataframe(df)
        # save files
        df.to_csv(processed_file_path)
        return df

    def preprocess_dataframe(self, dataframe):
        """TODO we might want to preprocess POS tags, for instance
        Given a dataframe, pre-compute necessary info.
        
        Args:
            dataframe (DataFrame)
        
        Returns:
            DataFrame
        """
        return dataframe

    def update(self, steps):
        """update launch progress to approximate percentage of launch completed"""
        self.launch_progress += (steps)/self.total

    def set_status(self, status):
        self.status = status
        print(status)

    def launch(self, force_prep=False):
        """Initialize the modeler for a project"""
        #TODO process upploaded csv
        assert self.ready()
        self.launch_progress = 0
        self.set_status("Gathering data")
        if 'O' not in set(self.labels.dict.values()):
            self.add_labels({max(list(self.labels.dict.keys()))+1:'O'})

        processed_file_path = os.path.join(DATASETS_PATH, self.dataset_uuid, 'processed.csv')
        bert_file_path = os.path.join(DATASETS_PATH, self.dataset_uuid, 'processed.bert')
        elmo_file_path = os.path.join(DATASETS_PATH, self.dataset_uuid, 'processed.elmo')
        nlp_file_path = os.path.join(DATASETS_PATH, self.dataset_uuid, 'processed.nlp')
        sbert_file_path = os.path.join(DATASETS_PATH, self.dataset_uuid, 'processed.sbert')
        if os.path.exists(processed_file_path) and not force_prep:
            df = pd.read_csv(processed_file_path)
            if 'span_label' in df.columns:
                df['span_label']=df['span_label'].apply(eval)
            # Let's say loading the file is ~half the launch time
            # (if the file already exists)
            self.total = 2
            self.update(1)
        else:
            datafiles = [os.path.join(DATASETS_PATH, self.dataset_uuid, d) \
                for d in os.listdir(os.path.join(DATASETS_PATH, self.dataset_uuid))]
            df = concat_dataset(datafiles)
            # expand total to account for time it takes to initialize the model
            self.total = len(df)*(1.1) 
            self.set_status("Preprocessing data")
            df = self.process_data(df, processed_file_path)

        # load list of 'allennlp.data.instance's. allennlp.data.instance can store true labels and tag info internally.
        if os.path.exists(nlp_file_path) and not force_prep:
            with open(nlp_file_path, 'rb') as f:
                sentences = pickle.load(f)
        else:
            #TODO define a universal reader for certain format
            # reader = RestaurantsDatasetReader()
            # data = reader.read(processed_file_path)
            #TODO handle when aux files do not exist
            pass
        bert_emb = np.load(bert_file_path, allow_pickle=True)
        elmo_emb = np.load(elmo_file_path, allow_pickle=True)
        sbert_emb = np.load(sbert_file_path, allow_pickle=True)
        for s, b, e, sb in zip(sentences, bert_emb, elmo_emb, sbert_emb):
            s.fields['bert'] = b
            s.fields['sbert'] = sb
            s.fields['elmo'] = e

        df['bert'] = bert_emb
        df['sbert'] = [sb for sb in sbert_emb]
        df['elmo'] = elmo_emb
        df['text_nlp'] = sentences

        columns_to_drop = list(
            set(df.columns).intersection(set(['span_label','file','label'])))
        df = df.drop(columns=columns_to_drop).reset_index()
        # since df['text_nlp'] contains true label info, drop 'labels' column.
        columns_to_drop = list(set(df.columns).difference(set(['index', 'Unnamed: 0', 'text', 'labels', 'split', 'bert', 'sbert',
       'elmo', 'text_nlp'])))
        if len(columns_to_drop) > 0:
            df = df.drop(columns=columns_to_drop)
        df_train = df[df['split']=='train']
        df_dev = df[df['split'] == 'dev']
        df_valid = df[df['split'] == 'valid']
        df_test = df[df['split'] == 'test']

        self.text_inv_dict = dict(
            zip(list(df['text']),list(df.index))
        )

        # TODO split heldout set if necessary
        # for now, passing empty df as heldout set
        df_heldout = df_test

        self.emb_dict = Embeddings(df)

        self.set_status("Initializing modeler")
        self.modeler = Modeler(df_train, df_dev, df_valid, df_test, df_heldout, self.labels, emb_dict=self.emb_dict)

        self.launch_progress = 1.0
        self.set_status("Finished")
        return self.modeler



def initialize_from_task(MODE="reviews"):
    """Load data for the provided mode, preprocess, and initialize model
    
    Args:
        MODE (str, optional): The task for which to load data
    
    Returns:
        Project
    
    Raises:
        Error: If the mode is not recognized
    """
    labels = []
    task_name = ""
    dataset_uuid = ""

    if MODE == "reviews":
        labels = {0: "ASPECT", 1:"OPINION"} # TODO: should be erased before deploy
        task_name = "Restaurant review aspect/opinion extraction: Aspect or Opinion"
        dataset_uuid = "reviews"
    elif MODE == "hotel":
        labels = {0: "ASPECT", 1:"OPINION"} # TODO: should be erased before deploy
        task_name = "Hotel review aspect/opinion extraction: Aspect or Opinion"
        dataset_uuid = "hotel"
    elif MODE == "bc5cdr":
        labels = {0: "CHEMICAL", 1:"DISEASE"} # TODO: should be erased before deploy
        task_name = "Bio-med chemical/disease extraction: Chemical or Disease"
        dataset_uuid = "bc5cdr"
    else:
        raise Error('MODE={} is not recognized.'.format(MODE))

    project = Project(name=task_name, dataset_uuid=dataset_uuid, labels=labels)
    #project.launch()
    return project

def allowed_file(filename: str):
    ALLOWED_EXTENSIONS = {'csv'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def add_file(file, dataset_uuid: str, label_col="label", text_col="text"):
    # save file
    if file and allowed_file(file.filename):
        file_name = secure_filename(file.filename)
        file_path = os.path.join(DATASETS_PATH, dataset_uuid, file_name)
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        file.save(file_path)


def mask_labelled(label_col_series):
    return label_col_series.notnull()


def label_map(l):
    #TODO use programatic way to assign labels
    if l == "B-AS" or l == "I-AS":
        return 0
    elif l == "B-OP" or l == "I-OP":
        return 1
    else:
        return -1

def concat_dataset(datafiles: list, delimiter=None, max_size=3000):
    """Make sure there is a labels column with span labels, and a text column
    concatenate all the relevant files"""
    dfs = []
    for i, filename in enumerate(datafiles, start=1):
        if allowed_file(filename):
            print("--- filename: " + filename)
            df = pd.read_csv(filename, header=0)

            ###### Below is the only part that varies from ruler data prep
            if not 'labels' in df.columns:
                names = ["UNK"]*len(df.columns)
                names[-1] = 'label'
                names[-2] = 'text'
                df = pd.read_csv(filename, names=names, header=0)
            df['span_label'] = df['labels'].apply(lambda x: list(map(label_map, x.split(','))))
            ##### end diff
            assert 'text' in df.columns
            # Add field indicating source file
            df["file"] = filename

            # TODO remove Modeler dependence on field 'label'
            # for now, pass dummy doc-level labels
            df['label'] = np.random.randint(0,2, size=len(df))

            # Remove delimiter chars
            if delimiter is not None:
                df['text'].replace(regex=True, inplace=True, to_replace=delimiter, value=r'')
            dfs.append(df)

    df_full = pd.concat(dfs).sample(frac=1, random_state=123)

    # split the data into labelled and unlabeled
    # TODO right now, we assume everything is labelled
    #labelled = df_full[mask_labelled(df_full.label)]
    labelled = df_full
    #unlabelled = df_full[~mask_labelled(df_full.label)]
    unlabelled = []

    # if all the data provided is labelled, 
    # set some aside to use for interaction examples (training set)
    if len(unlabelled) == 0:
        msk = np.random.rand(len(df_full)) < 0.5
        unlabelled = df_full[msk]
        labelled = df_full[~msk]

    # Make sure we have enough labelled data
    MIN_LABELLED_AMOUNT = 10
    assert len(labelled) >= MIN_LABELLED_AMOUNT,  \
        "Not enough labelled data. \
        (Only {} examples detected)".format(len(labelled))

    labelled = labelled[:min(max_size, len(labelled))].reset_index(drop=True)
    fifth = int(len(labelled)/5)
    labelled.at[:fifth*2, 'split'] = 'dev'
    labelled.at[fifth*2:fifth*3, 'split'] = 'valid'
    labelled.at[fifth*3:, 'split'] = 'test'

    unlabelled = unlabelled[:min(max_size, len(unlabelled))]
    unlabelled['split'] = 'train'

    # reset index
    df_full = pd.concat([labelled, unlabelled])
    df_full = df_full.reset_index(drop=True)
    return df_full

# testing
if __name__=='__main__':
    project = initialize_from_task("Restaurant")
