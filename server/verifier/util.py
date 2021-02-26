import tensorflow as tf
import numpy as np
from allennlp.data.fields import ArrayField
from allennlp.data import Instance
import pickle
from collections import Counter
import copy
import pandas as pd

def _get_label_majority_vote(instance, treat_tie_as):
    maj_vote = [None] * len(instance['tokens'])

    for i in range(len(instance['tokens'])):
        # Collects the votes for the ith token
        votes = {}
        for lf_labels in instance['WISER_LABELS'].values():
            if lf_labels[i] not in votes:
                votes[lf_labels[i]] = 0
            votes[lf_labels[i]] += 1

        # Takes the majority vote, not counting abstentions
        try:
            del votes['ABS']
        except KeyError:
            pass

        if len(votes) == 0:
            maj_vote[i] = treat_tie_as
        elif len(votes) == 1:
            maj_vote[i] = list(votes.keys())[0]
        else:
            sort = sorted(votes.keys(), key=lambda x: votes[x], reverse=True)
            first, second = sort[0:2]
            if votes[first] == votes[second]:
                maj_vote[i] = treat_tie_as
            else:
                maj_vote[i] = first
    return maj_vote

def get_mv_label_distribution(instances, label_to_ix, treat_tie_as):
    distribution = []
    for instance in instances:
        mv = _get_label_majority_vote(instance, treat_tie_as)
        for vote in mv:
            p = [0.0] * len(label_to_ix)
            p[label_to_ix[vote]] = 1.0
            distribution.append(p)
    return np.array(distribution)

def get_unweighted_label_distribution(instances, label_to_ix, treat_abs_as):
    # Counts votes
    distribution = []
    for instance in instances:
        for i in range(len(instance['tokens'])):
            votes = [0] * len(label_to_ix)
            for vote in instance['WISER_LABELS'].values():
                if vote[i] != "ABS":
                    votes[label_to_ix[vote[i]]] += 1
            distribution.append(votes)

    # For each token, adds one vote for the default if there are none
    distribution = np.array(distribution)
    for i, check in enumerate(distribution.sum(axis=1) == 0):
        if check:
            distribution[i, label_to_ix[treat_abs_as]] = 1

    # Normalizes the counts
    distribution = distribution / np.expand_dims(distribution.sum(axis=1), 1)

    return distribution

def _score_token_accuracy(predicted_labels, gold_labels):
    if len(predicted_labels) != len(gold_labels):
        raise ValueError("Lengths of predicted_labels and gold_labels must match")

    correct = 0
    votes = 0

    for i in range(len(gold_labels)):
        predict = predicted_labels[i]
        gold = gold_labels[i]
        if len(predict) > 2:
            predict = predict[2:]
        if len(gold) > 2:
            gold = gold[2:]
        if predict == gold:
            correct += 1
        if predicted_labels[i] != 'ABS':
            votes += 1

    return correct, votes

def _score_sequence_token_level(predicted_labels, gold_labels):
    if len(predicted_labels) != len(gold_labels):
        raise ValueError("Lengths of predicted_labels and gold_labels must match")

    tp, fp, fn = 0, 0, 0
    for i in range(len(predicted_labels)):
        prediction = predicted_labels[i]
        gold = gold_labels[i]

        if gold[0] == 'I' or gold[0] == 'B':
            if prediction[2:] == gold[2:]:
                tp += 1
            elif prediction[0] == 'I' or prediction[0] == 'B':
                fp += 1
                fn += 1
            else:
                fn += 1
        elif prediction[0] == 'I' or prediction[0] == 'B':
            fp += 1

    return tp, fp, fn

def score_tagging_rules(instances, gold_label_key='tags'):
    lf_scores = {}
    for instance in instances:
        for lf_name, predictions in instance['WISER_LABELS'].items():
            if lf_name not in lf_scores:
                # Initializes true positive, false positive, false negative,
                # correct, and total vote counts
                lf_scores[lf_name] = [0, 0, 0, 0, 0]

            scores = _score_sequence_token_level(predictions, instance[gold_label_key])
            lf_scores[lf_name][0] += scores[0]
            lf_scores[lf_name][1] += scores[1]
            lf_scores[lf_name][2] += scores[2]

            scores = _score_token_accuracy(predictions, instance[gold_label_key])
            lf_scores[lf_name][3] += scores[0]
            lf_scores[lf_name][4] += scores[1]

    # Computes accuracies
    for lf_name in lf_scores.keys():
        if lf_scores[lf_name][3] > 0:
            lf_scores[lf_name][3] = float(lf_scores[lf_name][3]) / lf_scores[lf_name][4]
            lf_scores[lf_name][3] = round(lf_scores[lf_name][3], ndigits=4)
        else:
            lf_scores[lf_name][3] = float('NaN')

    # Collects results into a dataframe
    column_names = ["TP", "FP", "FN", "Token Acc.", "Token Votes"]
    results = pd.DataFrame.from_dict(lf_scores, orient="index", columns=column_names)
    results = pd.DataFrame.sort_index(results)
    return results

def score_predictions(instances, predictions, gold_label_key='tags'):

    tp, fp, fn = 0, 0, 0
    corrects, votes = 0, 0
    offset = 0
    for instance in instances:
        length = len(instance[gold_label_key])
        scores = _score_sequence_token_level(
            predictions[offset:offset+length], instance[gold_label_key])
        tp += scores[0]
        fp += scores[1]
        fn += scores[2]

        scores = _score_token_accuracy(predictions[offset:offset+length], instance[gold_label_key])
        corrects += scores[0]
        votes += scores[1]

        offset += length

    # Collects results into a dataframe
    column_names = ["TP", "FP", "FN", "P", "R", "F1", "ACC", "COVERAGE"]
    p = round(tp / (tp + fp) if tp > 0 or fp > 0 else 0.0, ndigits=4)
    r = round(tp / (tp + fn) if tp > 0 or fn > 0 else 0.0, ndigits=4)
    f1 = round(2 * p * r / (p + r) if p > 0 and r > 0 else 0.0, ndigits=4)
    acc = round(corrects/votes if corrects > 0 and votes > 0 else 0.0, ndigits=4)
    coverage = round(votes/offset if votes > 0 and offset > 0 else 0.0, ndigits=4)
    record = [tp, fp, fn, p, r, f1, acc, coverage]
    index = ["Predictions (Token Level)"]
    results = pd.DataFrame.from_records(
        [record], columns=column_names, index=index)
    results = pd.DataFrame.sort_index(results)
    return results

def clean_inputs(inputs, model):
    if type(model).__name__ == "NaiveBayes":
        inputs = (inputs[0],)
    elif type(model).__name__ == "HMM":
        inputs = (inputs[0], inputs[1])
    else:
        raise ValueError("Unknown model type: %s" % str(type(model)))
    return inputs

def get_generative_model_inputs(instances, label_to_ix):
    label_name_to_col = {}
    link_name_to_col = {}

    # Collects label and link function names
    names = []
    if 'WISER_LABELS' in instances[0]:
        for name in instances[0]['WISER_LABELS']:
            names.append(name)
    for name in names:
        label_name_to_col[name] = len(label_name_to_col)

    # Counts total tokens
    total_tokens = 0
    for doc in instances:
        total_tokens += len(doc['tokens'])

    # Initializes output data structures
    label_votes = np.zeros((total_tokens, len(label_name_to_col)), dtype=np.int)
    seq_starts = np.zeros((len(instances),), dtype=np.int)

    # Populates outputs
    offset = 0
    for i, doc in enumerate(instances):
        seq_starts[i] = offset

        if 'WISER_LABELS' in doc:
            for name in sorted(doc['WISER_LABELS'].keys()):
                for j, vote in enumerate(doc['WISER_LABELS'][name]):
                    label_votes[offset + j, label_name_to_col[name]] = label_to_ix[vote]

        offset += len(doc['tokens'])

    return label_votes, seq_starts

def evaluate_generative_model(model, data, label_to_ix):

    inputs = clean_inputs(get_generative_model_inputs(data, label_to_ix), model)
    ix_to_label = dict(map(reversed, label_to_ix.items()))
    predictions = model.get_most_probable_labels(*inputs)
    label_predictions = [ix_to_label[ix] for ix in predictions]
    return score_predictions(data, label_predictions)

def train_generative_model(model, train_data, dev_data, label_to_ix, config):
    train_inputs = clean_inputs(get_generative_model_inputs(train_data, label_to_ix), model)

    best_p = float('-inf')
    best_r = float('-inf')
    best_f1 = float('-inf')
    best_params = None
    config.epochs=1
    model.estimate_label_model(*train_inputs, config=config)
    results = evaluate_generative_model(model, dev_data, label_to_ix)

    return results

def convert_label_to_ix(label_to_ix_dict, instance_arr):
    def map_b_to_i(label):
        if label.lower().startswith('b-'):
            return 'I-'+label[2:]
        else:
            return label
    return np.array(
            [label_to_ix_dict[map_b_to_i(tag)] for t_nlp in instance_arr for tag in t_nlp.fields['tags']])

def get_label_to_ix(data):
    tag_count = Counter()

    for instance in data:
        for tag in instance['tags']:
            if tag.lower().startswith('i-') or tag.lower().startswith('b-'):
                tag_count['I-'+tag[2:]] += 1
            else:
                tag_count[tag] += 1

    disc_label_to_ix = {value[0]: int(ix) for ix, value in enumerate(tag_count.most_common())}
    gen_label_to_ix = disc_label_to_ix.copy()

    for ix in gen_label_to_ix:
        gen_label_to_ix[ix] += 1
    gen_label_to_ix['ABS'] = 0

    return gen_label_to_ix, disc_label_to_ix

def remove_rule(data, name):
    """
    Removes a tagging or linking rule from a given dataset
    """

    for instance in data:
        if name in instance['WISER_LABELS']:
            del instance['WISER_LABELS'][name]

def get_marginals(i, num_tokens, unary_marginals, pairwise_marginals):

    unary_marginals_list = []
    pairwise_marginals_list = None if pairwise_marginals is None else []

    for _ in range(num_tokens):
        unary_marginals_list.append(unary_marginals[i])

        if pairwise_marginals is not None:
            pairwise_marginals_list.append(pairwise_marginals[i])
        i += 1

    return [unary_marginals_list, pairwise_marginals_list, i]

def get_complete_unary_marginals(unary_marginals, gen_label_to_ix, disc_label_to_ix):

    if unary_marginals is None or gen_label_to_ix is None or disc_label_to_ix is None:
        return unary_marginals

    new_unaries = np.zeros((len(unary_marginals), len(disc_label_to_ix)))

    for k, v in disc_label_to_ix.items():
        if k in gen_label_to_ix:
            new_unaries[:, v] = unary_marginals[:, gen_label_to_ix[k]-1]

    return new_unaries


def get_complete_pairwise_marginals(pairwise_marginals, gen_label_to_ix, disc_label_to_ix):

    if pairwise_marginals is None or gen_label_to_ix is None or disc_label_to_ix is None:
        return pairwise_marginals

    new_pairwise = np.zeros((len(pairwise_marginals), len(disc_label_to_ix), len(disc_label_to_ix)))

    for k1, v1 in disc_label_to_ix.items():
        for k2, v2 in disc_label_to_ix.items():
            if k1 in gen_label_to_ix and k2 in gen_label_to_ix:
                new_pairwise[:, v1, v2] = pairwise_marginals[:, gen_label_to_ix[k1]-1, gen_label_to_ix[k2]-1]

    return new_pairwise

def save_label_distribution(save_path, data, label_votes, seq_lengths, unary_marginals=None,
                            pairwise_marginals=None, gen_label_to_ix=None,
                            disc_label_to_ix=None, save_tags=True):

    unary_marginals = get_complete_unary_marginals(unary_marginals,
                                                   gen_label_to_ix,
                                                   disc_label_to_ix)

    pairwise_marginals = get_complete_pairwise_marginals(pairwise_marginals,
                                                      gen_label_to_ix,
                                                      disc_label_to_ix)

    i = 0
    instances = []
    for j, instance in enumerate(data):
        instance_tokens = instance['tokens']
        fields = {'tokens': instance_tokens}

        if 'sentence_spans' in instance:
            fields['sentence_spans'] = instance['sentence_spans']

        if 'tags' in instance and save_tags:
            fields['tags'] = instance['tags']

        if unary_marginals is not None:
            instance_unary_list, instance_pairwise_list, i = get_marginals(
                i, len(instance_tokens), unary_marginals, pairwise_marginals)

            fields['unary_marginals'] = ArrayField(np.array(instance_unary_list))

            if instance_pairwise_list is not None:
                fields['pairwise_marginals'] = ArrayField(np.array(instance_pairwise_list))
        if j == len(data)-1:
            fields['vote_mask'] = ArrayField(np.max(label_votes[seq_lengths[j]:]+1,1))
        else:
            fields['vote_mask'] = ArrayField(np.max(label_votes[seq_lengths[j]:seq_lengths[j+1]]+1,1))


        instances.append(Instance(fields))

    with open(save_path, 'wb') as f:
        pickle.dump(instances, f)

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

# This is a class te get sentence. The each sentence will be list of tuples with its tag and pos.
class sentence(object):
    def __init__(self, df):
        self.n_sent = 1
        self.df = df
        self.empty = False
        agg = lambda s: [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(),
                                                      s['POS'].values.tolist(),
                                                      s['Tag'].values.tolist())]
        self.grouped = self.df.groupby("Sentence #").apply(agg)
        self.sentences = [s for s in self.grouped]

    def get_text(self):
        try:
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

def get_keras_logreg(input_dim, output_dim=2):
    # set all random seeds
    import tensorflow as tf
    from numpy.random import seed as np_seed
    from random import seed as py_seed
    from snorkel.utils import set_seed as snork_seed
    snork_seed(123)
    tf.random.set_seed(123)
    np_seed(123)
    py_seed(123)

    model = tf.keras.Sequential()
    if output_dim == 1:
        loss = "binary_crossentropy"
        activation = tf.nn.sigmoid
    else:
        loss = "categorical_crossentropy"
        activation = tf.nn.softmax
    dense = tf.keras.layers.Dense(
        units=output_dim,
        input_dim=input_dim,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
    )
    model.add(dense)
    opt = tf.keras.optimizers.Adam(lr=0.01)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    return model



def get_keras_early_stopping(patience=10):
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=10, verbose=1, restore_best_weights=True
    )


def print_instances(instances):
    for instance in instances:
        print(instance)

