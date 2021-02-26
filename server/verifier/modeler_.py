import numpy as np
import pandas as pd
import os
import pickle
import sys

from math import log
from snorkel.labeling import LFAnalysis
from snorkel.labeling import LabelModel
from flyingsquid.label_model import LabelModel as flModel

from synthesizer.gll import *
from verifier.label_models import HMM, NaiveBayes

from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import *
from labelmodels import LearningConfig
from verifier.util import *


class Modeler:
    def __init__(self, df_train, df_dev, df_valid, df_test, df_heldout, labels, lfs={}, lf_dicts={}, label_model=None,
                 emb_dict={}):

        df_train["seen"] = 0
        self.df_train = df_train.reset_index()
        self.df_dev = df_dev.reset_index()
        self.df_valid = df_valid.reset_index()
        self.df_test = df_test.reset_index()
        self.df_heldout = df_heldout.reset_index()

        # tk_train and tk_dev are token-serialized data frame of df_train and df_dev
        self.tk_train = pd.DataFrame({'text': [tk for t in df_train.text_nlp for tk in t['tokens']]})
        self.tk_dev = pd.DataFrame({'text': [tk for t in df_dev.text_nlp for tk in t['tokens']]})

        self.emb_dict = emb_dict

        self.lfs = lfs
        self.lf_dicts = lf_dicts
        self.count = len(lfs)

        # gen_label_to_ix: dictionary of label_to_index for output of LFs. Example) {"ABS":0, "O":1, "OPINION":2, "ASPECT":3}
        # disc_label_to_ix: dictionary of label_to_index for output of LF aggregator. Example) {"O":0, "OPINION":1, "ASPECT":2}
        # Only difference between two is that gen_label_to_ix includes "ABS".
        self.gen_label_to_ix, self.disc_label_to_ix = get_label_to_ix(
            list(df_train['text_nlp']) + list(df_dev['text_nlp']))

        # inverted dictionary of gen_label_to_ix, disc_label_to_ix
        self.ix_to_gen_label, self.ix_to_disc_label = dict(map(reversed, self.gen_label_to_ix.items())), dict(
            map(reversed, self.disc_label_to_ix.items()))

        self.y_serial_train = np.array(
            [self.gen_label_to_ix[tag] for t_nlp in df_train['text_nlp'].values for tag in t_nlp.fields['tags']])
        self.y_serial_dev = np.array(
            [self.gen_label_to_ix[tag] for t_nlp in df_dev['text_nlp'].values for tag in t_nlp.fields['tags']])
        self.y_serial_test = np.array(
            [self.gen_label_to_ix[tag] for t_nlp in df_test['text_nlp'].values for tag in t_nlp.fields['tags']])
        self.y_serial_valid = np.array(
            [self.gen_label_to_ix[tag] for t_nlp in df_valid['text_nlp'].values for tag in t_nlp.fields['tags']])
        self.y_serial_heldout = np.array(
            [self.gen_label_to_ix[tag] for t_nlp in df_heldout['text_nlp'].values for tag in t_nlp.fields['tags']])

        self.ui_label_dict = dict()
        for i in labels.dict:
            if labels.dict[i] == 'O':
                self.ui_label_dict[i] = 'O'
            elif labels.dict[i] == 'ABSTAIN':
                self.ui_label_dict[i] = 'ABS'
            else:
                for gen_l in self.gen_label_to_ix:
                    if len(gen_l) > 2 and (gen_l.startswith('I-') or gen_l.startswith('B-')) and labels.dict[
                        i].startswith(gen_l[2:4]):
                        self.ui_label_dict[i] = gen_l

        _, self.train_seq_starts = get_generative_model_inputs(list(df_train['text_nlp']), self.gen_label_to_ix)
        _, self.dev_seq_starts = get_generative_model_inputs(list(df_dev['text_nlp']), self.gen_label_to_ix)
        _, self.test_seq_starts = get_generative_model_inputs(list(df_test['text_nlp']), self.gen_label_to_ix)
        _, self.valid_seq_starts = get_generative_model_inputs(list(df_valid['text_nlp']), self.gen_label_to_ix)
        _, self.heldout_seq_starts = get_generative_model_inputs(list(df_heldout['text_nlp']), self.gen_label_to_ix)

        self.l_train = None
        self.l_test = None
        self.l_dev = None
        self.l_heldout = None

        self.sbert_train_matrix = np.array(self.df_train.sbert.tolist())

        self.labels = labels

        # TODO enable users to change MODEL manually
        self.MODEL = FS  # HMM_, NB, MV, SNORKEL

        self.probs_train = None
        self.probs_test = None
        self.probs_dev = None
        self.probs_heldout = None

        self.pred_dev = None

        self.hmm_model = None
        self.nb_model = None
        self.mv_model = None
        self.snorkel_model = None

    def get_lfs(self):
        return list(self.lfs.values())

    def add_lfs(self, new_lfs: dict, new_lf_dicts: dict):
        self.lfs.update(new_lfs)
        self.lf_dicts.update(new_lf_dicts)
        if len(self.lfs) > 0:
            self.apply_lfs(lfs=new_lfs)
        self.count = len(self.lfs)

    def remove_lfs(self, old_lf_ids: list):
        for lf_id in old_lf_ids:
            del self.lfs[lf_id]
            remove_rule(list(self.df_dev['text_nlp']), lf_id)
            remove_rule(list(self.df_train['text_nlp']), lf_id)
        self.count = len(self.lfs)
        return len(self.lfs)

    def apply_lfs(self, lfs=None, apply_test=False):
        if lfs is None:
            lfs = self.lfs
        if apply_test:
            for lf_name in lfs:
                lfs[lf_name].apply(list(self.df_test['text_nlp']))
        else:
            for lf_name in lfs:
                lfs[lf_name].apply(list(self.df_train['text_nlp']))
                lfs[lf_name].apply(list(self.df_dev['text_nlp']))

    def lf_examples(self, lf_id, n=5):
        lf_index = list(self.lfs).index(lf_id)
        l_train = self.l_train[:, lf_index]
        labeled_examples = self.tk_train[l_train != 0]
        labeled_examples = np.where(l_train != 0)[0]
        samples = np.random.choice(labeled_examples, min(n, len(labeled_examples)))
        examples = []
        for sample in samples:
            for i in range(len(self.train_seq_starts)):
                if self.train_seq_starts[i] > sample:
                    break
            examples.append({'text': ' '.join(
                [t.text for t in self.tk_train[self.train_seq_starts[i - 1]:self.train_seq_starts[i]].text])})

        # samples = labeled_examples.sample(min(n, len(labeled_examples)), random_state=13)
        return examples#[{"text": t.text} for t in samples["text"].values]

    def lf_mistakes(self, lf_id, n=5):
        lf = self.lfs[lf_id]
        lf_index = list(self.lfs).index(lf_id)
        l_dev = self.l_dev[:, lf_index]
        labeled_examples = np.where((l_dev != 0) & (l_dev != self.y_serial_dev))[0]
        samples = np.random.choice(labeled_examples,min(n, len(labeled_examples)))
        mistakes = []
        for sample in samples:
            for i in range(len(self.dev_seq_starts)):
                if self.dev_seq_starts[i] > sample:
                    break
            mistakes.append({'text':' '.join([t.text for t in self.tk_dev[self.dev_seq_starts[i-1]:self.dev_seq_starts[i]].text])})
        return mistakes #[{"text": t.text} for t in samples["text"].values]

    def fit_label_model(self):
        self.l_train, _ = get_generative_model_inputs(list(self.df_train['text_nlp']), self.gen_label_to_ix)
        self.l_dev, _ = get_generative_model_inputs(list(self.df_dev['text_nlp']), self.gen_label_to_ix)


        if self.MODEL == HMM_:
            self.hmm_model = HMM(len(self.gen_label_to_ix) - 1, self.count,
                                 init_acc=0.9,
                                 acc_prior=1,
                                 balance_prior=10)
            config = LearningConfig()
            config.epochs = 1
            self.hmm_model.estimate_label_model(self.l_train, self.train_seq_starts, config=config)
            self.pred_dev, self.probs_dev = self.hmm_model.get_most_probable_labels(self.l_dev, self.dev_seq_starts)
            self.probs_dev = np.stack(self.probs_dev)
            self.probs_dev = self.probs_dev / np.sum(self.probs_dev, axis=1).reshape(np.shape(self.probs_dev)[0], 1)
            self.pred_dev_label = [self.ix_to_gen_label[ix] for ix in self.pred_dev]

            self.pred_train, self.probs_train = self.hmm_model.get_most_probable_labels(self.l_train,
                                                                                        self.train_seq_starts)
            self.probs_train = np.stack(self.probs_train)
            self.probs_train = self.probs_train / np.sum(self.probs_train, axis=1).reshape(
                np.shape(self.probs_train)[0], 1)
            self.pred_train_label = [self.ix_to_gen_label[ix] for ix in self.pred_train]
        elif self.MODEL == NB:
            self.nb_model = NaiveBayes(len(self.gen_label_to_ix) - 1, self.count,
                                       init_acc=0.9,
                                       acc_prior=0.01,
                                       balance_prior=1.0)
            config = LearningConfig()
            # config.epochs = 1
            self.nb_model.estimate_label_model(self.l_train, config=config)
            self.probs_dev = self.nb_model.get_label_distribution(self.l_dev)
            self.pred_dev = np.argmax(self.probs_dev, axis=1) + 1
            self.pred_dev_label = [self.ix_to_gen_label[ix] for ix in self.pred_dev]

            self.probs_train = self.nb_model.get_label_distribution(self.l_train)
            self.pred_train = np.argmax(self.probs_train, axis=1) + 1
            self.pred_train_label = [self.ix_to_gen_label[ix] for ix in self.pred_train]
        elif self.MODEL == SNORKEL:
            self.snorkel_model = LabelModel(cardinality=len(self.disc_label_to_ix), verbose=True)
            self.snorkel_model.fit(L_train=self.l_train - 1, n_epochs=1000, lr=0.001, log_freq=100, seed=123)
            self.pred_dev, self.probs_dev = self.snorkel_model.predict(L=self.l_dev - 1, return_probs=True,
                                                                       tie_break_policy='random')
            self.pred_dev_label = [self.ix_to_disc_label[ix] for ix in self.pred_dev]
            self.pred_train, self.probs_train = self.snorkel_model.predict(L=self.l_train - 1, return_probs=True,
                                                                           tie_break_policy='random')
            self.pred_train_label = [self.ix_to_disc_label[ix] for ix in self.pred_train]
        elif self.MODEL == FS and self.l_train.shape[1] >= 3:
            self.fs_model = flModel(self.l_train.shape[1])
            self.fs_model.fit(self.l_train)
            self.pred_dev = self.fs_model.predict(self.l_dev-1)
            self.pred_dev_label = [self.ix_to_disc_label[ix] for ix in self.pred_dev]
            pass
        else:
            self.pred_dev = get_mv_label_distribution(list(self.df_dev['text_nlp']), self.disc_label_to_ix, 'O')
            self.probs_dev = get_unweighted_label_distribution(list(self.df_dev['text_nlp']), self.disc_label_to_ix,
                                                               'O')
            self.pred_dev = np.argmax(self.pred_dev, axis=1)
            self.pred_dev_label = [self.ix_to_disc_label[ix] for ix in self.pred_dev]

            self.pred_train = get_mv_label_distribution(list(self.df_train['text_nlp']), self.disc_label_to_ix, 'O')
            self.probs_train = get_unweighted_label_distribution(list(self.df_train['text_nlp']), self.disc_label_to_ix,
                                                                 'O')
            self.pred_train = np.argmax(self.pred_train, axis=1)
            self.pred_train_label = [self.ix_to_disc_label[ix] for ix in self.pred_train]

    def analyze_lfs(self):
        if len(self.lfs) > 0:
            df = LFAnalysis(L=self.l_train - 1, lfs=self.get_lfs()).lf_summary()
            dev_df = LFAnalysis(L=self.l_dev - 1, lfs=self.get_lfs()).lf_summary(Y=self.y_serial_dev - 1)
            dev_df_acc = score_tagging_rules(list(self.df_dev['text_nlp']))
            dev_df['Recall'] = dev_df['Correct'] / (dev_df['Correct'] + dev_df_acc['FN'] + 0.000001)
            df = df.merge(dev_df, how="outer", suffixes=(" Training", " Dev."), left_index=True, right_index=True)
            return df
        return None

    def get_label_model_stats(self):
        res = score_predictions(list(self.df_dev['text_nlp']), self.pred_dev_label)
        examples_with_labels = self.y_serial_dev[self.y_serial_dev > self.gen_label_to_ix['O']]
        predicted_labels = self.pred_dev[self.y_serial_dev > self.gen_label_to_ix['O']]
        predicted_labels = predicted_labels[predicted_labels > self.disc_label_to_ix['O']]
        label_coverage = len(predicted_labels) / len(examples_with_labels)
        result = {
            'precision': res['P'][0],
            'recall': res['R'][0],
            'f1': res['F1'][0],
            'label_coverage': label_coverage,
        }
        return result

    def get_heldout_stats(self):
        # TODO incomplete
        if self.l_heldout is not None:
            return self.majority_model.score(L=self.l_heldout, Y=self.ys_heldout, metrics=["f1", "precision", "recall"])
        return {}

    def train(self):
        """
        if self.MODEL == 'Snorkel':
            self.train_label_predicted = self.label_model.predict(self.l_train, tie_break_policy='abstain')
        else:
            self.train_label_predicted = self.majority_model.predict(self.l_train, tie_break_policy='abstain')
        """
        pred_train_filtered = self.pred_train[self.pred_train != 0]

        if len(pred_train_filtered) == 0:
            print("Labeling functions cover none of the training examples!", file=sys.stderr)
            return {"F1-score": 0}

        sent_no_train = ['Sentence: {}'.format(i + 1) for i, s in enumerate(self.df_train['text_nlp']) for tk in
                         s['tokens']]
        words_train = [tk.text for i, s in enumerate(self.df_train['text_nlp']) for tk in s['tokens']]
        poses_train = [tk.pos_ for i, s in enumerate(self.df_train['text_nlp']) for tk in s['tokens']]
        tags_train = self.pred_train_label
        df_train = pd.DataFrame({
            'Sentence #': sent_no_train,
            'Word': words_train,
            'POS': poses_train,
            'Tag': tags_train
        })

        getter_train = sentence(df_train)
        sentences_train = [" ".join([s[0] for s in sent]) for sent in getter_train.sentences]
        sent_train = getter_train.get_text()
        sentences_train = getter_train.sentences

        X_train = [sent2features(s) for s in sentences_train]
        y_weak_train = [sent2labels(s) for s in sentences_train]

        sent_no_test = ['Sentence: {}'.format(i + 1) for i, s in enumerate(self.df_test['text_nlp']) for tk in
                        s['tokens']]
        words_test = [tk.text for i, s in enumerate(self.df_test['text_nlp']) for tk in s['tokens']]
        poses_test = [tk.pos_ for i, s in enumerate(self.df_test['text_nlp']) for tk in s['tokens']]
        y_test_label = [self.ix_to_gen_label[tag] for tag in self.y_serial_test]
        df_test = pd.DataFrame({
            'Sentence #': sent_no_test,
            'Word': words_test,
            'POS': poses_test,
            'Tag': y_test_label
        })

        getter_test = sentence(df_test)
        sentences_test = [" ".join([s[0] for s in sent]) for sent in getter_test.sentences]
        sent_test = getter_test.get_text()
        sentences_test = getter_test.sentences

        X_test = [sent2features(s) for s in sentences_test]
        y_true_test = [sent2labels(s) for s in sentences_test]

        crf = CRF(algorithm='lbfgs',
                  c1=0.1,
                  c2=0.1,
                  max_iterations=100,
                  all_possible_transitions=False)
        crf.fit(X_train, y_weak_train)
        # Predicting on the test set.
        y_pred = crf.predict(X_test)
        labels = [l[1] for l in self.ui_label_dict.items()]
        if 'ABS' in labels:
            labels.remove('ABS')
        if 'O' in labels:
            labels.remove('O')

        return self.get_stats(y_true_test, y_pred, labels)

    def get_heldout_lr_stats(self):
        X_heldout = self.vectorizer.transform(self.df_heldout.text.tolist())

        preds_test = self.keras_model.predict(x=X_heldout).argmax(axis=1)
        return self.get_stats(self.Y_heldout, preds_test)

    def get_stats(self, y_true_test, y_pred, labels):

        f1 = flat_f1_score(y_true_test, y_pred, average='weighted', labels=labels)
        r = flat_recall_score(y_true_test, y_pred, average='weighted', labels=labels)
        p = flat_precision_score(y_true_test, y_pred, average='weighted', labels=labels)
        a = flat_accuracy_score(y_true_test, y_pred)

        precisions = flat_precision_score(y_true_test, y_pred, average=None, labels=labels)
        recalls = flat_recall_score(y_true_test, y_pred, average=None, labels=labels)
        f1s = flat_f1_score(y_true_test, y_pred, average=None, labels=labels)
        res = {
            "F1-score": f1,
            "accuracy": a,
            'Precision': p,
            'Recall': r,
        }
        for i in range(len(precisions)):
            res['Recall' + str(i)] = recalls[i]
            res['Precision' + str(i)] = precisions[i]
            res['F1' + str(i)] = f1s[i]
        return res

    def entropy(self, prob_dist):
        # return(-(L_row_i==-1).sum())
        return -sum([x * log(x) for x in prob_dist if x > 0 and x < 1])

    def get_entropy_per_sent(self, idx):
        train_seqs = np.append(self.train_seq_starts, len(self.probs_train))
        entropy = [np.mean(np.sum(-self.probs_train[train_seqs[b]:train_seqs[b+1]] * np.log(self.probs_train[train_seqs[b]:train_seqs[b+1]]+1e-8), axis=1)) for b in idx]
        return entropy

    def next_text_fp(self):

        false_positives = np.where((self.pred_dev != 0) & (self.pred_dev != self.y_serial_dev))[0]
        sample_dev = np.random.choice(false_positives,1)[0]
        for i in range(len(self.dev_seq_starts)):
            if self.dev_seq_starts[i] > sample_dev:
                dev_fp_id = i
                break
        closest_id = np.argmax(self.sbert_train_matrix @ self.df_dev.sbert[dev_fp_id])

        return {"text": self.df_train['text'][closest_id], "id": int(closest_id)}

    def next_text(self, active_type=FALSE_POSITIVE):
        if self.pred_dev is not None and active_type==FALSE_POSITIVE:
            return self.next_text_fp()
        subset_size = 50

        min_times_seen = self.df_train["seen"].min()
        least_seen_examples = self.df_train[self.df_train["seen"] == min_times_seen]
        if ((len(self.lfs) == 0) or (len(least_seen_examples)==1) or (self.l_train is None) or (self.probs_train is None)):
            #return one of the least seen examples, chosen randomly
            res_idx = least_seen_examples.sample(1).index[0]
        else:
            #TODO: apply span-level model, majority model
            #take a sample of size subset_size, compute entropy, and return the example with highest entropy
            subset = least_seen_examples.sample(min(subset_size, len(least_seen_examples)))
            #l_train = self.l_train[self.tk_train.index.isin(subset.index)]

            entropy = np.array(self.get_entropy_per_sent(np.array(subset.index))) # get entropy for each text example

            subset = subset[entropy==max(entropy)]
            res_idx = subset.sample(1).index[0]
        self.df_train.at[res_idx, "seen"] += 1
        return {"text": self.df_train.at[res_idx, "text"], "id": int(res_idx)}

    def text_at(self, index):
        self.df_train.at[index, "seen"] += 1
        return {"text": self.df_train.at[index, "text"], "id": int(index)}

    def save(self, dir_name):
        #TODO save models
        self.apply_lfs(apply_test=True)
        self.l_test, _ = get_generative_model_inputs(list(self.df_test['text_nlp']), self.gen_label_to_ix)
        if self.MODEL != HMM_:
            self.hmm_model = HMM(len(self.gen_label_to_ix) - 1, self.count,
                                 init_acc=0.9,
                                 acc_prior=1,
                                 balance_prior=10)
            config = LearningConfig()
            config.epochs = 1
            self.hmm_model.estimate_label_model(self.l_train, self.train_seq_starts, config=config)
        if self.MODEL != NB:
            self.nb_model = NaiveBayes(len(self.gen_label_to_ix) - 1, self.count,
                                       init_acc=0.9,
                                       acc_prior=0.01,
                                       balance_prior=1.0)
            config = LearningConfig()
            self.nb_model.estimate_label_model(self.l_train, config=config)
        if self.MODEL != SNORKEL:
            self.snorkel_model = LabelModel(cardinality=len(self.disc_label_to_ix), verbose=True)
            self.snorkel_model.fit(L_train=self.l_train - 1, n_epochs=1000, lr=0.001, log_freq=100, seed=123)

        train_data = list(self.df_train['text_nlp'])
        dev_data = list(self.df_dev['text_nlp'])
        test_data = list(self.df_test['text_nlp'])

        self.pred_dev, self.probs_dev = self.hmm_model.get_most_probable_labels(self.l_dev, self.dev_seq_starts)
        self.probs_dev = np.stack(self.probs_dev)
        self.probs_dev = self.probs_dev / (
                    np.sum(self.probs_dev, axis=1).reshape(np.shape(self.probs_dev)[0], 1) + 0.0000001)
        self.pred_dev_label = [self.ix_to_gen_label[ix] for ix in self.pred_dev]

        self.pred_train, self.probs_train = self.hmm_model.get_most_probable_labels(self.l_train, self.train_seq_starts)
        self.probs_train = np.stack(self.probs_train)
        self.probs_train = self.probs_train / (
                    np.sum(self.probs_train, axis=1).reshape(np.shape(self.probs_train)[0], 1) + 0.0000001)
        self.pred_train_label = [self.ix_to_gen_label[ix] for ix in self.pred_train]

        self.pred_test, self.probs_test = self.hmm_model.get_most_probable_labels(self.l_test, self.test_seq_starts)
        self.probs_test = np.stack(self.probs_test)
        self.probs_test = self.probs_test / (
                    np.sum(self.probs_test, axis=1).reshape(np.shape(self.probs_test)[0], 1) + 0.0000001)
        self.pred_test_label = [self.ix_to_gen_label[ix] for ix in self.pred_test]

        save_label_distribution(dir_name + '/' + 'train_data_hmm.pkl', label_votes=self.l_train,
                                seq_lengths=self.train_seq_starts,
                                data=train_data,
                                unary_marginals=self.probs_train,
                                gen_label_to_ix=self.gen_label_to_ix,
                                disc_label_to_ix=self.disc_label_to_ix,
                                save_tags=True)
        save_label_distribution(dir_name + '/' + 'dev_data_hmm.pkl', label_votes=self.l_dev,
                                seq_lengths=self.dev_seq_starts,
                                data=dev_data,
                                unary_marginals=self.probs_dev,
                                gen_label_to_ix=self.gen_label_to_ix,
                                disc_label_to_ix=self.disc_label_to_ix,
                                save_tags=True)
        save_label_distribution(dir_name + '/' + 'test_data_hmm.pkl', label_votes=self.l_test,
                                seq_lengths=self.test_seq_starts,
                                data=test_data,
                                unary_marginals=self.probs_test,
                                gen_label_to_ix=self.gen_label_to_ix,
                                disc_label_to_ix=self.disc_label_to_ix,
                                save_tags=True)

        self.pred_dev = get_mv_label_distribution(list(self.df_dev['text_nlp']), self.disc_label_to_ix, 'O')
        self.probs_dev = get_unweighted_label_distribution(list(self.df_dev['text_nlp']), self.disc_label_to_ix, 'O')
        self.pred_dev = np.argmax(self.pred_dev, axis=1)
        self.pred_dev_label = [self.ix_to_disc_label[ix] for ix in self.pred_dev]

        self.pred_train = get_mv_label_distribution(list(self.df_train['text_nlp']), self.disc_label_to_ix, 'O')
        self.probs_train = get_unweighted_label_distribution(list(self.df_train['text_nlp']), self.disc_label_to_ix,
                                                             'O')
        self.pred_train = np.argmax(self.pred_train, axis=1)
        self.pred_train_label = [self.ix_to_disc_label[ix] for ix in self.pred_train]

        self.pred_test = get_mv_label_distribution(list(self.df_test['text_nlp']), self.disc_label_to_ix, 'O')
        self.probs_test = get_unweighted_label_distribution(list(self.df_test['text_nlp']), self.disc_label_to_ix, 'O')
        self.pred_test = np.argmax(self.pred_test, axis=1)
        self.pred_test_label = [self.ix_to_disc_label[ix] for ix in self.pred_test]

        save_label_distribution(dir_name + '/' + 'train_data_mv.pkl', label_votes=self.l_train,
                                seq_lengths=self.train_seq_starts,
                                data=train_data,
                                unary_marginals=self.probs_train,
                                gen_label_to_ix=self.gen_label_to_ix,
                                disc_label_to_ix=self.disc_label_to_ix,
                                save_tags=True)
        save_label_distribution(dir_name + '/' + 'dev_data_mv.pkl', label_votes=self.l_dev,
                                seq_lengths=self.dev_seq_starts,
                                data=dev_data,
                                unary_marginals=self.probs_dev,
                                gen_label_to_ix=self.gen_label_to_ix,
                                disc_label_to_ix=self.disc_label_to_ix,
                                save_tags=True)
        save_label_distribution(dir_name + '/' + 'test_data_mv.pkl', label_votes=self.l_test,
                                seq_lengths=self.test_seq_starts,
                                data=test_data,
                                unary_marginals=self.probs_test,
                                gen_label_to_ix=self.gen_label_to_ix,
                                disc_label_to_ix=self.disc_label_to_ix,
                                save_tags=True)

        self.probs_dev = self.nb_model.get_label_distribution(self.l_dev)
        self.pred_dev = np.argmax(self.probs_dev, axis=1) + 1
        self.pred_dev_label = [self.ix_to_gen_label[ix] for ix in self.pred_dev]

        self.probs_train = self.nb_model.get_label_distribution(self.l_train)
        self.pred_train = np.argmax(self.probs_train, axis=1) + 1
        self.pred_train_label = [self.ix_to_gen_label[ix] for ix in self.pred_train]

        self.probs_test = self.nb_model.get_label_distribution(self.l_test)
        self.pred_test = np.argmax(self.probs_test, axis=1) + 1
        self.pred_test_label = [self.ix_to_gen_label[ix] for ix in self.pred_test]

        save_label_distribution(dir_name + '/' + 'train_data_nb.pkl', label_votes=self.l_train,
                                seq_lengths=self.train_seq_starts,
                                data=train_data,
                                unary_marginals=self.probs_train,
                                gen_label_to_ix=self.gen_label_to_ix,
                                disc_label_to_ix=self.disc_label_to_ix,
                                save_tags=True)
        save_label_distribution(dir_name + '/' + 'dev_data_nb.pkl', label_votes=self.l_dev,
                                seq_lengths=self.dev_seq_starts,
                                data=dev_data,
                                unary_marginals=self.probs_dev,
                                gen_label_to_ix=self.gen_label_to_ix,
                                disc_label_to_ix=self.disc_label_to_ix,
                                save_tags=True)
        save_label_distribution(dir_name + '/' + 'test_data_nb.pkl', label_votes=self.l_test,
                                seq_lengths=self.test_seq_starts,
                                data=test_data,
                                unary_marginals=self.probs_test,
                                gen_label_to_ix=self.gen_label_to_ix,
                                disc_label_to_ix=self.disc_label_to_ix,
                                save_tags=True)
        self.pred_dev, self.probs_dev = self.snorkel_model.predict(L=self.l_dev - 1, return_probs=True,
                                                                   tie_break_policy='random')
        self.pred_dev_label = [self.ix_to_disc_label[ix] for ix in self.pred_dev]
        self.pred_train, self.probs_train = self.snorkel_model.predict(L=self.l_train - 1, return_probs=True,
                                                                       tie_break_policy='random')
        self.pred_train_label = [self.ix_to_disc_label[ix] for ix in self.pred_train]
        self.pred_test, self.probs_test = self.snorkel_model.predict(L=self.l_test - 1, return_probs=True,
                                                                     tie_break_policy='random')
        self.pred_test_label = [self.ix_to_disc_label[ix] for ix in self.pred_test]

        save_label_distribution(dir_name + '/' + 'train_data_snorkel.pkl', label_votes=self.l_train,
                                seq_lengths=self.train_seq_starts,
                                data=train_data,
                                unary_marginals=self.probs_train,
                                gen_label_to_ix=self.gen_label_to_ix,
                                disc_label_to_ix=self.disc_label_to_ix,
                                save_tags=True)
        save_label_distribution(dir_name + '/' + 'dev_data_snorkel.pkl', label_votes=self.l_dev,
                                seq_lengths=self.dev_seq_starts,
                                data=dev_data,
                                unary_marginals=self.probs_dev,
                                gen_label_to_ix=self.gen_label_to_ix,
                                disc_label_to_ix=self.disc_label_to_ix,
                                save_tags=True)
        save_label_distribution(dir_name + '/' + 'test_data_snorkel.pkl', label_votes=self.l_test,
                                seq_lengths=self.test_seq_starts,
                                data=test_data,
                                unary_marginals=self.probs_test,
                                gen_label_to_ix=self.gen_label_to_ix,
                                disc_label_to_ix=self.disc_label_to_ix,
                                save_tags=True)

    def load(self, dir_name):
        #TODO incomplete
        with open(os.path.join(dir_name, 'model_lfs.pkl'), "rb") as file:
            lfs = pickle.load(file)
            """label_model = LabelModel.load(os.path.join(dir_name, 'label_model.pkl'))
            self.lfs = lfs
            self.label_model = label_model
            """
