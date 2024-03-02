from abc import ABC
from collections import defaultdict, Counter
import logging

import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
import stanza

class LinguisticFeatures(ABC):
    SENT_LEVEL_FEATURES = ["sentence_length", "tree_depth", "top_constituents",
                            "tense", "subject_number", "object_number"]
    RANDOM = ["random_binary"]
    WORD_LEVEL_FEATURES = ["pos_tags", "smallest_constituents", "word_depth"]
    ALL_FEATURES = SENT_LEVEL_FEATURES + RANDOM

    def __init__(self, words_file="/proj/inductive-bias.shadow/abakalov.data/words_fmri.npy"):
        nltk.download("punkt")

        self.words = np.load(words_file)
        self.text = " ".join(self.words)
        self.text_tokenized = [sent.split() for sent in sent_tokenize(self.text)]
        nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse,constituency',
                              dir="/proj/inductive-bias.shadow/abakalov.trash", use_gpu=False, tokenize_pretokenized=True)
        self.parsed_text = nlp(self.text_tokenized)

    @staticmethod
    def _get_smallest_phrase_constituents(parsed_sent):
        phrase_constituents = ["ADJP", "ADVP", "CONJP", "FRAG", "INTJ", "LST", "NAC", "NP", "NX", "PP", \
                            "PRN", "PRT", "QP", "RRC", "UCP", "VP", "WHADJP", "WHAVP", "WHNP", "WHPP", "X"]
        
        def find_smallest_phrase_constituents(tree, last_constituent=None):
            if len(tree.children) == 0:
                return [last_constituent]
            to_return = []
            if tree.label in phrase_constituents:
                for child in tree.children:
                    to_return = to_return + find_smallest_phrase_constituents(child, last_constituent=tree.label)
            else:
                for child in tree.children:
                    to_return = to_return + find_smallest_phrase_constituents(child, last_constituent=last_constituent)
            return to_return
        
        smallest_constituents = find_smallest_phrase_constituents(parsed_sent.constituency, last_constituent=None)
        assert len(smallest_constituents) == len(parsed_sent.words)
        return smallest_constituents

    @staticmethod
    def _get_feat(word, feat):
        if word is None:
            return None
        if word.feats is None:
            return "UNDEF"
        feats = {feat.split("=")[0]: feat.split("=")[1] for feat in word.feats.split("|")}
        if feat not in feats:
            return "UNDEF"
        return feats[feat]

    @staticmethod
    def _get_word_depth(parsed_sent):
        tree = defaultdict(list)
        word_depth = [0 for i in range(len(parsed_sent.words) + 1)]
        for word in parsed_sent.words:
            tree[word.head].append(word.id)
        word_depth[tree[0][0]] = 0
        def count_words_depth(root_id):
            for word in tree[root_id]:
                word_depth[word] = word_depth[root_id] + 1
                count_words_depth(word)
        count_words_depth(tree[0][0])
        return word_depth

    @staticmethod
    def _get_subj_number(parsed_sent, word_depth):
        potential_subj = []
        for word in parsed_sent.words:
            if word.deprel == "nsubj":
                potential_subj.append(word)
        if len(potential_subj) == 0:
            return None
        subj = sorted(potential_subj, key=lambda subj: word_depth[subj.id])[0]
        return LinguisticFeatures._get_feat(subj, "Number")

    @staticmethod
    def _get_obj_number(parsed_sent, word_depth):
        potential_obj = []
        for word in parsed_sent.words:
            if word.deprel == "obj":
                potential_obj.append(word)
        if len(potential_obj) == 0:
            return None
        obj = sorted(potential_obj, key=lambda obj: word_depth[obj.id])[0]
        return LinguisticFeatures._get_feat(obj, "Number")

    @staticmethod
    def _get_verb_tense(parsed_sent, word_depth):
        potential_verbs = []
        for word in parsed_sent.words:
            feat = LinguisticFeatures._get_feat(word, "Tense")
            if feat is not None and feat != "UNDEF":
                potential_verbs.append(word)
        if len(potential_verbs) == 0:
            return None
        verb = sorted(potential_verbs, key=lambda vb: word_depth[vb.id])[0]
        return LinguisticFeatures._get_feat(verb, "Tense")


    @staticmethod
    def get_features_for_sent(parsed_sent):
        smallest_constituents = LinguisticFeatures._get_smallest_phrase_constituents(parsed_sent)
        
        word_depth =  LinguisticFeatures._get_word_depth(parsed_sent)
        subj_num = LinguisticFeatures._get_subj_number(parsed_sent, word_depth)
        obj_num = LinguisticFeatures._get_obj_number(parsed_sent, word_depth)
        vb_tense = LinguisticFeatures._get_verb_tense(parsed_sent, word_depth)

        return {
            "sentence_length": len(parsed_sent.words),
            "tree_depth": max(word_depth),
            "top_constituents": tuple([child.label for child in parsed_sent.constituency.children[0].children]),
            "tense": vb_tense,
            "subject_number": subj_num,
            "object_number": obj_num,

            "pos_tags": [word.upos for word in parsed_sent.words],
            "smallest_constituents": smallest_constituents,
            "word_depth": word_depth
        }
    
    def _get_features_per_sent(self):
        features_list = []
        for sent in self.parsed_text.sentences:
            features_list.append(LinguisticFeatures.get_features(sent))
        return features_list

    @staticmethod
    def feature_to_class_sentence_length(feature):
        if feature <= 7:
            return 0
        if feature <= 16:
            return 1
        return 3
    
    @staticmethod
    def feature_to_class_tree_depth(feature):
        if feature <= 3:
            return 0
        return 1
    
    @staticmethod
    def feature_to_class_top_constituents(feature):
        if feature == ('NP', 'VP'):
            return 0
        return 1
    
    @staticmethod
    def feature_to_class_tense(feature):
        if str(feature) == "Past":
            return 0
        return 1
    
    @staticmethod
    def feature_to_class_subject_number(feature):
        if str(feature) == "Sing":
            return 0
        return 1
    
    @staticmethod
    def feature_to_class_object_number(feature):
        if str(feature) == "UNDEF":
            return 0
        return 1
    
    @staticmethod
    def feature_to_class_pos_tags(feature):
        if str(feature) == "VERB":
            return 0
        if str(feature) == "NOUN":
            return 1
        if str(feature) == "PRON":
            return 2
        if str(feature) == "ADP":
            return 3
        return 4
    
    @staticmethod
    def feature_to_class_smallest_constituents(feature):
        if str(feature) == "NP":
            return 0
        return 1
    
    @staticmethod
    def feature_to_class_word_depth(feature):
        if feature <= 1:
            return 0
        if feature <= 3:
            return 1
        return 2
    
    FEATURE_TO_CLASS_GET_METHOD = {
        "sentence_length": feature_to_class_sentence_length,
        "tree_depth": feature_to_class_tree_depth,
        "top_constituents": feature_to_class_top_constituents,
        "tense": feature_to_class_tense,
        "subject_number": feature_to_class_subject_number,
        "object_number": feature_to_class_object_number,

        "pos_tags": feature_to_class_pos_tags,
        "smallest_constituents": feature_to_class_smallest_constituents,
        "word_depth": feature_to_class_word_depth
    }

    def get_regression_targets(self, feature):
        assert feature in self.ALL_FEATURES
        if feature in self.SENT_LEVEL_FEATURES:
            features_per_sent = self._get_features_per_sent()
            Y_per_sent = [
                self.FEATURE_TO_CLASS_GET_METHOD[feature](item[feature])
                for item in features_per_sent
            ]
            Y_per_word = []
            for i, sent in self.parsed_text.sentences:
                Y_per_word.extend([Y_per_sent[i]] * len(sent.words))
            return np.array(Y_per_word)
        if feature in self.RANDOM:
            total_num_words = sum([len(sent.words) for sent in self.parsed_text.sentences])
            return np.random.randint(0, 2, size=total_num_words)
        if feature in self.WORD_LEVEL_FEATURES:
            features_per_sent = self._get_features_per_sent()
            Y_per_word = [
                self.FEATURE_TO_CLASS_GET_METHOD[feature](item[feature])
                for features_arr in features_per_sent for item in features_arr
            ]
            return np.array(Y_per_word)
        

