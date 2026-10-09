##
 # Harvey Mudd College, CS159
 # Swarthmore College, CS65
 # Copyright (c) 2018 Harvey Mudd College Computer Science Department, Claremont, CA
 # Copyright (c) 2018 Swarthmore College Computer Science Department, Swarthmore, PA
##

import argparse
import sys
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict
from PCLDataReader import PCLLabels, PCLFeatures, PCLVocab
from sklearn.dummy import DummyClassifier 


class BinaryLabels(PCLLabels):
    """ Inherits from the PCLLabels class
        labels true/false for condescension
    """
    

    # reminder: underscore indicates that it's meant to act as a priv function
    # and not be called by external code
    def _extract_label(self, example):
        """Overriding the parent extract label to extract the condescension attribute"""
        return example.get("condescension")

class CategoryLabels(PCLLabels):
    """Inherits from PCLLabels; labels each example with its topical category."""

    def _extract_label(self, example):
        """Overrides the parent to return the example's category attribute (e.g. "refugee")."""
        return example.get("category")

SAVIOR_WORDS = {"help", "hope", "dream", "dreams", "deserve", "plight",
                "unfortunate", "needy", "inspiring", "blessed"}

class MyFeatures(PCLFeatures):
    """Inherits from PCLFeatures; bag-of-words features from the vocabulary plus engineered features."""

    def _get_feature_name(self, i):
        """Returns the name of column i of X (DictVectorizer sorts and keeps only seen features)."""
        return self.vectorizer.feature_names_[i]

    def _get_num_features(self):
        """Returns the number of features the vectorizer learned (columns of X)."""
        return len(self.vectorizer.feature_names_)


    def _extract_features(self, example):
        """Returns the example's in-vocabulary words plus any engineered features."""
        words = self.extract_text(example)
        features = [w for w in words if self.initial_vocab[w] is not None]


        # Change 1: one NUM feature for every token containing a digit
        # features += ["NUM" for w in words if any(c.isdigit() for c in w)]

        # # Change 2: flag exclamation marks
        # if "!" in words:
        #     features.append("HAS_EXCLAMATION")

        # # Change 3: one pooled feature for "savior"-style words
        features += ["SAVIOR_WORD" for w in words if w in SAVIOR_WORDS]

        return features


# class MyFeatures(PCLFeatures):
#     """TODO: your docstring"""

#     def _extract_features(self, example):
#         """TODO: your docstring"""
#         words = self.extract_text(example)
#         return [w for w in words if self.initial_vocab[w] is not None]

#     def _get_feature_name(self, i):
#         """TODO: your docstring"""
#         return self.vectorizer.feature_names_[i]

#     def _get_num_features(self):
#         """TODO: your docstring"""
#         return len(self.vectorizer.feature_names_)


def do_experiment(args):
    """Train and evaluate a Naive Bayes condescension classifier.

    Builds bag-of-words features and binary labels from args.data_file,
    then gets a prediction and confidence for each example in one of two ways:
      - args.test_category: train on every example outside that category and
        predict only the examples inside it (tests topic generalization).
      - args.xvalidate (k): k-fold cross-validation, so every example is
        predicted by a model that never saw it during training.

    Writes one line per prediction to args.output_file:
        <example id> <true|false> <probability of the predicted class>
    """
    # 1. Vocabulary, feature extractor, and labelers.
    vocab = PCLVocab(args.vocabulary, args.vocab_size, args.stop_words)
    features = MyFeatures(vocab)
    binary_labels = BinaryLabels()
    category_labels = CategoryLabels()

    # 2. The classifier (alpha=1.0, i.e. add-one smoothing, by default).
    # clf = DummyClassifier(strategy="most_frequent")
    # clf = DummyClassifier(strategy="stratified", random_state=0)
    clf = MultinomialNB()

    # 3. Feature matrix, condescension labels, and categories, all read
    #    from the same file. --train_size 0 or unset means use everything.
    max_instances = args.train_size if args.train_size else None
    X, ids = features.process(args.data_file, max_instances=max_instances)
    args.data_file.seek(0)
    y = np.array(binary_labels.process(args.data_file, max_instances=max_instances))
    args.data_file.seek(0)
    categories = np.array(category_labels.process(args.data_file, max_instances=max_instances))

    # 4. Class probabilities for each example being predicted.
    if args.test_category is not None:
        if args.test_category not in category_labels.labels:
            sys.exit(f"Unknown category {args.test_category!r}; "
                     f"choose from {sorted(category_labels.labels)}")
        test_cat = category_labels.labels[args.test_category]
        test_idx = np.where(categories == test_cat)[0]
        train_idx = np.where(categories != test_cat)[0]

        clf.fit(X[train_idx], y[train_idx])
        probs = clf.predict_proba(X[test_idx])
        classes = clf.classes_
        out_ids = [ids[i] for i in test_idx]
    else:
        probs = cross_val_predict(clf, X, y, cv=args.xvalidate,
                                  method="predict_proba")
        classes = np.unique(y)
        out_ids = ids

    # 5. Predicted class = the column with the highest probability;
    #    confidence = that probability.
    best_cols = probs.argmax(axis=1)
    predictions = classes[best_cols]
    confidences = probs.max(axis=1)

    # 6. One line per prediction, in the format the eval script expects.
    for ex_id, pred, conf in zip(out_ids, predictions, confidences):
        args.output_file.write(f"{ex_id} {binary_labels[pred]} {conf}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=argparse.FileType('rb'), help="Data file containing labeled training instances")
    parser.add_argument("vocabulary", type=argparse.FileType('r'), help="File containing vocabulary words")
    parser.add_argument("-o", "--output_file", type=argparse.FileType('w'), default=sys.stdout, help="Write predictions to FILE", metavar="FILE")
    parser.add_argument("-v", "--vocab_size", type=int, metavar="N", help="Only count the top N words from the vocab file", default=None)
    parser.add_argument("-s", "--stop_words", type=int, metavar="N", help="Exclude the top N words as stop words", default=None)
    parser.add_argument("--train_size", type=int, metavar="N", help="Only train on the first N instances. N=0 means use all training instances.", default=None)

    eval_group = parser.add_mutually_exclusive_group(required=True)
    eval_group.add_argument("-t", "--test_category")
    eval_group.add_argument("-x", "--xvalidate", type=int)

    args = parser.parse_args()
    do_experiment(args)

    for fp in (args.output_file, args.data_file, args.vocabulary): fp.close()