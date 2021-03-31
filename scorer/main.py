import pdb
import logging
import argparse
import os
import glob 
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, precision_recall_fscore_support

import sys
sys.path.append('.')
from format_checker.main import check_format
"""
Scoring of Task 5 with the metrics Average Precision, R-Precision, P@N, RR@N. 
"""

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


MAIN_THRESHOLDS = [1, 3, 5, 10, 20, 30]


def read_gold_and_pred(gold_fpath, pred_fpath):
    """
    TLDR for function :

    If any answer is 'nan' in ground truth, evaluation for that question is not done
    For instance, if q1 is 'no' and all q2-5 are 'nan' in ground truth, then evaluation only for q1 is done
    """

    """
    Read gold and predicted data.
    :param gold_fpath: the original annotated gold file, where the last 4th column contains the labels.
    :param pred_fpath: a file with line_number and score at each line.
    :return: {line_number:label} dict; list with (line_number, score) tuples.
    """

    logging.info("Reading gold predictions from file {}".format(gold_fpath))

    truths = {}
    truths_ids = {}
    for i in range(7):
        truths[i+1] = []
        truths_ids[i+1] = {}
    with open (gold_fpath, "r") as tr_file:
        for i, line in enumerate(tr_file):
            line = line.lower().strip()
            if len(line) == 0:
                break
            # append \t in the beginning for easy indexing of questions (1-based instead of 0-based for Q1 to Q7)
            line = "\t" + str(line)
            # print(line)

            # This will contain [id, tweet, q1_ans, q2_ans, ..., q7_ans]
            questions = line.split("\t")
            # print(questions)

            for j in range(7):
                if questions[j+1] != "nan": 
                    truths[j+1].append(questions[j+1])
                    truths_ids[j+1][i] = i

    # truths : contains the answers for each label in an array, provided the answer is not 'nan'
    # truth_ids : contains the sentence id for each label in a dict, provided the answer is not 'nan'

    logging.info('Reading predicted ranking order from file {}'.format(pred_fpath))
    
    submitted = {}
    for i in range(7):
        submitted[i+1] = []
    with open(pred_fpath) as pred_f:
        for i, line in enumerate(pred_f):
            line = line.lower().strip()
            if len(line) == 0:
                break
            line = "\t" + line
            questions = line.split("\t")

            for j in range(7):
                if i in truths_ids[j+1]: # By default checks in keys of truth_ids
                    submitted[j+1].append(questions[j+1])

    for i in range(7):
        if len(truths[i+1]) != len(submitted[i+1]):
            logging.error('The predictions do not match the lines from the gold file - missing or extra line for questions {}'.format(i+1))
            raise ValueError('The predictions do not match the lines from the gold file - missing or extra line for questions {}'.format(i+1))

    return truths, submitted

def evaluate(truths, submitted, all_classes):
    acc = accuracy_score(truths, submitted)
    f1 = f1_score(truths, submitted, labels=all_classes, average='weighted')
    p_score = precision_score(truths, submitted, labels=all_classes, average='weighted')
    r_score = recall_score(truths, submitted, labels=all_classes, average='weighted')
    
    return acc, f1, p_score, r_score
   

def validate_files(pred_file, gold_file):
    if not check_format(pred_file):
        logging.error('Bad format for pred file {}. Cannot score.'.format(pred_file))
        return False
    return True

def print_single_metric(title, value):
    line_separator = '=' * 120
    logging.info('{:<30}'.format(title) + '{0:<10.4f}'.format(value))
    logging.info(line_separator)

def print_metrics_info():
    line_separator = '=' * 120
    logging.info('Description of the evaluation metrics: ')
    logging.info('!!! THE OFFICIAL METRIC USED FOR THE COMPETITION RANKING IS MEAN AVERAGE PRECISION (MAP) !!!')
    logging.info(line_separator)
    logging.info(line_separator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold_file_path", '-g',
        help="Single string containing the path to file with gold annotations.",
        type=str,
        required=True
    )
    parser.add_argument(
        "--pred_file_path", '-p',
        help="Single string containing the path to file with ranked line_numbers.",
        type=str,
        required=True
    )
    args = parser.parse_args()

    if validate_files(args.pred_file_path, args.gold_file_path):
        logging.info("Started evaluating results for the task ...")

        truths, submitted = _read_gold_and_pred(args.gold_file_path, args.pred_file_path)

        scores = {
            'acc': [],
            'f1': [],
            'p_score': [],
            'r_score': [],
        }
        all_classes = ["yes", "no"]
        for i in range(7):
            acc, f1, p_score, r_score = evaluate(truths[i+1], submitted[i+1], all_classes)
            for metric in scores:
                scores[metric].append(eval(metric))
        logging.info('{:=^120}'.format(' RESULTS for {} '.format(args.pred_file_path)))
        print_single_metric('ACCURACY:', np.mean(scores['acc']))
        print_single_metric('F1:', np.mean(scores['f1']))
        print_single_metric('PRECISION:', np.mean(scores['p_score']))
        print_single_metric('RECALL:', np.mean(scores['r_score']))
        print_metrics_info()
    else:
        print('%sCOULDNT PROCESS THE FILE (%s) %s'%('\033[91m', submission_fp, '\033[0m'))
