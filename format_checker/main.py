import argparse
import re
import logging


"""
This script checks whether the results format for task is correct.
It also provides some warnings about possible errors.

The correct format of the results file is the following:
<q1_pred> <TAB> <q2_pred> <TAB> <q3_pred> <TAB> <q4_pred> <TAB> <q5_pred> <TAB> <q6_pred> <TAB> <q7_pred>

where <qX_pred> is the class prediction for question X, it could be yes/no/nan
"""

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


def check_format(file_path):
    with open(file_path) as out:
        file_content = out.read().strip()
        for i, line in enumerate(file_content.split('\n')):
            if len(line.strip().split('\t')) != 7:
                # 1. Check line format.
                logging.error("Number of things given is not 7: {}".format(line))
                return False
            q1, q2, q3, q4, q5, q6, q7 = line.strip().split('\t')

            if not q1 in ['yes', 'no']:
                logging.error("Wrong format for q1: {}".format(line))
                return False
            if not q6 in ['yes', 'no']:
                logging.error("Wrong format for q6: {}".format(line))
                return False
            if not q7 in ['yes', 'no']:
                logging.error("Wrong format for q7: {}".format(line))
                return False
            if not q2 in ['yes', 'no', 'nan']:
                logging.error("Wrong format for q1: {}".format(line))
                return False
            if not q3 in ['yes', 'no', 'nan']:
                logging.error("Wrong format for q1: {}".format(line))
                return False
            if not q4 in ['yes', 'no', 'nan']:
                logging.error("Wrong format for q1: {}".format(line))
                return False
            if not q5 in ['yes', 'no', 'nan']:
                logging.error("Wrong format for q1: {}".format(line))
                return False
    logging.info("File is of correct format: {}".format(file_path))
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file_path", "-p", required=True, help="The absolute path to the file you want to check.", type=str)
    args = parser.parse_args()
    logging.info("Checking file for the CLARIN hackathon competition: {}".format(args.pred_file_path))
    check_format(args.pred_file_path)