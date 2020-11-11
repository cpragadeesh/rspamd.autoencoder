# -*- coding: utf-8 -*-
import click
import logging
import json
from pathlib import Path


def read_scan_results(input_filepath, email_type):

    results = []

    file = open(input_filepath, 'r')
    lines = file.readlines()

    for line in lines:
        json_result = json.loads(line)

        results.append([])
        for sym in json_result['symbols']:
            results[-1].append(sym)

        results[-1].append(email_type)

    return results


@click.command()
@click.option('--ham', type=click.Path(exists=True))
@click.option('--spam', type=click.Path(exists=True))
@click.option('--out', type=click.Path())
def main(ham, spam, out):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).

        python make_dataset.py --ham ../../data/processed/ham.scan_results.txt --spam ../../data/processed/spam.scan_results.txt --out ../../data/processed/dataset.txt
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    ham_scan_results = []
    spam_scan_results = []

    if ham:
        ham_scan_results = read_scan_results(ham, "HAM")

    if spam:
        spam_scan_results = read_scan_results(spam, "SPAM")

    with open(out, 'w') as f:
        f.write(json.dumps(ham_scan_results + spam_scan_results))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
