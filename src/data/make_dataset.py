# -*- coding: utf-8 -*-
import click
import logging
import json
from pathlib import Path
from all_symbols import all_symbols

def read_scan_results(input_filepath, email_type):

    results = []

    spam_count = 0
    ham_count = 0

    file = open(input_filepath, 'r')
    lines = file.readlines()

    for line in lines:
        json_result = json.loads(line)
        
        if json_result['action'] in ["reject", "add header", "greylist"]:
            spam_count += 1
        else:
            ham_count += 1

        results.append([])
        for sym in json_result['symbols']:
            results[-1].append(sym)

        results[-1].append(email_type)

    print(email_type)
    print "Ham count: " + str(ham_count)
    print "Spam count: " + str(spam_count)

    return results


def get_unique_symbols(results):

    unique_symbols = set()
    
    for result in results:
        for symbol in result:
            unique_symbols.add(symbol)

    return tuple(unique_symbols)



def make_dataset(results, unique_symbols):

    SPAM_LABEL = 1
    HAM_LABEL = -1

    dataset = []

    for result in results:

        dataset.append([])

        for symbol in unique_symbols:
            if symbol in result:
                dataset[-1].append(1)
            else:
                dataset[-1].append(0)

        if result[-1] == "SPAM":
            dataset[-1].append(SPAM_LABEL)
        else:
            dataset[-1].append(HAM_LABEL)

    return dataset

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
        logger.info("reading ham dataset")
        ham_scan_results = read_scan_results(ham, "HAM")
        logger.info("Read " + str(len(ham_scan_results)) + " ham results")

    if spam:
        logger.info("reading spam dataset")
        spam_scan_results = read_scan_results(spam, "SPAM")
        logger.info("Read " + str(len(spam_scan_results)) + " spam results")

    results = spam_scan_results + ham_scan_results

    unique_symbols = get_unique_symbols(results)

    logger.info("Found " + str(len(unique_symbols)) + " unique symbols")

    dataset = make_dataset(results, all_symbols)

    with open(out, 'w') as f:
        f.write(json.dumps(dataset))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
