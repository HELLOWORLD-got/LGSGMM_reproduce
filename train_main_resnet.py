"""Train the paper-oriented model using configs/mdmt.json by default."""
import sys

from lgsgmm.run import main


if __name__ == '__main__':
    main(['train', *sys.argv[1:]])
