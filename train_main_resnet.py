"""Train the paper-oriented model using configs/mdmt.json by default."""
import sys

from lgsgmm.run import main
import torch.distributed as dist


if __name__ == '__main__':
    try:
        main(['train', *sys.argv[1:]])
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
