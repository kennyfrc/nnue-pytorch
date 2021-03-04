import re
import os
import subprocess
import sys
import time
import argparse


def convert_ckpt(root_dir):
    """ Find the list of checkpoints that are available, and convert those that have no matching .nnue """
    # run96/run0/default/version_0/checkpoints/epoch=3.ckpt, or epoch=3-step=321151.ckpt
    p = re.compile("epoch.*\.ckpt")
    ckpts = []
    for path, subdirs, files in os.walk(root_dir, followlinks=False):
        for filename in files:
            m = p.match(filename)
            if m:
                ckpts.append(os.path.join(path, filename))

    # lets move the .nnue files a bit up in the tree, and get rid of the = sign.
    # run96/run0/default/version_0/checkpoints/epoch=3.ckpt -> run96/run0/nn-epoch3.nnue
    for ckpt in ckpts:
        nnue_file_name = re.sub("default/version_[0-9]+/checkpoints/", "", ckpt)
        nnue_file_name = re.sub(r"epoch\=([0-9]+).*\.ckpt", r"nn-epoch\1.nnue", nnue_file_name)
        if not os.path.exists(nnue_file_name):
            command = "{} serialize.py {} {} ".format(sys.executable, ckpt, nnue_file_name)
            ret = os.system(command)
            if ret != 0:
                print("Error serializing!")

def main():
    # basic setup
    parser = argparse.ArgumentParser(
        description="Finds the strongest .nnue / .ckpt in tree, playing games.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="""The directory where to look, recursively, for .nnue or .ckpt.
                 This directory will be used to store additional files,
                 in particular the ranking (ordo.out)
                 and game results (out.pgn and c_chess.out).""",
    )
    args = parser.parse_args()

    convert_ckpt(args.root_dir)


if __name__ == "__main__":
    main()
