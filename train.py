"""Thin delegator to the organized orchestrator in dada_tf.train.

This preserves the original entrypoint `python train.py`.
"""

from dada_tf.train import main

if __name__ == "__main__":
    main()
