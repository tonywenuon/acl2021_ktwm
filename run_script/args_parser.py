
import argparse
import tensorflow as tf

def kwfm_add_arguments(parser):
    parser.add_argument("--data_set", default='wizard', type=str, help="Currently, this can be wizard, reddit or de_en")
    parser.add_argument("--exp_name", default='ktwm', type=str, help="The experiment name.")
    parser.add_argument("--epochs", default=100, type=int, help="Epoch numbers.")
    parser.add_argument("--batch_size", default=50,type=int, help="Batch size.")
    parser.add_argument("--src_seq_length", default=30,type=int, help="Source sequence length")
    parser.add_argument("--tar_seq_length", default=30,type=int, help="Target sequence length")
    parser.add_argument("--fact_seq_length", default=30,type=int, help="Fact sequence length")
    parser.add_argument("--embedding_dim", default=100, type=int, help="Word embedding size.")
    parser.add_argument("--num_heads", default=4,type=int, help="Multi-head numbers.")
    parser.add_argument("--beam_size", default=3,type=int, help="Used for beam search.")
    parser.add_argument("--transformer_depth", default=1,type=int, help="The number of Transformer blocks.")
    parser.add_argument("--fact_number", default=1,type=int, help="How many facts should be injected.")
    parser.add_argument("--vocab_size", default=50000, type=int, help="Vocabulary size.")
    parser.add_argument("--early_stop_patience", default=3, type=int, help="Indicate how many step to show current loss.")
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate.")
    parser.add_argument("--lr_min", default=0.00001, type=float, help="When learning rate decays to lr_min, it stops.")
    parser.add_argument("--lr_decay_patience", default=2, type=float, help="When lr doesn't change for this number, it begin to decay'.")
    parser.add_argument("--checkpoints_dir", default='log', type=str, help="The folder to save checkpoint.")
    parser.add_argument("--outputs_dir", default='outputs', type=str, help="The folder to save test outputs file.")


