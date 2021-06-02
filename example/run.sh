
data_set='wizard'

python train_ktwm.py --exp_name=ktwm --data_set=$data_set --src_seq_length=30 --fact_seq_length=30 --fact_number=1 --vocab_size=30000

