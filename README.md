# Knowledge Term Weighting Model (KTWM)

## Introduction
This work is the implementation of the paper [*Knowledge-Grounded Dialogue Generation with Term-level De-noising*](). This work is concerned with improving dialogue generation models through fine-grain refined knowledge. The assumption is that the terms in the retrieved knowledge is not equally important. By assigning different weights to terms, the proposed model can filter out some noisy terms on the fly. To achieve this, the KTWM designs a Simulated Response Vectors which is tranformed from the posts to assign weights to knowledge terms. The results show that the proposed model outperforms strong baselines. More details refer to the following paper:

> Knowledge-Grounded Dialogue Generation with Term-level De-noising

## Data Format
In the data folder, all of the datasets are put there. The directionary lever should be like:
* -data
* --wizard
* ---train
* ---valid
* ---test
</br>
For details and data format, referring to the data folder.


## Usage
In the example folder, there is a script 'run.sh'. Go into the example folder and run the following code. BTW, the configuration can be seen and changed in folder 'configuration/'.

```
python train_ktwm.py \
  --exp_name=ktwm \
  --data_set=wizard \
  --src_seq_length=30 \
  --fact_seq_length=30 \
  --fact_number=1 \
```

## Citation

We appreciate anyone who uses this repo or gets insight from this work to cite the paper. Many thanks!

```
@inproceedings{zheng2021ktwm,
  title={Knowledge-Grounded Dialogue Generation with Term-level De-noising},
  author={Zheng, Wen and Zhou, Ke and Milic-Frayling},
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics: Findings},
  year={2021}
}
```
