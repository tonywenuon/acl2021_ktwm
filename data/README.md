# This is the data sets folder. 
It should contain: `dataset_name/train`, `dataset_name/valid` and `dataset_name/test`. In this repo, we set a toy dataset which is randomly extracted from the Wizard of Wikipedia dataset. One can see the required from `wizard/train(valid/test)`.

## train (valid) folder

Four files are required in the train folder: 

* pro_qa.txt
* oracle_sent_fact.txt
* global.src.token.dic
* global.tar.token.dic

In `pro_qa.txt`, its format is: `index \t post \t response`, where `index` is an identifier which should be the same as the index in the `oracle_sent_fact.txt`. `post` and `response` are the normal training (p, r) pair.

In `oracle_sent_fact.txt`, its format is: `index \t knowledge1 \t knowledge2 ......`. it contains all of the retrieved knowledge corresponding each (p, r) pair. Note the `index` should be the same as it in the `pro_qa.txt`. If there is no knowledge for a (p, r) pair, its knowledge should be set to "no_fact" (i.e. `index \t no_fact`). It is worth to note that this knowledge set is retrieved with taking response as query. In the training phase, we can get acess to the response and thus we use response to retrieve high-quality knowledge.

In `global.src.token.dic` and `global.tar.token.dic`, its format is: `word \t index`. It contains all of the vocabularys that are used in the model. In our setting, the words are chosen by ranking their term frequency. `global.src.token.dic` relates to the post and knowledge set and `global.tar.token.dic` refers to the response set, but in our setting, we set these two vocabulary set as the same.

While for the valid folder, the `pro_qa.txt` and `oracle_sent_fact.txt` are exactly the same as them in the train folder.

## test folder
Two files are required in the test folder: 

* pro_qa.txt
* sent_fact.txt

`pro_qa.txt` is the same as the train folder. For `sent_fact.txt`, the only difference is that the knowledge is retrieved by the posts rather than the responses because the responses are not applicable during test phase.

If one would like to use KTWM on your own dataset, simply changing your data to the required format.


