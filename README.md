# quora_duplicate
Implementation of semantic question matching with deep learning mentioned in the blog of Quora.

## Description

This project includes 4 RNN models mentioned in the [blog](https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning) of quora. These models have a common structure, which can be summarised as a series steps:

- First generate word embeddings for each sentence.
- Second represent each word by using RNN network (could be LSTM or GRU or bidirectional RNN) which captures context of each sentence.
- Third use computational operations (simple concatenate or attend mechanisms or other ops) to combine several timesteps context vectors into a single vector. This step differentiate the 4 models in this project.
- At last use dense feed forward neural networks for the final classification.

`basic_rnn.py` - implements basic RNN, the first approach in quora blog.

`res_distance.py` - implements the second approach in quora blog.

`decomposable_attention.py` - implements an attention-based approach from [2], the third approach in quora blog 

`compare_aggregate.py` - implements another attention-based approach from [3].

`train_model.py` - train and test BiMPM model.

`data_util.py` - data processing.

`config.py` - hyper-parameters.

`layers.py` - other layers, word embedding layers, context layer, etc.

I also implement BiMPM [4] model, however the matching operations are very complicated which result in a very slow training time. You could find the codes from [here](https://github.com/ijinmao/BiMPM_keras).

## Requirements

- python 2.7
- tensorflow 1.1.0
- keras 2.0.3
- numpy 1.12.1
- pandas 0.19.2
- nltk 3.2.2
- gensim 1.0.1

## References

[[1]](https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning) Lili Jiang, Shuo Chang, and Nikhil Dandekar. "Semantic Question Matching with Deep Learning."

[[2]](https://arxiv.org/abs/1611.01747) Ankur P Parikh, Oscar Tackstrom, Dipanjan Das, and Jakob Uszkoreit. "A decomposable atten- tion model for natural language inference."

[[3]](https://arxiv.org/abs/1606.01933) Shuohang Wang and Jing Jiang. "A compare-aggregate model for matching text sequences."

[[4]](https://arxiv.org/pdf/1702.03814) Zhiguo Wang, Wael Hamza and Radu Florian. "Bilateral Multi-Perspective Matching for Natural Language Sentences."