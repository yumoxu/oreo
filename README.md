# Oreo: Text Summarization with Oracle Expection


## Build Summarization Data with Oreo Labels
First download raw summarization data from [this link](https://drive.google.com/file/d/1FG4oiQ6rknIeL2WLtXD0GWyh6pBH9-hX/view) for CNN/DM and [this link](https://drive.google.com/file/d/1PnFCwqSzAUr78uEcA_Q15yupZ5bTAQIb/view) for other datasets (XSum, MultiNews, Reddit and WikiHow). Put the downloaded datasets under `raw_data`. 

You can then build your summarization data with Oreo labels for BertSum as follows:

1. Beam search

Run the following command for `split={train, val, test}`:
```python
split=train && dataset=CNNDM && beam=256 && summary_size=3 && src_json_fn=${split}_${dataset}_bert.jsonl && dump_json_fn=cnndm_bert-beams_${beam}-steps_${summary_size}.${split}.beam_json && py src/labels/build_beam_json.py --task build_beam_json_from_bert --src ~/oreo/raw_data/$src_json_fn --save ~/oreo/json_data/$dump_json_fn --beam $beam --summary_size $summary_size 
```
See `json_data/cnndm_sample.valid.beam_json` for output examples.

2. Build sentence labels

Run the following command for `split={train, val, test}`:
```bash
split=train && oracle_dist=uniform && beam=256 && summary_size=3 && beam_json_fn=cnndm_bert-beams_${beam}-steps_${summary_size}.${split}.beam_json && py src/labels/build_bert_json.py --task build_bert_json --src ~/oreo/json_data/$beam_json_fn --oracle_dist ${oracle_dist} --store_hard_labels --oracle_dist_topk 16
```
See `json_data/cnndm_sample.valid.bert_json` for output examples.

3. Shard Bert files

```bash
oracle_dist=uniform_top_16_oracle_dist && beam=256 && summary_size=3 && save_dir=cnndm_bert-beams_${beam}-steps_${summary_size}-${oracle_dist}-hard_and_soft && py src/labels/build_bert_json.py --task shard_bert_json --save ~/oreo/json_data/${save_dir}
```

4. Format to PyTorch files

```bash
oracle_dist=uniform_top_16_oracle_dist && beam=256 && summary_size=3 && dir_name=cnndm_bert-beams_${beam}-steps_${summary_size}-${oracle_dist}-hard_and_soft && bert_json_dir=~/oreo/json_data/${dir_name} && bert_data_dir=~/oreo/bert_data/${dir_name} && py src/preprocess.py -mode format_to_bert_with_precal_labels -raw_path ${bert_json_dir} -save_path ${bert_data_dir} -lower -n_cpus 1 -log_file ./logs/preprocess.log 
```

## Model training and evaluation
We are currently cleaning the code for model training and evaluation, and will soon upload the code for both supervised and zero-shot summarization experiments. 
We recommend you to check out the official implementation of [BertSum](https://github.com/nlpyang/PreSumm/tree/dev) and [GSum](https://github.com/neulab/guided_summarization), which we used for extractive and abstractive experiments, respectively. 
Note that for a fair comparison between different labeling schemes, we follow their standard training configuration without any additional hyper-parameter optimization (e.g., for our specific labeling scheme). 