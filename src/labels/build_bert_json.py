import json
import os
import sys
from os.path import dirname, abspath, exists
import math
import time
from pathlib import Path
sys_path = dirname(abspath(__file__))
for _ in range(4):
    sys_path = dirname(sys_path)
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

import argparse
from utils import cal_labels_from_beam2score, cal_beam2sentence_score


def get_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task", default=None, type=str, required=True)
    parser.add_argument("--src", default=None, type=str, required=False)
    parser.add_argument("--save", default=None, type=str, required=False)
    parser.add_argument("--prefix", default=None, type=str, required=False)

    parser.add_argument('--tokenized', action='store_true')

    parser.add_argument("--beam", default=None, required=False)
    parser.add_argument("--summary_size", default=None, type=int, required=False)
    parser.add_argument("--oracle_dist", default='uniform', type=str, required=False)
    parser.add_argument("--oracle_dist_topk", default=None, type=int, required=False)

    parser.add_argument('--store_hard_labels', action='store_true')

    parser.add_argument('--no_target_norm', action='store_true')
    parser.add_argument("--norm_gamma", default=1.0, type=float, required=False)
    
    parser.add_argument("--sent_topk_as_positive", default=None, type=int, required=False)
    parser.add_argument("--hyp2sent_pool", default='mean', type=str, required=False)

    args = parser.parse_args()
    if args.beam:
        if args.beam in ('inf', 'math.inf'):
            args.beam = math.inf
        elif args.beam.isdigit():
            args.beam = int(args.beam)
    
    return args


def build_labels(beam_json_obj, 
        oracle_dist, oracle_dist_topk, 
        sent_topk_as_positive, 
        hyp2sent_pool,
        normalize,
        norm_gamma):
    """
        Build sentence labels from a beam record (a json object).
    """
    def build_baseline_labels(beam_json_obj, oracle_dist):
        """
            Build sentence labels for baselines (best and greedy).
        """
        n_sents = len(beam_json_obj['text'])
        labels = [0] * n_sents
        
        if oracle_dist == 'best':
            if not beam_json_obj['beam2score']:
                return labels

            sorted_beam2score = sorted(beam_json_obj['beam2score'].items(), key=lambda item: item[1], reverse=True)
            sids_in_best_beam = [int(sid) for sid in sorted_beam2score[0][0].split(',')]
            for sid in sids_in_best_beam:
                if sid >= n_sents:
                    continue
                labels[sid] = 1
            return labels
        
        elif oracle_dist == 'greedy':
            greedy = beam_json_obj['greedy'][:n_sents]
            labels[:len(greedy)] = greedy
            return labels
        
        else:
            raise NotImplementedError(oracle_dist)

    if oracle_dist in ('best', 'greedy'):
        return build_baseline_labels(beam_json_obj, oracle_dist)
    
    if oracle_dist == 'sentence_rouge':
        beam2sentence_score = cal_beam2sentence_score(
            doc_sent_list=beam_json_obj['text'], 
            abstract_sent_list=beam_json_obj['target'], 
            beam2summary_score=beam_json_obj['beam2score']
        )
    else:
        beam2sentence_score = None

    labels = cal_labels_from_beam2score(
        n_sents=len(beam_json_obj['text']),
        beam2score=beam_json_obj['beam2score'], 
        oracle_dist=oracle_dist,
        oracle_dist_topk=oracle_dist_topk,
        normalize=normalize,
        norm_gamma=norm_gamma,
        sent_topk_as_positive=sent_topk_as_positive,
        beam2sentence_score=beam2sentence_score,
        hyp2sent_pool=hyp2sent_pool,
    )
    return labels


def _get_best(beam2score):
    if not beam2score:
        return []
    
    sorted_beam2score = sorted(beam2score.items(), key=lambda item: item[1], reverse=True)
    best = [int(sid) for sid in sorted_beam2score[0][0].split(',')]
    return best


def build_bert_json(beam_json_fp, 
        oracle_dist, oracle_dist_topk, 
        normalize=True,
        norm_gamma=1.0,
        sent_topk_as_positive=None, 
        hyp2sent_pool='mean',
        tokenized=False,
        store_hard_labels=False,
    ):
    """
        Build json files for Bert training from dumped beam results.
    """
    beam_json_fn = beam_json_fp.split('/')[-1]
    json_dir = Path('/'.join(beam_json_fp.split('/')[:-1]))

    prefix, corpus_type, affix = beam_json_fn.split('.')
    
    if oracle_dist_topk:
        bert_json_fn = f'{prefix}-{oracle_dist}_top_{oracle_dist_topk}_oracle_dist'
    else:
        bert_json_fn = f'{prefix}-{oracle_dist}_oracle_dist'

    if sent_topk_as_positive:
        bert_json_fn += f'-top_{sent_topk_as_positive}_sent_as_pos'
    
    if not normalize:
        bert_json_fn += f'-no_target_norm'

    if normalize and norm_gamma != 1.0:
        bert_json_fn += f'-norm_gamma_{norm_gamma}'

    if hyp2sent_pool != 'mean':
        bert_json_fn += f'-hyp2sent_pool_{hyp2sent_pool}'
    
    if store_hard_labels:
        bert_json_fn += '-hard_and_soft'

    bert_json_fn += f'.{corpus_type}.bert_json'

    bert_json_fp = json_dir / bert_json_fn

    print(f'beam_json_fn: {beam_json_fn}')
    print(f'bert_json_fn: {bert_json_fn}')

    n_line = 0
    with open(beam_json_fp) as f, open(bert_json_fp, 'a') as dump_f:
        for line in f:
            json_obj = json.loads(line)

            labels = build_labels(json_obj, 
                oracle_dist=oracle_dist, 
                oracle_dist_topk=oracle_dist_topk, 
                sent_topk_as_positive=sent_topk_as_positive, 
                hyp2sent_pool=hyp2sent_pool,
                normalize=normalize,
                norm_gamma=norm_gamma
            )
            
            bert_json_obj = {
                'src': json_obj['text'] if tokenized else [s.split() for s in json_obj['text']],
                'tgt': json_obj['target'] if tokenized else [s.split() for s in json_obj['target']],
                'labels': labels,
            }

            if store_hard_labels:
                best = [1 if i in _get_best(json_obj['beam2score']) else 0 for i in range(len(json_obj['text']))]
                bert_json_obj['best'] = best
                # greedy = [1 if i in json_obj['greedy'] else 0 for i in range(len(json_obj['text']))]
                greedy = [0] * len(json_obj['text'])
                greedy[:len(json_obj['greedy'])] = json_obj['greedy']
                bert_json_obj['greedy'] = greedy
                
            if 'src_sentences' in json_obj:
                bert_json_obj['src_sentences'] = json_obj['src_sentences']

            if 'tgt_sentences' in json_obj:
                bert_json_obj['tgt_sentences'] = json_obj['tgt_sentences']
            
            dump_f.write(json.dumps(bert_json_obj)+'\n')

            n_line += 1
            if n_line % 1000 == 0:
                print(f'processed {n_line} lines. Time: {time.clock()}')


def build_bert_json_for_greedy(preproc_json_fp, tokenized=False):
    """ 
        Build .bert_json for greedy labels from preproc json file.
    """
    preproc_json_fn = preproc_json_fp.split('/')[-1]
    prefix, corpus_type, affix = preproc_json_fn.split('.')

    json_dir = Path('/'.join(preproc_json_fp.split('/')[:-1]))
    
    dataset = prefix.split('-')[0]
    if 'cnndm' in dataset:
        bert_json_fp = json_dir / f'greedy.{corpus_type}.bert_json'
    else:
        bert_json_fp = json_dir / f'{dataset}-greedy.{corpus_type}.bert_json'

    n_line = 0
    with open(preproc_json_fp) as f, open(bert_json_fp, 'a') as dump_f:
        for line in f:
            json_obj = json.loads(line)

            assert len(json_obj['greedy']) == len(json_obj['text'])

            bert_json_obj = {
                'src': json_obj['text'] if tokenized else [s.split() for s in json_obj['text']],
                'tgt': json_obj['target'] if tokenized else [s.split() for s in json_obj['target']],
                'labels': json_obj['greedy'],
            }

            dump_f.write(json.dumps(bert_json_obj)+'\n')

            n_line += 1
            if n_line % 1000 == 0:
                print(f'processed {n_line} lines. Time: {time.clock()}')


def discretize_bert_json(prefix):
    corpus_types = ['train', 'test', 'valid']
    json_dir = Path('/'.join(prefix.split('/')[:-1]))
    json_prefx = prefix.split('/')[-1]
    print(f'json_prefx: {json_prefx}')

    for corpus_type in corpus_types:
        bert_json_fp = json_dir / f'{json_prefx}.{corpus_type}.bert_json'
        dump_json_fp = json_dir / f'{json_prefx}.discrete.{corpus_type}.bert_json'
        print(f'bert_json_fp: {bert_json_fp}')
        print(f'dump_json_fp: {dump_json_fp}')

        with open(bert_json_fp) as f, open(dump_json_fp, 'a') as dump_f:
            for line in f:
                json_obj = json.loads(line)
                labels = json_obj['labels']

                discret_labels = [0 if l == 0.0 else 1 for l in labels]
                json_obj['labels'] = discret_labels
                dump_f.write(json.dumps(json_obj)+'\n')


def shard_bert_json(shard_dump_dir):
    if exists(shard_dump_dir):
        raise FileExistsError(shard_dump_dir)

    os.mkdir(shard_dump_dir)
    
    corpus_types = ['train', 'test', 'valid']
    
    json_dir = Path('/'.join(shard_dump_dir.split('/')[:-1]))
    bert_json_prefix = shard_dump_dir.split('/')[-1]
    shard_dump_dir = Path(shard_dump_dir)
    
    print(f'shard_dump_dir: {shard_dump_dir}')

    shard_size = 2000
    for corpus_type in corpus_types:
        bert_json_fp = json_dir / f'{bert_json_prefix}.{corpus_type}.bert_json'
        print(f'bert_json_fp: {bert_json_fp}')

        shard_id = 0
        dataset = []
        with open(bert_json_fp) as f:
            for line in f:
                dataset.append(line.strip('\n'))

                if len(dataset) > shard_size:
                    dump_json_fp = shard_dump_dir / f'{corpus_type}.{shard_id}.json'
                    with open(dump_json_fp, 'a') as dump_f:
                        dump_f.write('\n'.join(dataset))
                    shard_id += 1
                    dataset = []

        if len(dataset) > 0:
            dump_json_fp = shard_dump_dir / f'{corpus_type}.{shard_id}.json'
            with open(dump_json_fp, 'a') as dump_f:
                dump_f.write('\n'.join(dataset))
            shard_id += 1
            dataset = []


if __name__ == '__main__':
    args = get_args()
        
    if args.task == 'build_bert_json':
        build_bert_json(beam_json_fp=args.src, 
            oracle_dist=args.oracle_dist, 
            oracle_dist_topk=args.oracle_dist_topk,
            normalize=not args.no_target_norm,
            norm_gamma=args.norm_gamma,
            sent_topk_as_positive=args.sent_topk_as_positive,
            hyp2sent_pool=args.hyp2sent_pool,
            tokenized=args.tokenized,
            store_hard_labels=args.store_hard_labels)

    elif args.task == 'build_bert_json_for_greedy':
        build_bert_json_for_greedy(preproc_json_fp=args.src, tokenized=args.tokenized)

    elif args.task == 'discretize_bert_json':
        discretize_bert_json(prefix=args.prefix)

    elif args.task == 'shard_bert_json':
        shard_bert_json(shard_dump_dir=args.save)

    else:
        raise ValueError(args.task)