import json
import sys
from os.path import dirname, abspath
import math
import time
sys_path = dirname(abspath(__file__))
for _ in range(4):
    sys_path = dirname(sys_path)
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

import argparse
from utils import beam_selection, get_sentence_scores, convert_beam2score


def get_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task", default=None, type=str, required=True)
    parser.add_argument("--src", default=None, type=str, required=False)
    parser.add_argument("--save", default=None, type=str, required=False)

    parser.add_argument('--tokenized', action='store_true')

    parser.add_argument("--beam", default=None, required=False)
    parser.add_argument("--summary_size", default=None, type=int, required=False)

    args = parser.parse_args()
    if args.beam:
        if args.beam in ('inf', 'math.inf'):
            args.beam = math.inf
        elif args.beam.isdigit():
            args.beam = int(args.beam)
    
    return args


def build_beam_json_for_bert(src_json_fp, dump_json_fp, beam, summary_size):
    """
        Perform beam search on bert data and dump results to a json file. 
    """
    n_line = 0

    print(f'start processing. Time: {time.clock()}')
    with open(src_json_fp) as f, open(dump_json_fp, 'a') as dump_f:
        for line in f:
            json_obj = json.loads(line)

            doc_sent_list = json_obj['text']
            greedy_labels = json_obj['label'] if 'label' in json_obj else None

            beam2score = beam_selection(
                doc_sent_list=doc_sent_list, 
                abstract_sent_list=json_obj['summary'], 
                summary_size=summary_size, 
                beam_size=beam,
                tokenized=False)[0]
            
            beam2score = convert_beam2score(beam2score)

            dump_json_obj = {'text': json_obj['text'],
                'target': json_obj['summary'],
                'summary_size': summary_size, 
                'beam_size': beam,
                'beam2score': beam2score,
            }
            if greedy_labels:
                dump_json_obj['greedy'] = greedy_labels
            
            dump_f.write(json.dumps(dump_json_obj)+'\n')

            n_line += 1
            if n_line % 1000 == 0:
                print(f'processed {n_line} lines. Time: {time.clock()}')


def build_beam_json_for_bart(src_json_fp, dump_json_fp, beam, summary_size, add_sentence_scores=True):
    """
        Perform beam search on bart data and dump results to a json file. 
    """
    n_line = 0

    SRC_KEY, TGT_KET, GREEDY_KEY = 'src', 'tgt', 'greedy'
    print(f'start processing. Time: {time.clock()}')
    with open(src_json_fp) as f, open(dump_json_fp, 'a') as dump_f:
        for line in f:
            json_obj = json.loads(line)

            doc_sent_list = json_obj[SRC_KEY]
            beam2score = beam_selection(
                doc_sent_list=doc_sent_list, 
                abstract_sent_list=json_obj[TGT_KET], 
                summary_size=summary_size, 
                beam_size=beam,
                tokenized=True)[0]
            
            beam2score = convert_beam2score(beam2score)

            dump_json_obj = {'text': json_obj[SRC_KEY],
                'target': json_obj[TGT_KET],
                'summary_size': summary_size, 
                'beam_size': beam,
                'beam2score': beam2score,
                'greedy': json_obj[GREEDY_KEY],
                'src_sentences': json_obj['src_sentences'],
                'tgt_sentences': json_obj['tgt_sentences'],
            }
            
            if add_sentence_scores:
                sentence_scores = get_sentence_scores(doc_sent_list=doc_sent_list, 
                    abstract_sent_list=json_obj[TGT_KET], 
                    tokenized=True)
                dump_json_obj['sentence_scores'] = sentence_scores
            
            dump_f.write(json.dumps(dump_json_obj)+'\n')

            n_line += 1
            if n_line % 1000 == 0:
                print(f'processed {n_line} lines. Time: {time.clock()}')


if __name__ == '__main__':
    args = get_args()

    if args.task == 'build_beam_json_for_bert':
    
        build_beam_json_for_bert(src_json_fp=args.src, 
            dump_json_fp=args.save, 
            beam=args.beam, 
            summary_size=args.summary_size)

    elif args.task == 'build_beam_json_for_bart':

        build_beam_json_for_bart(src_json_fp=args.src, 
            dump_json_fp=args.save, 
            beam=args.beam, 
            summary_size=args.summary_size,
            add_sentence_scores=True)

    else:
        raise ValueError(args.task)