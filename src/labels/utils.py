from collections import defaultdict
import re
import random


def _get_ngrams(n, text):
    """Calcualtes n-grams.
    Args:
        n: which n-grams to calculate
        text: An array of tokens
    Returns:
        A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0
    words = sum(sentences, [])
    return _get_ngrams(n, words)


def cal_rouge(evaluated_ngrams, reference_ngrams, beta=1.0):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = (1.0 + beta ** 2) * ((precision * recall) / (beta ** 2 * precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size, beta=1.0, tokenized=False):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    if tokenized:
        abstract = sum(abstract_sent_list, [])
        sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    else:
        abstract = abstract_sent_list
        sents = [_rouge_clean(s).split() for s in doc_sent_list]

    abstract = _rouge_clean(' '.join(abstract)).split()
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams, beta)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams, beta)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return sorted(selected)
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def evaluate(candidate, evaluated_1grams, evaluated_2grams, reference_1grams, reference_2grams):
    candidates_1 = [evaluated_1grams[idx] for idx in candidate]
    candidates_1 = set.union(*map(set, candidates_1))
    candidates_2 = [evaluated_2grams[idx] for idx in candidate]
    candidates_2 = set.union(*map(set, candidates_2))
    rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
    rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
    rouge_score = rouge_1 + rouge_2
    return rouge_score


def _rouge_clean(s):
    return re.sub(r'[^a-zA-Z0-9 ]', '', s)


def convert_beam2score(beam2score):
    new_beam2score = {}
    for beam, score in beam2score.items():
        new_beam2score[','.join([str(sid) for sid in beam])] = score

    return new_beam2score


def beam_selection(doc_sent_list, abstract_sent_list, summary_size, beam_size, tokenized):
    if tokenized:
        abstract = sum(abstract_sent_list, [])
        sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    else:
        abstract = abstract_sent_list
        sents = [_rouge_clean(s).split() for s in doc_sent_list]

    abstract = _rouge_clean(' '.join(abstract)).split()
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    def _merge_and_prune(beam2score, beam_size):
        """
            First we merge duplicated beams, 
                e.g., (1,2) and (2,1) have the same score and will lead to same candidates in the next step.
                So we remove (2,1). 
            
            Then we select `beam_size` sets from deduplicated beams.
        """
        ranked_records = sorted(beam2score.items(), key=lambda item: item[1], reverse=True)
        processed_beam2score = dict()  # key: sorted beam via sentence ids
        for beam, score in ranked_records[:beam_size]:
            processed_beam2score[beam] = score
        
        return processed_beam2score
        
    def _step(step_id, beam2score):
        beams = list(beam2score.keys())
        visited = set()

        for beam in beams:
            if len(beam) != step_id:  # skip finished beams
                continue

            score = beam2score[beam]
            beam = list(beam)

            # select next sentence for the current beam
            for i in range(len(sents)):
                if i in beam:
                    continue
                
                candidate = tuple(sorted(beam + [i]))
                if candidate in visited:
                    continue

                rouge_score = evaluate(candidate, 
                    evaluated_1grams, evaluated_2grams, 
                    reference_1grams, reference_2grams)
                
                visited.add(candidate)

                # only add when sents[i] helps
                # otherwise its credit will be negative
                if rouge_score > score:
                    beam2score[candidate] = rouge_score
        
        # print(f'step: {step_id}, beam2score: {len(beam2score)}')
        beam2score = _merge_and_prune(beam2score, beam_size=beam_size)
        # print(f'step: {step_id}, beam2score (after prune): {len(beam2score)}, beam_size: {beam_size}')
        return beam2score

    beam2score = {tuple(): 0.0}
    for step_id in range(summary_size):
        beam2score = _step(step_id, beam2score)

    if tuple() in beam2score:
        del beam2score[tuple()]

    return beam2score, evaluated_1grams, reference_1grams, evaluated_2grams, reference_2grams


def get_sentence_scores(doc_sent_list, abstract_sent_list, tokenized):
    if tokenized:
        abstract = sum(abstract_sent_list, [])
        sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    else:
        abstract = abstract_sent_list
        sents = [_rouge_clean(s).split() for s in doc_sent_list]

    abstract = _rouge_clean(' '.join(abstract)).split()
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    sentence_scores = [
        evaluate([i], 
            evaluated_1grams, evaluated_2grams, 
            reference_1grams, reference_2grams)
        for i in range(len(sents))
    ]

    return sentence_scores


def cal_beam2sentence_score(doc_sent_list, abstract_sent_list, beam2summary_score):
    abstract = abstract_sent_list
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(s).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    sentence_scores = [0.0] * len(sents)
    for i in range(len(sents)):
        sentence_scores[i] = evaluate([i], 
            evaluated_1grams, evaluated_2grams, 
            reference_1grams, reference_2grams)

    beam2sentence_score = {}
    for beam in beam2summary_score.keys():
        scores = [sentence_scores[int(sid)] for sid in beam.split(',')]

        score = sum(scores) / float(len(scores))
        beam2sentence_score[beam] = score
    
    return beam2sentence_score


def f_scores(p, r, beta):
    return (1.0 + beta ** 2) * ((p * r) / (beta ** 2 * p + r + 1e-8))


def max_min_normalization(values, use_nonzero_as_min=True, gamma=1.0):
    min_v, max_v = min(values), max(values)

    # if max_v == min_v:
    #     scores = [1.0] * len(values)
    #     return scores
    if min_v == 0.0:
        if max_v == min_v:  # all zeros
            return [0.0] * len(values)

        if use_nonzero_as_min:
            min_v = sorted(set(values))[1]
            assert min_v > 0
            if max_v == min_v:
                return [0.0 if v == 0.0 else 1.0 for v in values]
    else:
        if max_v == min_v:
            return [1.0] * len(values)

    assert max_v > min_v

    scores = [max(0.0, (v-min_v)/(max_v-min_v)) for v in values]
    if gamma>1.0:
        scores = [1.0 if s==1.0 else s/gamma for s in scores]
    
    return scores


def binarize_scores(values, sent_topk_as_positive, pad_with_rand=False):
    val2sent_ids = defaultdict(set)
    for sent_id, val in enumerate(values):
        val2sent_ids[val].add(sent_id)
            
    pos_indices = []
    for val in sorted(val2sent_ids.keys(), reverse=True):
        # if len(pos_indices) >= sent_topk_as_positive or val == 0.0:
        #     break

        if len(pos_indices) >= sent_topk_as_positive:
            break
        
        if val == 0.0:
            if not pad_with_rand:
                break

            n_samples = sent_topk_as_positive - len(pos_indices)
            items = random.sample(val2sent_ids[val], min(n_samples, len(val2sent_ids[val])))
            pos_indices.extend(items)
            continue

        pos_indices.extend(val2sent_ids[val])

    scores = [0] * len(values)
    for i in pos_indices:
        scores[i] = 1
    return scores


def make_distribution(values):
    total_mass = float(sum(values))
    if total_mass==0:
        return [1.0/len(values) for _ in values]
    
    return [v/total_mass for v in values]


def cal_labels_from_beam2score(n_sents, 
        beam2score, 
        oracle_dist='uniform', 
        oracle_dist_topk=None,
        normalize=True,
        norm_gamma=1.0,
        sent_topk_as_positive=None,
        beam2sentence_score=None,
        hyp2sent_pool='mean',
    ):
    """
        Calculate sentences labels with expected oracle values, 
        based on their appearances in summary hypotheses found via beam search.
        
        oracle_dist: uniform or weight annealing
        oracle_dist_topk: only set topK beams with prob>0, and the rest to zero

        hyp2sent_pool: 'mean' for oreo and 'max' for ormax.
    """

    def _build_oracle_dist(oracle_dist, oracle_dist_topk, sorted_beam2score, beam2sentence_score=None):
        n_hyps = len(sorted_beam2score)
        dist = [0.0] * n_hyps
        
        end = oracle_dist_topk if oracle_dist_topk else n_hyps

        if oracle_dist == 'uniform':
            dist[:end] = [1.0/end] * end
        
        elif oracle_dist == 'linear_anneal':
            # the kth-ranked hyp has prob of 0
            anneal_rate = 1.0/end
            dist[:end] = [1.0-i*anneal_rate for i in range(end)]
            dist = make_distribution(dist)
        
        elif oracle_dist == 'rouge_anneal':
            dist[:end] = [s for _, s in sorted_beam2score[:end]]
            dist = make_distribution(dist)

        elif oracle_dist == 'sentence_rouge':
            dist[:end] = [beam2sentence_score[beam] for beam, _ in sorted_beam2score[:end]]
            dist = make_distribution(dist)

        elif oracle_dist == 'position_rank_anneal':
            top_beams = [beam for beam, _ in sorted_beam2score[:end]]
            beam_data = [[beam, hyp_rouge_rank, [int(sid) for sid in beam.split(',')]] for hyp_rouge_rank, beam in enumerate(top_beams)]
            sorted_beam_data = sorted(beam_data, key=lambda item: item[-1], reverse=False)   # leading sentences come with higher prob

            anneal_rate = 1.0/end
            for pos_rank, beam_item in enumerate(sorted_beam_data):
                _, hyp_rouge_rank, _ = beam_item
                dist[hyp_rouge_rank] = 1.0 - pos_rank * anneal_rate
            
            dist = make_distribution(dist)

        else:
            raise NotImplementedError(oracle_dist)

        return dist

    values = [0.0] * n_sents
    
    if not beam2score:
        return values
        
    sorted_beam2score = sorted(beam2score.items(), key=lambda item: item[1], reverse=True)
    oracle_dist = _build_oracle_dist(oracle_dist=oracle_dist, 
        oracle_dist_topk=oracle_dist_topk, 
        sorted_beam2score=sorted_beam2score,
        beam2sentence_score=beam2sentence_score,
    )

    all_values = [list() for _ in range(n_sents)]  # for save all related hyp values for each sent

    freqs = [0.0] * n_sents
    for hyp_rank, beam_and_score in enumerate(sorted_beam2score):
        hyp, hyp_val = beam_and_score
        sent_score = hyp_val * oracle_dist[hyp_rank]

        sent_ids = [int(sid) for sid in hyp.split(',')]

        for sent_id in sent_ids:
            if hyp2sent_pool == 'max':
                all_values[sent_id].append(sent_score)
            elif hyp2sent_pool == 'mean':
                values[sent_id] += sent_score
            else:
                raise NotImplementedError(hyp2sent_pool)

            freqs[sent_id] += 1
    
    if hyp2sent_pool == 'max':
        values = [max(val) if val else 0.0 for val in all_values]

    if sent_topk_as_positive:
        sentence_scores = binarize_scores(values, 
            sent_topk_as_positive=sent_topk_as_positive, 
            pad_with_rand=True)
        return sentence_scores
    
    if normalize:
        sentence_scores = max_min_normalization(values, gamma=norm_gamma)
    
    return sentence_scores
