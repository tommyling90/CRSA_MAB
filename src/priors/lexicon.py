import numpy as np

def base_lexicon_matrix(meanings, utterances, tau):
    meanings = np.asarray(meanings)
    utterances = np.asarray(utterances, dtype=int)

    ranks = meanings[:, utterances]

    values = (tau - ranks + 1) / tau

    return np.where(ranks <= tau, values, 0.0).astype(float)


def history_lexicon_mask(utterances, history):
    utterances = np.asarray(utterances, dtype=int)

    previous_utterances = [
        int(event["utterance"])
        for event in history
    ]

    if not previous_utterances:
        return np.ones(len(utterances), dtype=float)

    last_utterance = previous_utterances[-1]
    previously_used = set(previous_utterances)

    return np.array(
        [
            0.0
            if int(u) in previously_used and int(u) != last_utterance
            else 1.0
            for u in utterances
        ],
        dtype=float,
    )


def lexicon_matrix(meanings, utterances, tau, history):
    base = base_lexicon_matrix(
        meanings=meanings,
        utterances=utterances,
        tau=tau,
    )

    history_mask = history_lexicon_mask(
        utterances=utterances,
        history=history,
    )

    return base * history_mask[None, :]