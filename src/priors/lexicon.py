def lexicon_without_history(m_S, tau_S, u):
    return int(m_S[u] <= tau_S)

def lexicon_with_history(m_S, tau_S, u, w):
    if len(w) != 0:
        utterances = [event["utterance"] for event in w]
        if u in utterances and u != utterances[-1]:
            return 0
    return lexicon_without_history(m_S, tau_S, u)