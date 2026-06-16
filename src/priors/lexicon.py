def lexicon_without_history(m_S, tau_S, u):
    return int(m_S[u] <= tau_S)

def lexicon_with_history(m_S, tau_S, u, last_w):
    if u in last_w and u != last_w[-1]:
        return 0
    return lexicon_without_history(m_S, tau_S, u)