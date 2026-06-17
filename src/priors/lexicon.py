def lexicon_without_history(m_S, tau_S, u):
    return int(m_S[u] <= tau_S)

def lexicon_with_history(m_S, tau_S, u, w):
    if u in w and u != w[-1]:
        return 0
    return lexicon_without_history(m_S, tau_S, u)