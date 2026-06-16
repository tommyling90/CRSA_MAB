from src.transforms.matrix_to_meanings import generate_meaning_space

def joint_prior(m_A, m_B, tau_A, tau_B, y, y_opt):
    return int(y == y_opt and compatible(m_A, m_B, tau_A, tau_B, y_opt))

def y_prior_given_mS_mL(m_A, m_B, tau_A, tau_B, y, y_opt):
    #original formula: P(y, mS, mL) / sum_{y'}(P(y', mS, mL))
    #sum_{y'}(P(y', mS, mL)) = P(y*, mS, mL) since for all the other y's it is 0
    #But whenever y is not y*, the numerator is 0
    #Whenever (m_A[y*] <= tau_A and m_B[y*] <= tau_B) (the "compatible" func) is not satisfied, the denominator is 0
    #So simplify to the following
    if y != y_opt:
        return 0
    if compatible(m_A, m_B, tau_A, tau_B, y_opt):
        return 1
    return 0

def mL_prior_given_mS(m_A, m_B, tau_A, tau_B, y_opt, k, n):
    # original formula: sum_{y'}(P(y', mS, mL)) / sum_{mL}sum_{y'}(P(y', mS, mL))
    # if compatible func not satisfied numerator always 0
    # for denominator, go over the generator and sum the compatible
    # calling Meaning Gen Func here since it will be exhausted after 1 run
    Ml_space_gen = generate_meaning_space(k, n)

    numerator = compatible(m_A, m_B, tau_A, tau_B, y_opt)
    denominator = sum(
        compatible(m_A, m_L, tau_A, tau_B, y_opt)
        for m_L in Ml_space_gen
    )
    if denominator == 0:
        return 0
    return numerator / denominator

def compatible(m_A, m_B, tau_A, tau_B, y_opt):
    return int(m_A[y_opt] <= tau_A and m_B[y_opt] <= tau_B)