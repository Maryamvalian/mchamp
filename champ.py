#st
# get the data
    sel = [evoked.ch_names.index(name) for name in gain_info["ch_names"]]
    M = evoked.data[sel]
U, S, _ = _safe_svd(CM, full_matrices=False)
S = S[np.newaxis, :]
del CM
CMinv = np.dot(U / (S + eps), U.T)
CMinvG = np.dot(CMinv, G)
A = np.dot(CMinvG.T, M)  # mult. w. Diag(gamma) in gamma update

if update_mode == 1:
    # MacKay fixed point update (10) in [1]
    numer = gammas ** 2 * np.mean((A * A.conj()).real, axis=1)
    # mean across each source over all time points (axis 1)
    denom = gammas * np.sum(G * CMinvG, axis=0)
    # sums the values across all sensors for each source (along axis=0).
    # aggregates the effect of the interactions across the sensors and gives the total contribution for each source.
elif update_mode == 2:
    # modified MacKay fixed point update (11) in [1]
    numer = gammas * np.sqrt(np.mean((A * A.conj()).real, axis=1))
    denom = np.sum(G * CMinvG, axis=0)  # sqrt is applied below
else:
    raise ValueError("Invalid value for update_mode")

if group_size == 1:
    if denom is None:
        gammas = numer
    else:
        gammas = numer / np.maximum(denom_fun(denom), np.finfo("float").eps)
else:
    numer_comb = np.sum(numer.reshape(-1, group_size), axis=1)
    if denom is None:
        gammas_comb = numer_comb
    else:
        denom_comb = np.sum(denom.reshape(-1, group_size), axis=1)
        gammas_comb = numer_comb / denom_fun(denom_comb)

    gammas = np.repeat(gammas_comb / group_size, group_size)
