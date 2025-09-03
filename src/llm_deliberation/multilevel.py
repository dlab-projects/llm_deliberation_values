import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import multiprocessing as mp

K = 5

model_idxs = {
    "GPT": 0,
    "Claude": 1,
    "Gemini": 2
}

def verdict2num(verdict):
    if verdict == "NTA":
        return 0
    elif verdict == "YTA" or verdict == "YWBTA":
        return 1
    elif verdict == "NAH":
        return 2
    elif verdict == "ESH":
        return 3
    elif verdict == "INFO":
        return 4
    else:
        raise ValueError(f"Unknown verdict: {verdict}")


def make_verdict2num_mapper(verdict2num):
    """
    Wrap verdict2num so it works whether it is a dict or a function.
    Returns a function f(verdict_str) -> int
    """
    if callable(verdict2num):
        return verdict2num
    elif isinstance(verdict2num, dict):
        return lambda v: verdict2num[v]
    else:
        raise TypeError("verdict2num must be dict or callable")


def df_to_verdict_lists(df, n_agents):
    if n_agents == 2:
        return [df["Agent_1_verdicts"].tolist(),
                df["Agent_2_verdicts"].tolist()]
    elif n_agents == 3:
        return [df["Agent_1_verdicts"].tolist(),
                df["Agent_2_verdicts"].tolist(),
                df["Agent_3_verdicts"].tolist()]
    else:
        raise ValueError("n_agents must be 2 or 3")

def concat_experiments(exp_dicts):
    if not exp_dicts:
        raise ValueError("No experiments provided")
    keys = exp_dicts[0].keys()
    out = {}
    for k in keys:
        out[k] = np.concatenate([e[k] for e in exp_dicts], axis=0) if len(exp_dicts) > 1 else exp_dicts[0][k]
    return out

def build_all_data(sync_exps, sync_model_ids, rr_exps, rr_model_ids,
                   missing_tokens=("", None)):
    all_parts = []

    # --- synchronous (2-way) ---
    for df, mids in zip(sync_exps, sync_model_ids):
        vlists = df_to_verdict_lists(df, n_agents=2)
        part = convert_synchronous(vlists, mids, verdict2num, missing_tokens=missing_tokens)
        all_parts.append(part)

    # --- round-robin (2-way & 3-way) ---
    for df, mids in zip(rr_exps, rr_model_ids):
        n_agents = len(mids)
        vlists = df_to_verdict_lists(df, n_agents=n_agents)
        part = convert_round_robin(vlists, mids, verdict2num, missing_tokens=missing_tokens)
        all_parts.append(part)

    data = concat_experiments(all_parts)

    # Quick sanity summary
    N = len(data["y"])
    K = data["same_prev_mat"].shape[1] if N else len(verdict2num)
    summary = {
        "rows": N,
        "K": K,
        "sync_rows": int(data["is_sync"].sum()),
        "rr_rows": int((~data["is_sync"]).sum()),
        "round>=2 rows": int((data["round_idx"] >= 2).sum()),
        "same_prev_nonzero (round>=2)": int((data["same_prev_mat"][data["round_idx"] >= 2].sum(axis=1) > 0).sum()),
        "within_nonzero (RR only)": int((data["E_within_mat"][~data["is_sync"]].sum(axis=1) > 0).sum()),
    }
    print(summary)

    return data


def _one_hot(k):
    v = np.zeros(K, dtype=np.float32); v[k] = 1.0; return v

def _is_missing(v, missing_tokens):
    return (v is None) or (isinstance(v, float) and np.isnan(v)) or (v in missing_tokens)


def convert_synchronous(verdict_lists, model_indices, verdict2num, missing_tokens=("", None)):
    n_agents = len(verdict_lists)
    n_dilemmas = len(verdict_lists[0])

    y, model_idx, dilemma_idx, round_idx, speaker_pos = [], [], [], [], []
    same_prev_rows, E_prev_rows, E_within_rows, is_sync = [], [], [], []

    last_label = {(m, d): None for d in range(n_dilemmas) for m in model_indices}
    self_cum   = {(m, d): np.zeros(K, dtype=np.float32) for d in range(n_dilemmas) for m in model_indices}
    total_prev = {d: np.zeros(K, dtype=np.float32) for d in range(n_dilemmas)}

    for d in range(n_dilemmas):
        n_rounds_d = len(verdict_lists[0][d])
        for r in range(n_rounds_d):
            round_sum = np.zeros(K, dtype=np.float32)
            present_agents = []
            for a, m in enumerate(model_indices):
                v_str = verdict_lists[a][d][r]
                if _is_missing(v_str, missing_tokens): 
                    continue
                v = verdict2num(v_str)

                y.append(v); model_idx.append(m); dilemma_idx.append(d)
                round_idx.append(r + 1); speaker_pos.append(0); is_sync.append(True)

                same_prev_rows.append(np.zeros(K, np.float32) if last_label[(m,d)] is None
                                      else _one_hot(last_label[(m,d)]))

                eprev = total_prev[d] - self_cum[(m, d)]
                eprev[eprev < 0] = 0.0
                E_prev_rows.append(eprev)

                E_within_rows.append(np.zeros(K, np.float32))

                round_sum += _one_hot(v)
                present_agents.append((m, v))

            total_prev[d] += round_sum
            for (m, v) in present_agents:
                self_cum[(m, d)] += _one_hot(v)
                last_label[(m, d)] = v

    return {
        "y": np.asarray(y, dtype=np.int64),
        "model_idx": np.asarray(model_idx, dtype=np.int64),
        "dilemma_idx": np.asarray(dilemma_idx, dtype=np.int64),
        "round_idx": np.asarray(round_idx, dtype=np.int32),
        "speaker_pos": np.asarray(speaker_pos, dtype=np.int32),
        "same_prev_mat": np.vstack(same_prev_rows).astype(np.float32) if same_prev_rows else np.zeros((0, K), np.float32),
        "E_prev_mat": np.vstack(E_prev_rows).astype(np.float32) if E_prev_rows else np.zeros((0, K), np.float32),
        "E_within_mat": np.vstack(E_within_rows).astype(np.float32) if E_within_rows else np.zeros((0, K), np.float32),
        "is_sync": np.ones(len(y), dtype=bool),
    }

def convert_round_robin(verdict_lists, model_indices, verdict2num, missing_tokens=("", None)):
    n_agents = len(verdict_lists)
    n_dilemmas = len(verdict_lists[0])

    y, model_idx, dilemma_idx, round_idx, speaker_pos = [], [], [], [], []
    same_prev_rows, E_prev_rows, E_within_rows, is_sync = [], [], [], []

    last_label = {(m, d): None for d in range(n_dilemmas) for m in model_indices}
    self_cum   = {(m, d): np.zeros(K, dtype=np.float32) for d in range(n_dilemmas) for m in model_indices}
    total_prev = {d: np.zeros(K, dtype=np.float32) for d in range(n_dilemmas)}

    for d in range(n_dilemmas):
        n_rounds_d = max(len(verdict_lists[a][d]) for a in range(n_agents))
        for r in range(n_rounds_d):
            within_counts = np.zeros(K, dtype=np.float32)
            round_sum = np.zeros(K, dtype=np.float32)
            present_in_round = []

            for pos, (a, m) in enumerate(zip(range(n_agents), model_indices), start=1):
                if r >= len(verdict_lists[a][d]): 
                    continue
                v_str = verdict_lists[a][d][r]
                if _is_missing(v_str, missing_tokens):
                    continue
                v = verdict2num(v_str)

                y.append(v); model_idx.append(m); dilemma_idx.append(d)
                round_idx.append(r + 1); speaker_pos.append(pos); is_sync.append(False)

                same_prev_rows.append(np.zeros(K, np.float32) if last_label[(m,d)] is None
                                      else _one_hot(last_label[(m,d)]))

                eprev = total_prev[d] - self_cum[(m, d)]
                eprev[eprev < 0] = 0.0
                E_prev_rows.append(eprev)

                E_within_rows.append(within_counts.copy())

                within_counts += _one_hot(v)
                round_sum += _one_hot(v)
                present_in_round.append((m, v))

            total_prev[d] += round_sum
            for (m, v) in present_in_round:
                self_cum[(m, d)] += _one_hot(v)
                last_label[(m, d)] = v

    return {
        "y": np.asarray(y, dtype=np.int64),
        "model_idx": np.asarray(model_idx, dtype=np.int64),
        "dilemma_idx": np.asarray(dilemma_idx, dtype=np.int64),
        "round_idx": np.asarray(round_idx, dtype=np.int32),
        "speaker_pos": np.asarray(speaker_pos, dtype=np.int32),
        "same_prev_mat": np.vstack(same_prev_rows).astype(np.float32) if same_prev_rows else np.zeros((0, K), np.float32),
        "E_prev_mat": np.vstack(E_prev_rows).astype(np.float32) if E_prev_rows else np.zeros((0, K), np.float32),
        "E_within_mat": np.vstack(E_within_rows).astype(np.float32) if E_within_rows else np.zeros((0, K), np.float32),
        "is_sync": np.zeros(len(y), dtype=bool),
    }


class DeliberationModel(nn.Module):
    def __init__(self, M, D, K=5):
        super().__init__()
        self.theta_raw = nn.Parameter(torch.zeros(M, K))
        self.phi_raw   = nn.Parameter(torch.zeros(D, K))
        self.alpha     = nn.Parameter(torch.zeros(M))
        self.gamma     = nn.Parameter(torch.tensor(0.0))

    def forward(self, mi, di, same_prev, E_prev, E_within):
        theta = self.theta_raw - self.theta_raw.mean(dim=1, keepdim=True)
        phi   = self.phi_raw   - self.phi_raw.mean(dim=1, keepdim=True)
        logits = theta[mi] + phi[di] + self.alpha[mi].unsqueeze(1)*same_prev + self.gamma*(E_prev+E_within)
        return torch.log_softmax(logits, dim=1)

def fit_map_blended(exp, lr=1e-2, epochs=50, batch_size=4096,
                    sigma_theta=1.0, sigma_phi=1.0, sigma_alpha=0.5, sigma_gamma=0.5,
                    device="cpu", seed=0):
    torch.manual_seed(seed)
    y = torch.as_tensor(exp["y"], dtype=torch.long, device=device)
    mi = torch.as_tensor(exp["model_idx"], dtype=torch.long, device=device)
    di = torch.as_tensor(exp["dilemma_idx"], dtype=torch.long, device=device)
    sp = torch.as_tensor(exp["same_prev_mat"], dtype=torch.float32, device=device)
    ep = torch.as_tensor(exp["E_prev_mat"], dtype=torch.float32, device=device)
    ew = torch.as_tensor(exp["E_within_mat"], dtype=torch.float32, device=device)

    N, K = sp.shape
    M = int(mi.max().item()) + 1
    D = int(di.max().item()) + 1

    model = DeliberationModel(M, D, K=K).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    dl = DataLoader(TensorDataset(mi, di, sp, ep, ew, y), batch_size=batch_size, shuffle=True)

    nll = nn.NLLLoss(reduction='mean')
    for _ in range(epochs):
        for mi_b, di_b, sp_b, ep_b, ew_b, y_b in dl:
            logp = model(mi_b, di_b, sp_b, ep_b, ew_b)
            loss = nll(logp, y_b)
            theta, phi, alpha, gamma = model.theta_raw, model.phi_raw, model.alpha, model.gamma
            prior = (theta.pow(2).sum()/(2*sigma_theta**2) +
                     phi.pow(2).sum()/(2*sigma_phi**2) +
                     alpha.pow(2).sum()/(2*sigma_alpha**2) +
                     gamma.pow(2)/(2*sigma_gamma**2))
            loss = loss + prior / N
            opt.zero_grad(); loss.backward(); opt.step()

    return model

class DeliberationModelSplit(nn.Module):
    def __init__(self, M, D, K=5):
        super().__init__()
        self.theta_raw = nn.Parameter(torch.zeros(M, K))   # model × label
        self.phi_raw   = nn.Parameter(torch.zeros(D, K))   # dilemma × label
        self.alpha     = nn.Parameter(torch.zeros(M))      # self-stickiness per model
        self.gamma_prev   = nn.Parameter(torch.tensor(0.0))
        self.gamma_within = nn.Parameter(torch.tensor(0.0))

    def forward(self, mi, di, sp, ep, ew):
        # row-center over labels for ID
        theta = self.theta_raw - self.theta_raw.mean(dim=1, keepdim=True)
        phi   = self.phi_raw   - self.phi_raw.mean(dim=1, keepdim=True)
        logits = (
            theta[mi] +
            phi[di] +
            self.alpha[mi].unsqueeze(1) * sp +
            self.gamma_prev * ep +
            self.gamma_within * ew
        )
        return torch.log_softmax(logits, dim=1)

def fit_map_split(exp, lr=1e-2, epochs=60, batch_size=8192,
                  sigma_theta=1.0, sigma_phi=1.0, sigma_alpha=0.5,
                  sigma_gamma_prev=0.5, sigma_gamma_within=0.5,
                  device=None, seed=0, tol=1e-4, verbose=False, early_stop_patience=10):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    y  = torch.as_tensor(exp["y"], dtype=torch.long, device=device)
    mi = torch.as_tensor(exp["model_idx"], dtype=torch.long, device=device)
    di = torch.as_tensor(exp["dilemma_idx"], dtype=torch.long, device=device)
    sp = torch.as_tensor(exp["same_prev_mat"], dtype=torch.float32, device=device)
    ep = torch.as_tensor(exp["E_prev_mat"], dtype=torch.float32, device=device)
    ew = torch.as_tensor(exp["E_within_mat"], dtype=torch.float32, device=device)

    N, K = sp.shape
    M = int(mi.max().item()) + 1
    D = int(di.max().item()) + 1

    model = DeliberationModelSplit(M, D, K=K).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.NLLLoss(reduction='mean')  # <-- don't overwrite this

    dl = DataLoader(TensorDataset(mi, di, sp, ep, ew, y), batch_size=batch_size, shuffle=True)

    best_obj = float("inf")
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for mi_b, di_b, sp_b, ep_b, ew_b, y_b in dl:
            logp = model(mi_b, di_b, sp_b, ep_b, ew_b)
            nll_val = criterion(logp, y_b)

            theta, phi, alpha = model.theta_raw, model.phi_raw, model.alpha
            gp, gw = model.gamma_prev, model.gamma_within
            prior = (
                theta.pow(2).sum()/(2*sigma_theta**2) +
                phi.pow(2).sum()/(2*sigma_phi**2) +
                alpha.pow(2).sum()/(2*sigma_alpha**2) +
                gp.pow(2)/(2*sigma_gamma_prev**2) +
                gw.pow(2)/(2*sigma_gamma_within**2)
            ) / N

            loss = nll_val + prior

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # monitor objective once per epoch
        with torch.no_grad():
            logp_all = model(mi, di, sp, ep, ew)
            nll_all = criterion(logp_all, y)
            theta, phi, alpha = model.theta_raw, model.phi_raw, model.alpha
            gp, gw = model.gamma_prev, model.gamma_within
            prior_all = (
                theta.pow(2).sum()/(2*sigma_theta**2) +
                phi.pow(2).sum()/(2*sigma_phi**2) +
                alpha.pow(2).sum()/(2*sigma_alpha**2) +
                gp.pow(2)/(2*sigma_gamma_prev**2) +
                gw.pow(2)/(2*sigma_gamma_within**2)
            ) / N
            obj = (nll_all + prior_all).item()

        if verbose:
            if epoch % 10 == 0:
                print(f"epoch {epoch+1}: obj={obj:.6f}, nll={nll_all.item():.6f}")

        # simple early stopping on the full objective
        if obj + tol < best_obj:
            best_obj = obj
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                if verbose:
                    print(f"Early stop at epoch {epoch+1}")
                break

    return model

def _subset_by_dilemmas(exp, d_ids):
    m = np.isin(exp["dilemma_idx"], d_ids)
    return {k: v[m] for k, v in exp.items()}

def _subset_by_dilemmas(exp, d_ids):
    mask = np.isin(exp["dilemma_idx"], d_ids)
    return {k: v[mask] for k, v in exp.items()}

def _one_boot(args):
    exp, dilemmas, seed, epochs, batch, lr, device = args
    rng = np.random.default_rng(seed)
    draw = rng.choice(dilemmas, size=len(dilemmas), replace=True)
    boot = _subset_by_dilemmas(exp, draw)
    m = fit_map_split(boot, device=device, epochs=epochs, batch_size=batch, lr=lr, seed=seed, verbose=False)
    return {
        "gamma_prev": float(m.gamma_prev.item()),
        "gamma_within": float(m.gamma_within.item()),
        "alpha": m.alpha.detach().cpu().numpy().copy(),
        "theta": m.theta_raw.detach().cpu().numpy().copy(),
        "phi": m.phi_raw.detach().cpu().numpy().copy()
    }

def bootstrap_split(exp, B=200, epochs=20, batch_size=4096, lr=2e-2, device="cpu", n_jobs=None, base_seed=0):
    dilemmas = np.unique(exp["dilemma_idx"])
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
    args = [(exp, dilemmas, base_seed + b, epochs, batch_size, lr, device) for b in range(B)]
    with mp.Pool(processes=n_jobs) as pool:
        outs = pool.map(_one_boot, args)

    # stack results
    gp = np.array([o["gamma_prev"] for o in outs])
    gw = np.array([o["gamma_within"] for o in outs])
    alpha = np.stack([o["alpha"] for o in outs])
    theta = np.stack([o["theta"] for o in outs])
    #phi   = np.stack([o["phi"]   for o in outs])

    # confidence intervals for scalars
    ci_gp = np.percentile(gp, [2.5, 50, 97.5])
    ci_gw = np.percentile(gw, [2.5, 50, 97.5])
    ci_alpha = np.percentile(alpha, [2.5, 50, 97.5], axis=0)

    return {
        "gamma_prev": gp,
        "gamma_within": gw,
        "alpha": alpha,
        "theta": theta,
        #"phi": phi,
        "ci_prev": ci_gp,
        "ci_within": ci_gw,
        "ci_alpha": ci_alpha
    }