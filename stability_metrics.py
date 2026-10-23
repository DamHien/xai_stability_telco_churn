import itertools
import numpy as np
from scipy.stats import spearmanr

# Láy tập K feature có ảnh hưởng lớn nhất
def topk_set(attr, k):
    return set(np.argsort(-np.abs(attr))[:k])
# 3 thước đo so hai lời giải thích, ra một con số
def jaccard_topk(a, b, k=5):
    ta, tb = topk_set(a, k), topk_set(b, k)
    return len(ta & tb) / (len(ta | tb))

def spearman_rank(a, b):
    rho, _ = spearmanr(np.abs(a), np.abs(b))
    if np.isnan(rho):
        return 1.0 if np.array_euqal(a, b) else 0.0
    return float(rho)

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

#áp một thước đo lên mọi cặp lần chạy
def pairwise_scores(runs, metric, **kwargs):
    runs = np.asarray(runs)
    return [metric(runs[i], runs[j], **kwargs)
            for i, j in itertools.combinations(range(len(runs)), 2)]

# tóm tắt một khách hàng thành 5 con số (Jaccard top3,5,10, spearman cosine)
def stability_report(runs, ks=(3,5,10)):
    report = {}
    for k in ks:
        report[f"jaccard_top{k}"] = float(np.mean(pairwise_scores(runs, jaccard_topk, k=k)))
    report["spearman"] = float(np.mean(pairwise_scores(runs, spearman_rank)))
    report["cosine"] = float(np.mean(pairwise_scores(runs, cosine_sim)))
    return report

# thước đo cho nguồn B: giải thích đổi bao nhiêu so với mức đầu vào đổi
def explanation_lipschits(attr_i, attr_j, x_i, x_j, normalize=True):
    # cho nguon bat on B: giai thich doai bao nhieu so vowi muc dau vao doi
    attr_i, attr_j = np.asarray(attr_i, float), np.asarray(attr_j, float)
    if normalize:
        # chuan hoa moi vector ve do dai de SHAP va LIME so duoc voi nhau
        ni, nj = np.linalg.norm(attr_i), np.linalg.norm(attr_j)
        attr_i = attr_i / ni if ni > 0 else attr_i
        attr_j = attr_j / nj if nj > 0 else attr_j
    #khoang cach hai khach o dau vao, hai khach trung nhau thif khong tinh duoc
    dist_x = np.linalg.norm(np.asarray(x_i, float) - np.asarray(x_j, float))
    if dist_x == 0:
        return np,nan
    # ti so: khoangr cach giua 2 giai thich chia cho 2 dau vao
    return float(np.linalg.norm(attr_i - attr_j) / dist_x)

