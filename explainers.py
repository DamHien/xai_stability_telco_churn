# các hàm tính lời giải thích cho dự án XAI stability
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

def make_background(X_train_p, n=30):
    #kernelShap can mot du lieu nen de khi thieu feature thif thay bang ig
    # shap.kmeans gom du lieu train thanh n diem dai dien
    return shap.kmeans(X_train_p.values, n)

def lime_attribution(model, X_train_p, x, seed, num_samples=1000):
    # x: mot khach, dang mang 45 so. seed: dieu khien phan ngau nhien cua lime
    columns = list(X_train_p.columns)
    explainer = LimeTabularExplainer(
        training_data = X_train_p.values, #du lieu tham chieu
        feature_names = columns,
        mode="classification",
        discretize_continuous = False, #giu nguyen gia tri so
        random_state =seed,
    )
    exp = explainer.explain_instance(
        x,
        # ham ime dung de hoi model: boc mang thanh bang co ten cot roi lay xac suat 2 lop
        lambda a: model.predict_proba(pd.DataFrame(a, columns=columns)),
        num_features = len(columns), # lay trong so cua ca 45 feature
        num_samples = num_samples, # so khach gia duoc tao ra
    )
    weights = dict(exp.as_map()[1]) # trong so cho lop 1 
    #tra ve mang 45 so theo dung thu tu cot; cot vang mat thi la 0
    return np.array([weights.get(k, 0.0) for k in range(len(columns))])

def kernelshap_attribution(model, background, columns, x, seed, nsamples=300):
    # kernelshap dung np.random toan cuc, dat seed ngay truoc khi chay de tai lap duoc
    np.random.seed(seed)
    # tra ve xac xuat roi di 
    f = lambda a: model.predict_proba(pd.DataFrame(a, columns=columns))[:, 1]
    explainer = shap.KernelExplainer(f, background)
    values = explainer.shap_values(x.reshape(1, -1), nsamples=nsamples, silent=True) # (1, -1) : doi mang 45 so thang bang 1 dong
    return np.asarray(values).ravel() # tra ve mang 1 chiu 45 so

def exact_shap_attribution(model, X_train_p, x):
    # Ban shap tinh chinh xac, khong co yeu to ngay nhien
    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, X_train_p.values)
    else:
        explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x.reshape(1, -1))
    #gom shap ve mang lop 1
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return values.ravel()