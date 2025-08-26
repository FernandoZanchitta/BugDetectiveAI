import re
import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from typing import Optional, Literal

def wilcoxon_metric_test(
    df_a: pd.DataFrame,
    model_a: str,
    metric: str,
    df_b: Optional[pd.DataFrame] = None,
    model_b: Optional[str] = None,
    pair_on: str = "sample_uuid",
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    zero_method: Literal["wilcox", "pratt", "zsplit"] = "wilcox",
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: Optional[int] = 0,
) -> pd.Series:
    """
    Wilcoxon pareado para métricas por `sample_uuid`, com IC bootstrap do diff_median
    e contagens de wins/ties/losses.

    Formato das colunas esperadas: f"metric_{model}_{metric}"

    Exemplos
    --------
    # (1) Mesmo dataset, modelos diferentes
    # wilcoxon_metric_test(baseline, "gpt-4o", "codebleu", model_b="claude-3.5-sonnet", alternative="greater")

    # (2) Datasets diferentes, mesmo modelo
    # wilcoxon_metric_test(baseline, "gpt-4o", "codebleu", df_b=sys_apr, model_b="gpt-4o", alternative="two-sided")

    Parâmetros relevantes
    ---------------------
    alternative : "greater" testa se model_a > model_b.
    zero_method : "wilcox" descarta empates; "pratt" incorpora.
    n_boot, alpha, seed : controle do bootstrap do diff_median.
    """
    if model_b is None:
        raise ValueError("You must provide model_b for comparison.")

    col_a = f"metric_{model_a}_{metric}"
    col_b = f"metric_{model_b}_{metric}"

    def _validate(df, col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in provided DataFrame.")

    _validate(df_a, col_a)
    if df_b is None:
        df_b = df_a
    _validate(df_b, col_b)

    # Seleção, alinhamento e limpeza
    a = df_a[[pair_on, col_a]].rename(columns={col_a: "a"})
    b = df_b[[pair_on, col_b]].rename(columns={col_b: "b"})
    m = (
        a.merge(b, on=pair_on, how="inner")
         .replace([np.inf, -np.inf], np.nan)
         .dropna(subset=["a", "b"])
    )
    if m.empty:
        raise ValueError("No paired, non-null observations after alignment on sample_uuid.")

    x = m["a"].to_numpy()
    y = m["b"].to_numpy()
    diffs = x - y

    # Contagens de wins/ties/losses (após limpeza, antes do zero_method)
    wins = int(np.sum(diffs > 0))
    losses = int(np.sum(diffs < 0))
    ties = int(np.sum(diffs == 0))
    n_paired = int(len(diffs))
    n_nonzero = wins + losses
    if n_nonzero == 0:
        raise ValueError("All paired differences are zero after alignment; Wilcoxon is undefined.")

    # Teste de Wilcoxon (pareado)
    stat, pval = wilcoxon(x, y, alternative=alternative, zero_method=zero_method)

    # Efeitos/métricas-resumo
    median_a = float(np.median(x))
    median_b = float(np.median(y))
    mean_a = float(np.mean(x))
    mean_b = float(np.mean(y))
    diff_med = float(np.median(diffs))
    diff_mean = float(np.mean(diffs))
    cl = float(wins / n_nonzero)  # Common Language: P(a > b | não-empate)

    # IC bootstrap para diff_median (Hodges–Lehmann no caso pareado)
    if n_boot and n_boot > 0:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n_paired, size=(n_boot, n_paired))
        boot_medians = np.median(diffs[idx], axis=1)
        ci_low, ci_high = np.quantile(boot_medians, [alpha/2, 1 - alpha/2])
        ci_low = float(ci_low)
        ci_high = float(ci_high)
    else:
        ci_low = float("nan")
        ci_high = float("nan")

    # Saída consolidada
    res = pd.Series({
        "model_a": model_a,
        "model_b": model_b,
        "metric": metric,
        "col_a": col_a,
        "col_b": col_b,
        "n_paired": n_paired,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "wins_pct": wins / n_paired,
        "ties_pct": ties / n_paired,
        "losses_pct": losses / n_paired,
        "n_nonzero": n_nonzero,
        "statistic": float(stat),
        "pvalue": float(pval),
        "alternative": alternative,
        "median_a": median_a,
        "median_b": median_b,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "diff_median": diff_med,
        "diff_median_ci_low": ci_low,
        "diff_median_ci_high": ci_high,
        "diff_mean": diff_mean,
        "cl_effect": cl,               # ~0.5 neutro; >0.5 favorece model_a
        "n_boot": int(n_boot),
        "alpha": float(alpha),
        "seed": seed if seed is not None else np.nan,
        "zero_method": zero_method,
    })
    return res

# ---------- 1) Identificação de experimento + Tidy ----------
def _infer_experiment(df: pd.DataFrame, fallback: str) -> str:
    """
    Usa df['prompt'] para identificar o experimento, senão cai no fallback fornecido.
    Esperado: 'baseline', 'system_apr', 'style_based'.
    """
    if "prompt" in df.columns:
        vals = pd.unique(df["prompt"].astype(str))
        if len(vals) == 1:
            return vals[0]
    return fallback
def parse_metric_col(col: str) -> tuple[str, str]:
    if not col.startswith("metric_"):
        raise ValueError(f"Coluna não é de métrica: {col}")
    inner = col[len("metric_"):]              # ex: "gpt-4o_ast_score"
    model, metric = inner.split("_", 1)       # -> "gpt-4o", "ast_score"
    # Sanidade: modelos não devem conter "_"
    if "_" in model:
        raise ValueError(f"Parsing incorreto: model='{model}' contém '_' para coluna {col}")
    return model, metric

def tidy_metrics(df: pd.DataFrame, dataset_label: str, id_cols=("sample_uuid",)) -> pd.DataFrame:
    experiment = _infer_experiment(df, dataset_label)
    metric_cols = [c for c in df.columns if c.startswith("metric_")]

    long = []
    base = df[list(id_cols)].copy()
    base["experiment"] = experiment

    for c in metric_cols:
        model, metric = parse_metric_col(c)
        tmp = base.copy()
        tmp["model"] = model
        tmp["metric"] = metric
        tmp["value"] = df[c].values
        long.append(tmp)

    return pd.concat(long, ignore_index=True)

def _parse_metric_columns(df: pd.DataFrame):
    cols = [c for c in df.columns if c.startswith("metric_")]
    parsed = []
    for c in cols:
        model, metric = parse_metric_col(c)
        parsed.append((c, model, metric))
    return parsed

def available_models(df: pd.DataFrame) -> set:
    return {m for _, m, _ in _parse_metric_columns(df)}

def metrics_for_model(df: pd.DataFrame, model: str) -> set:
    return {metric for _, m, metric in _parse_metric_columns(df) if m == model}

# ---------- 3) Ajuste de p-values (Holm) por família ----------
def _holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """
    Holm-Bonferroni, retorna p-ajustados no mesmo ordenamento de entrada.
    """
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty_like(pvals, dtype=float)
    prev = 0.0
    for k, idx in enumerate(order):
        rank = k + 1  # 1..m na ordem crescente
        val = (m - k) * pvals[idx]
        val = min(1.0, val)
        prev = max(prev, val)
        adj[idx] = prev
    return adj

def adjust_pvalues_by_family(df: pd.DataFrame, group_cols=("scope", "metric")) -> pd.DataFrame:
    out = df.copy()
    out["p_adj"] = np.nan
    out["reject_05"] = False
    for _, gidx in out.groupby(list(group_cols)).groups.items():
        idx = list(gidx)
        padj = _holm_adjust(out.loc[idx, "pvalue"].to_numpy())
        out.loc[idx, "p_adj"] = padj
        out.loc[idx, "reject_05"] = padj <= 0.05
    return out

# ---------- 4) Runner para TODAS as combinações ----------
def run_all_wilcoxon(
    datasets: dict,  # ex: {"baseline": baseline, "system_apr": sys_apr, "style_based": style}
    *,
    alternative_within: str = "two-sided",
    alternative_across: str = "two-sided",
    zero_method: str = "wilcox",
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Roda:
      (A) Dentro de cada dataset: TODOS pares de modelos para TODAS as métricas em comum.
      (B) Entre datasets: para cada par de datasets, mesmo modelo e métricas em comum.

    Retorna DataFrame consolidado com p-ajustado (Holm) por família (scope, metric).
    """
    rows = []

    # (A) Dentro do mesmo dataset
    for dname, df in datasets.items():
        mods = sorted(list(available_models(df)))
        for model_a, model_b in itertools.combinations(mods, 2):
            mets = sorted(list(metrics_for_model(df, model_a).intersection(metrics_for_model(df, model_b))))
            for metric in mets:
                res = wilcoxon_metric_test(
                    df, model_a, metric,
                    model_b=model_b,
                    alternative=alternative_within,
                    zero_method=zero_method,
                    n_boot=n_boot, alpha=alpha, seed=seed
                )
                r = res.to_dict()
                r["scope"] = "within_dataset_models"
                r["dataset"] = dname
                r["dataset_a"] = dname
                r["dataset_b"] = dname
                rows.append(r)

    # (B) Entre datasets (mesmo modelo)
    dnames = list(datasets.keys())
    for dA, dB in itertools.combinations(dnames, 2):
        dfA, dfB = datasets[dA], datasets[dB]
        common_models = sorted(list(available_models(dfA).intersection(available_models(dfB))))
        for model in common_models:
            mets = sorted(list(metrics_for_model(dfA, model).intersection(metrics_for_model(dfB, model))))
            for metric in mets:
                res = wilcoxon_metric_test(
                    dfA, model, metric,
                    df_b=dfB, model_b=model,
                    alternative=alternative_across,
                    zero_method=zero_method,
                    n_boot=n_boot, alpha=alpha, seed=seed
                )
                r = res.to_dict()
                r["scope"] = "across_datasets_model"
                r["dataset"] = None
                r["dataset_a"] = dA
                r["dataset_b"] = dB
                rows.append(r)

    out = pd.DataFrame(rows)

    # Ajuste Holm por família (scope, metric)
    if not out.empty:
        out = adjust_pvalues_by_family(out, group_cols=("scope", "metric"))
    return out

# ---------- 5) (Opcional) construir o tidy global para inspeção/plots ----------
def build_global_tidy(datasets: dict) -> pd.DataFrame:
    tlist = [tidy_metrics(df, name) for name, df in datasets.items()]
    return pd.concat(tlist, ignore_index=True)
