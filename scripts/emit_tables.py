"""
Script to emit publication LaTeX tables for CoopGCN from empirical evaluation results:
- Table 1: Multi-Dataset Overall Recommendation Accuracy & Long-Tail Bias Mitigation
- Table 2: THE Central Make-or-Break Ablation (Shapley vs. Attention)
- Table 3: Adversarial Edge Noise Immunity Degradation Analysis
- Table 4: Complete 10-Row Component Ablation Study (G1, G2, G3, L_game)
"""

import os
import pandas as pd


def emit_table_1_overall_performance(df_overall, output_dir="tables"):
    """
    df_overall: Multi-index DataFrame (dataset, model) or single dataset DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)
    tab1_tex = r"""\begin{table*}[t]
\centering
\caption{Overall collaborative filtering performance across target benchmark datasets under global temporal splits (70\% train / 10\% validation / 20\% test). Best results in \textbf{bold}.}
\label{tab:overall_results}
\resizebox{\linewidth}{!}{
\begin{tabular}{lcccccc}
\toprule
\textbf{Model / Architecture} & \textbf{NDCG@20} & \textbf{Recall@20} & \textbf{TR@20 (Tail)} & \textbf{Coverage@20} & \textbf{Gini Index} & \textbf{NDCG Gain (\%)} \\
\midrule
"""
    for model_name, row in df_overall.iterrows():
        gain = row.get("NDCG Gain (%)", 0.0)
        tab1_tex += f"{model_name} & {row['NDCG@20']:.4f} & {row['Recall@20']:.4f} & {row['TR@20']:.4f} & {row['Coverage@20']:.4f} & {row['Gini']:.4f} & {gain:+.1f}\\% \\\\\n"

    tab1_tex += r"""\bottomrule
\end{tabular}}
\end{table*}
"""
    path1 = os.path.join(output_dir, "tab1_overall_performance.tex")
    with open(path1, "w") as f:
        f.write(tab1_tex)
    print(f"✅ Table 1 emitted to {path1}")


def emit_table_2_central_ablation(df_ablation, output_dir="tables"):
    os.makedirs(output_dir, exist_ok=True)
    tab2_tex = r"""\begin{table}[t]
\centering
\caption{THE Central Make-or-Break Ablation: Axiomatic Shapley vs. Heuristic Attention vs. Uniform Weighting across target datasets.}
\label{tab:central_ablation}
\begin{tabular}{lcccc}
\toprule
\textbf{Weighting / Attribution Method} & \textbf{NDCG@20} & \textbf{TR@20 (Tail)} & \textbf{Coverage@20} & \textbf{Gini Index} \\
\midrule
"""
    for model_name, row in df_ablation.iterrows():
        tab2_tex += f"{model_name} & {row['NDCG@20']:.4f} & {row['TR@20']:.4f} & {row['Coverage@20']:.4f} & {row['Gini']:.4f} \\\\\n"

    tab2_tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    path2 = os.path.join(output_dir, "tab2_central_ablation.tex")
    with open(path2, "w") as f:
        f.write(tab2_tex)
    print(f"✅ Table 2 emitted to {path2}")


def emit_table_3_noise_immunity(df_noise, output_dir="tables"):
    os.makedirs(output_dir, exist_ok=True)
    tab3_tex = r"""\begin{table}[t]
\centering
\caption{Adversarial edge noise immunity analysis under 0\%, 5\%, 10\%, and 20\% injected edge noise.}
\label{tab:noise_immunity}
\begin{tabular}{lcccc}
\toprule
\textbf{Noise Ratio} & \textbf{LightGCN} & \textbf{GAT-CF} & \textbf{DyHuCoG} & \textbf{CoopGCN (Ours)} \\
\midrule
"""
    for idx_label, row in df_noise.iterrows():
        tab3_tex += f"{idx_label} & {row.get('LightGCN', 0.0):.4f} & {row.get('GAT-CF', 0.0):.4f} & {row.get('DyHuCoG', 0.0):.4f} & {row.get('CoopGCN (Ours)', 0.0):.4f} \\\\\n"

    tab3_tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    path3 = os.path.join(output_dir, "tab3_noise_immunity.tex")
    with open(path3, "w") as f:
        f.write(tab3_tex)
    print(f"✅ Table 3 emitted to {path3}")


def emit_table_4_component_ablation(df_comp, output_dir="tables"):
    os.makedirs(output_dir, exist_ok=True)
    tab4_tex = r"""\begin{table}[t]
\centering
\caption{Complete Component Ablation Study across empirical evaluation runs.}
\label{tab:component_ablation}
\resizebox{\linewidth}{!}{
\begin{tabular}{lccccc}
\toprule
\textbf{Model / Architecture Configuration} & \textbf{NDCG@20} & \textbf{Recall@20} & \textbf{TR@20 (Tail)} & \textbf{Coverage@20} & \textbf{Gini Index} \\
\midrule
"""
    for model_name, row in df_comp.iterrows():
        tab4_tex += f"{model_name} & {row['NDCG@20']:.4f} & {row['Recall@20']:.4f} & {row['TR@20']:.4f} & {row['Coverage@20']:.4f} & {row['Gini']:.4f} \\\\\n"

    tab4_tex += r"""\bottomrule
\end{tabular}}
\end{table}
"""
    path4 = os.path.join(output_dir, "tab4_component_ablation.tex")
    with open(path4, "w") as f:
        f.write(tab4_tex)
    print(f"✅ Table 4 emitted to {path4}")


def emit_all_tables(df_overall, df_ablation, df_comp, df_noise, output_dir="tables"):
    emit_table_1_overall_performance(df_overall, output_dir=output_dir)
    emit_table_2_central_ablation(df_ablation, output_dir=output_dir)
    emit_table_3_noise_immunity(df_noise, output_dir=output_dir)
    emit_table_4_component_ablation(df_comp, output_dir=output_dir)
