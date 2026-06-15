import pandas as pd

KEY_COLS = ['bundle', 'case_id', 'difficulty_level', 'case_type', 'final_result', 'is_no_liability']

df1 = pd.read_csv('law_model_comparison1.csv')
df2 = pd.read_csv('law_model_comparison2.csv')
df3 = pd.read_csv('law_model_comparison3.csv')
df4 = pd.read_csv('law_model_comparison4.csv')
df5 = pd.read_csv('law_model_comparison5.csv')
df = df1.merge(df2, on=KEY_COLS, how='outer').merge(df3, on=KEY_COLS, how='outer').merge(df4, on=KEY_COLS, how='outer').merge(df5, on=KEY_COLS, how='outer')

MODELS = [
    ("qwen-30B-instruct", "instruct"),
    ("qwen-30B-thinking", "thinking"),
    ("deepseek-v3.2",     "deepseek-v3.2"),
    # ("glm5",            "glm5"),
    ("qwen3.6-flash",     "qwen3.6-flash"),
    ("gemini-3.1-flash",  "gemini-3.1-flash"),
    ("gpt-5.4-mini",      "gpt-5.4-mini"),
    ("deepseek-r1",       "deepseek-r1"),
    ("deepseek-v4-flash", "deepseek-v4-flash"),
    ("minimax-m2.5",      "minimax-m2.5"),
    ("minimax-m2.7",      "minimax-m2.7"),
    ("deepseek-v4-pro",   "deepseek-v4-pro")
]

METRICS = [
    "answer_correct",
    "rouge_l",
    "bertscore_f1",
    "roscoe_sa_prec",
    "roscoe_sa_rec",
    "roscoe_sa_f1",
    "rule_f1",
    "rule_precision",
    "rule_recall",
    "rule_exact_match",
    "rule_extra",
    "rule_missing",
    "rule_wrong",
    "srmr_precision",
    "srmr_recall",
    "srmr_f1",
    "reasoning_score",
    "key_rule_f1_orig",
]


rows = []
for name, prefix in MODELS:
    row = {"model": name}
    for metric in METRICS:
        col = f'{prefix}_{metric}'
        row[metric] = df[col].mean() if col in df.columns else float("nan")
    rows.append(row)

result = pd.DataFrame(rows).set_index("model")
print(result.to_string(float_format="{:.4f}".format))
print()

output_path = "avg_scores_law_by_model.csv"
result.to_csv(output_path, float_format="%.4f")
print(f"Saved to {output_path}")
