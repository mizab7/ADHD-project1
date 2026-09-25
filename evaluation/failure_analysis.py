import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_prediction_failures(manifest_df, y_true, y_pred, y_probs, output_dir="./results"):
    """
    Performs failure analysis breakdown across imaging sites, demographics, and scan duration.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_eval = manifest_df.copy().reset_index(drop=True)
    df_eval['y_true'] = y_true
    df_eval['y_pred'] = y_pred
    df_eval['adhd_prob'] = y_probs[:, 1] if y_probs.ndim == 2 else y_probs
    
    # Classify error type
    # 0: Correct Control, 1: Correct ADHD, 2: False Positive (Control -> ADHD error), 3: False Negative (ADHD -> Control error)
    def classify_error(row):
        if row['y_true'] == row['y_pred']:
            return "Correct"
        elif row['y_true'] == 0 and row['y_pred'] == 1:
            return "False Positive (Control -> ADHD Error)"
        elif row['y_true'] == 1 and row['y_pred'] == 0:
            return "False Negative (ADHD -> Control Error)"
        return "Unknown"
        
    df_eval['error_type'] = df_eval.apply(classify_error, axis=1)
    
    print("\n==========================================")
    print(" FAILURE ANALYSIS BREAKDOWN BY SITE")
    print("==========================================")
    site_err = pd.crosstab(df_eval['site'], df_eval['error_type'], margins=True)
    print(site_err)
    
    print("\n==========================================")
    print(" DEMOGRAPHIC & SCAN CHARACTERISTICS BY ERROR TYPE")
    print("==========================================")
    demo_err = df_eval.groupby('error_type')[['age', 'timepoints', 'adhd_prob']].mean()
    print(demo_err)
    
    out_csv = os.path.join(output_dir, "failure_analysis_report.csv")
    df_eval.to_csv(out_csv, index=False)
    print(f"\nDetailed failure analysis report saved to: {out_csv}\n")
    return df_eval
