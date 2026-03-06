import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Setup paths
ROOT_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT_DIR / "data" / "processed"
FIGURES_DIR  = ROOT_DIR / "output" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def analyze_client_distribution():
    print("="*60)
    print("  CLIENT DISTRIBUTION ANALYSIS (IID vs NON-IID)")
    print("="*60)
    
    files = {
        "NY (IID)": "Hospital_NY_IID.csv",
        "TX (IID)": "Hospital_TX_IID.csv",
        "NY (Non-IID)": "Hospital_NY_NONIID.csv",
        "TX (Non-IID)": "Hospital_TX_NONIID.csv",
    }
    
    stats = []
    
    for label, fname in files.items():
        path = DATA_DIR / fname
        if not path.exists():
            print(f"File not found: {path} - please run Step 1 (03_generate_client_splits.ipynb) first.")
            continue
            
        df = pd.read_csv(path)
        
        prevalence = df["Diabetes_binary"].mean() * 100
        avg_age    = df["Age"].mean() if "Age" in df.columns else df["AgeCAT"].mean() if "AgeCAT" in df.columns else df["Age_Categories"].mean() if "Age_Categories" in df.columns else df.get("AgeCAT5", df.columns[df.columns.str.contains("Age", case=False)]).mean().iloc[0] if df.columns[df.columns.str.contains("Age", case=False)].any() else 0
        avg_bmi    = df["BMI"].mean()
        
        print(f"--- {label} ---")
        print(f"  Rows        : {len(df):,}")
        print(f"  Diabetes %  : {prevalence:.2f}%")
        print(f"  Avg Age CAT : {avg_age:.2f}")
        print(f"  Avg BMI     : {avg_bmi:.2f}\n")
        
        hospital = "NY" if "NY" in label else "TX"
        mode     = "IID" if "IID" in label and "Non" not in label else "Non-IID"
        
        stats.append({
            "Hospital": hospital,
            "Mode": mode,
            "Diabetes Prevalence (%)": prevalence
        })
        
    if not stats:
        return
        
    df_stats = pd.DataFrame(stats)
    
    # ---------------------------------------------------------
    # Generate Figure
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    sns.set_palette("muted")
    ax = sns.barplot(
        data=df_stats, 
        x="Mode", 
        y="Diabetes Prevalence (%)", 
        hue="Hospital"
    )
    
    plt.title("Client Data Heterogeneity: IID vs Non-IID Splits", pad=20, fontsize=14, fontweight='bold')
    plt.ylabel("Diabetes Prevalence (%)", fontsize=12)
    plt.xlabel("Data Distribution Mode", fontsize=12)
    plt.ylim(0, 30)
    
    # Add percentage labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', 
                        xytext=(0, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
    
    plt.legend(title="Client Hospital", fontsize=11)
    plt.tight_layout()
    
    out_path = FIGURES_DIR / "fig_client_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Figure saved to: {out_path}")

if __name__ == "__main__":
    analyze_client_distribution()
