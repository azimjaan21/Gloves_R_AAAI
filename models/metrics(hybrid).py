import pandas as pd

# Load the CSV file with evaluation results
csv_path = r"C:\Users\dalab\Desktop\azimjaan21\Gloves_R_AAAI\hybrid_results\hybrid_evaluation_results.csv"
df = pd.read_csv(csv_path)

# Get counts of True Positives, False Positives, and False Negatives for Gloves and No Gloves
TP_gloves = df["True Positives (Gloves)"].sum()
FP_gloves = df["False Positives (Gloves)"].sum()
FN_gloves = df["False Negatives (Gloves)"].sum()

TP_no_gloves = df["True Positives (No Gloves)"].sum()
FP_no_gloves = df["False Positives (No Gloves)"].sum()
FN_no_gloves = df["False Negatives (No Gloves)"].sum()

# Compute Precision, Recall, and F1 Score for Gloves
precision_gloves = TP_gloves / (TP_gloves + FP_gloves + 1e-6)  # Avoid division by zero
recall_gloves = TP_gloves / (TP_gloves + FN_gloves + 1e-6)
f1_gloves = 2 * (precision_gloves * recall_gloves) / (precision_gloves + recall_gloves + 1e-6)

# Compute Precision, Recall, and F1 Score for No Gloves
precision_no_gloves = TP_no_gloves / (TP_no_gloves + FP_no_gloves + 1e-6)
recall_no_gloves = TP_no_gloves / (TP_no_gloves + FN_no_gloves + 1e-6)
f1_no_gloves = 2 * (precision_no_gloves * recall_no_gloves) / (precision_no_gloves + recall_no_gloves + 1e-6)

# Compute Average Precision (AP) for Gloves and No Gloves (mAP50)
mAP50_gloves = TP_gloves / (TP_gloves + FP_gloves + FN_gloves + 1e-6)
mAP50_no_gloves = TP_no_gloves / (TP_no_gloves + FP_no_gloves + FN_no_gloves + 1e-6)
mAP50 = (mAP50_gloves + mAP50_no_gloves) / 2  # Mean AP across both classes

# Print Results
print("\n📌 **Final Evaluation Metrics**")
print(f"🟢 Gloves - Precision: {precision_gloves:.4f}, Recall: {recall_gloves:.4f}, F1 Score: {f1_gloves:.4f}, mAP50: {mAP50_gloves:.4f}")
print(f"🔴 No Gloves - Precision: {precision_no_gloves:.4f}, Recall: {recall_no_gloves:.4f}, F1 Score: {f1_no_gloves:.4f}, mAP50: {mAP50_no_gloves:.4f}")
print(f"⭐ Overall mAP50: {mAP50:.4f}")

# Save results to a new CSV file
output_csv_path = r"C:\Users\dalab\Desktop\azimjaan21\Gloves_R_AAAI\hybrid_results\final_metrics.csv"
final_metrics_df = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1 Score", "mAP50"],
    "Gloves": [precision_gloves, recall_gloves, f1_gloves, mAP50_gloves],
    "No Gloves": [precision_no_gloves, recall_no_gloves, f1_no_gloves, mAP50_no_gloves],
    "Overall": ["-", "-", "-", mAP50]
})
final_metrics_df.to_csv(output_csv_path, index=False)

print(f"\n✅ Evaluation results saved in: {output_csv_path}")
