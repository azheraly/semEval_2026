
# 2. Precision / Recall / F1 Bar Plot
report = classification_report(
    y_test,
    pred,
    target_names=["Human", "Machine"],
    output_dict=True
)

df_report = pd.DataFrame(report).transpose()
metrics = df_report.loc[["Human", "Machine"], ["precision", "recall", "f1-score"]]

plt.figure(figsize=(8,5))
metrics.plot(kind="bar")
plt.ylabel("Score")
plt.ylim(0,1)
plt.title("Classification Metrics – ML Model")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("plots/metrics_ml.pdf")
plt.show()



# 3. Error Type Distribution (TP / TN / FP / FN)


tp = np.sum((y_test == 1) & (pred == 1))
tn = np.sum((y_test == 0) & (pred == 0))
fp = np.sum((y_test == 0) & (pred == 1))
fn = np.sum((y_test == 1) & (pred == 0))

error_df = pd.DataFrame({
    "Error Type": ["TP", "TN", "FP (Human→Machine)", "FN (Machine→Human)"],
    "Count": [tp, tn, fp, fn]
})

plt.figure(figsize=(7,5))
sns.barplot(x="Error Type", y="Count", data=error_df)
plt.title("Error Type Distribution – ML Model")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("plots/error_types_ml.pdf")
plt.show()


# 4. Prediction Confidence Analysis
🔹 Logistic Regression


scores = lr_model.predict_proba(X_test_tfidf)[:,1]

# scores = svm_model.decision_function(X_test_tfidf)



correct_mask = (pred == y_test)

plt.figure(figsize=(7,5))
sns.kdeplot(scores[correct_mask], label="Correct", fill=True)
sns.kdeplot(scores[~correct_mask], label="Incorrect", fill=True)
plt.xlabel("Prediction Confidence")
plt.title("Confidence Distribution – ML Model")
plt.legend()
plt.tight_layout()
plt.savefig("plots/confidence_ml.pdf")
plt.show()



# 5. Code Length–Wise Error Analysis


test_df = pd.DataFrame({
    "text": X_test_text,
    "true": y_test,
    "pred": pred
})

test_df["code_length"] = test_df["text"].apply(lambda x: len(x.split()))

bins = [0, 50, 150, 1000]
labels = ["Short", "Medium", "Long"]
test_df["length_bin"] = pd.cut(test_df["code_length"], bins=bins, labels=labels)

error_rate = test_df.groupby("length_bin").apply(
    lambda x: (x["true"] != x["pred"]).mean()
)

plt.figure(figsize=(7,5))
error_rate.plot(kind="bar")
plt.ylabel("Error Rate")
plt.title("Error Rate vs Code Length – ML Model")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plots/error_vs_length_ml.pdf")
plt.show()



# 6. TF-IDF Feature Bias Analysis (Logistic Regression)



feature_names = tfidf.get_feature_names_out()
coefficients = lr_model.coef_[0]

coef_df = pd.DataFrame({
    "token": feature_names,
    "weight": coefficients
})

top_machine = coef_df.sort_values("weight", ascending=False).head(20)
top_human = coef_df.sort_values("weight").head(20)

print("\nTop Machine-Indicative Tokens:\n")
print(top_machine)

print("\nTop Human-Indicative Tokens:\n")
print(top_human)




# 📌 7. Misclassified Sample Extraction (Manual Inspection)

misclassified = test_df[test_df["true"] != test_df["pred"]]

fp_samples = misclassified[(misclassified["true"] == 0) & (misclassified["pred"] == 1)]
fn_samples = misclassified[(misclassified["true"] == 1) & (misclassified["pred"] == 0)]

fp_samples.to_csv("analysis/false_positives_ml.csv", index=False)
fn_samples.to_csv("analysis/false_negatives_ml.csv", index=False)


# 8. Final ML Performance Summary (Table)

summary = {
    "Accuracy": accuracy_score(y_test, pred),
    "Macro F1": f1_score(y_test, pred, average="macro"),
    "FP Rate": fp / (fp + tn),
    "FN Rate": fn / (fn + tp)
}

summary_df = pd.DataFrame(summary, index=["ML Model"])
print(summary_df)
