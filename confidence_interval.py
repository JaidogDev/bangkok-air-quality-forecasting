import numpy as np
import pandas as pd
import json


VANILLA_PRED_PATH = "outputs/test_predictions.csv"
IMPROVED_PRED_PATH = "outputs/improved_lstm_dense_tanh/test_predictions.csv"

N_BOOTSTRAPS = 10000
SEED = 42


def bootstrap_ci_mae_rmse(y_true, y_pred, n_bootstraps=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)

    mae_scores = []
    rmse_scores = []

    for _ in range(n_bootstraps):
        indices = rng.choice(n, size=n, replace=True)

        sample_true = y_true[indices]
        sample_pred = y_pred[indices]

        errors = sample_pred - sample_true

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))

        mae_scores.append(mae)
        rmse_scores.append(rmse)

    mae_scores = np.array(mae_scores)
    rmse_scores = np.array(rmse_scores)

    return {
        "mae_mean": float(np.mean(mae_scores)),
        "mae_ci_lower": float(np.percentile(mae_scores, 2.5)),
        "mae_ci_upper": float(np.percentile(mae_scores, 97.5)),
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_ci_lower": float(np.percentile(rmse_scores, 2.5)),
        "rmse_ci_upper": float(np.percentile(rmse_scores, 97.5)),
    }


def bootstrap_ci_improvement(vanilla_df, improved_df, n_bootstraps=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(vanilla_df)

    if len(improved_df) != n:
        raise ValueError("Vanilla and improved prediction files must have the same number of rows.")

    vanilla_true = vanilla_df["actual_pm25"].values
    vanilla_pred = vanilla_df["lstm_predicted_pm25"].values
    improved_true = improved_df["actual_pm25"].values
    improved_pred = improved_df["lstm_predicted_pm25"].values

    if not np.allclose(vanilla_true, improved_true):
        raise ValueError("Actual PM2.5 values do not match between vanilla and improved files.")

    mae_improvements = []
    rmse_improvements = []

    for _ in range(n_bootstraps):
        indices = rng.choice(n, size=n, replace=True)

        v_true = vanilla_true[indices]
        v_pred = vanilla_pred[indices]
        i_true = improved_true[indices]
        i_pred = improved_pred[indices]

        vanilla_errors = v_pred - v_true
        improved_errors = i_pred - i_true

        vanilla_mae = np.mean(np.abs(vanilla_errors))
        improved_mae = np.mean(np.abs(improved_errors))

        vanilla_rmse = np.sqrt(np.mean(vanilla_errors ** 2))
        improved_rmse = np.sqrt(np.mean(improved_errors ** 2))

        mae_improvement = (vanilla_mae - improved_mae) / vanilla_mae * 100
        rmse_improvement = (vanilla_rmse - improved_rmse) / vanilla_rmse * 100

        mae_improvements.append(mae_improvement)
        rmse_improvements.append(rmse_improvement)

    mae_improvements = np.array(mae_improvements)
    rmse_improvements = np.array(rmse_improvements)

    return {
        "mae_improvement_mean_percent": float(np.mean(mae_improvements)),
        "mae_improvement_ci_lower_percent": float(np.percentile(mae_improvements, 2.5)),
        "mae_improvement_ci_upper_percent": float(np.percentile(mae_improvements, 97.5)),
        "rmse_improvement_mean_percent": float(np.mean(rmse_improvements)),
        "rmse_improvement_ci_lower_percent": float(np.percentile(rmse_improvements, 2.5)),
        "rmse_improvement_ci_upper_percent": float(np.percentile(rmse_improvements, 97.5)),
    }


def print_result(name, result):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"MAE  mean: {result['mae_mean']:.4f}")
    print(f"MAE  95% CI: [{result['mae_ci_lower']:.4f}, {result['mae_ci_upper']:.4f}]")
    print(f"RMSE mean: {result['rmse_mean']:.4f}")
    print(f"RMSE 95% CI: [{result['rmse_ci_lower']:.4f}, {result['rmse_ci_upper']:.4f}]")


def main():
    vanilla_df = pd.read_csv(VANILLA_PRED_PATH)
    improved_df = pd.read_csv(IMPROVED_PRED_PATH)

    vanilla_result = bootstrap_ci_mae_rmse(
        y_true=vanilla_df["actual_pm25"].values,
        y_pred=vanilla_df["lstm_predicted_pm25"].values,
        n_bootstraps=N_BOOTSTRAPS,
        seed=SEED
    )

    improved_result = bootstrap_ci_mae_rmse(
        y_true=improved_df["actual_pm25"].values,
        y_pred=improved_df["lstm_predicted_pm25"].values,
        n_bootstraps=N_BOOTSTRAPS,
        seed=SEED
    )

    improvement_result = bootstrap_ci_improvement(
        vanilla_df=vanilla_df,
        improved_df=improved_df,
        n_bootstraps=N_BOOTSTRAPS,
        seed=SEED
    )

    print_result("Vanilla LSTM", vanilla_result)
    print_result("Improved LSTM", improved_result)

    print("\nImprovement: Improved vs Vanilla")
    print("--------------------------------")
    print(f"MAE improvement mean: {improvement_result['mae_improvement_mean_percent']:.2f}%")
    print(
        "MAE improvement 95% CI: "
        f"[{improvement_result['mae_improvement_ci_lower_percent']:.2f}%, "
        f"{improvement_result['mae_improvement_ci_upper_percent']:.2f}%]"
    )

    print(f"RMSE improvement mean: {improvement_result['rmse_improvement_mean_percent']:.2f}%")
    print(
        "RMSE improvement 95% CI: "
        f"[{improvement_result['rmse_improvement_ci_lower_percent']:.2f}%, "
        f"{improvement_result['rmse_improvement_ci_upper_percent']:.2f}%]"
    )

    # =========================
    # Save confidence interval results to root project
    # =========================
    results = {
        "vanilla_lstm": vanilla_result,
        "improved_lstm": improved_result,
        "improvement_percent": improvement_result,
        "n_bootstraps": N_BOOTSTRAPS,
        "seed": SEED,
        "vanilla_prediction_path": VANILLA_PRED_PATH,
        "improved_prediction_path": IMPROVED_PRED_PATH
    }

    with open("confidence_interval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    with open("confidence_interval_results.txt", "w", encoding="utf-8") as f:
        f.write("Vanilla LSTM\n")
        f.write("------------\n")
        f.write(f"MAE  mean: {vanilla_result['mae_mean']:.4f}\n")
        f.write(f"MAE  95% CI: [{vanilla_result['mae_ci_lower']:.4f}, {vanilla_result['mae_ci_upper']:.4f}]\n")
        f.write(f"RMSE mean: {vanilla_result['rmse_mean']:.4f}\n")
        f.write(f"RMSE 95% CI: [{vanilla_result['rmse_ci_lower']:.4f}, {vanilla_result['rmse_ci_upper']:.4f}]\n\n")

        f.write("Improved LSTM\n")
        f.write("-------------\n")
        f.write(f"MAE  mean: {improved_result['mae_mean']:.4f}\n")
        f.write(f"MAE  95% CI: [{improved_result['mae_ci_lower']:.4f}, {improved_result['mae_ci_upper']:.4f}]\n")
        f.write(f"RMSE mean: {improved_result['rmse_mean']:.4f}\n")
        f.write(f"RMSE 95% CI: [{improved_result['rmse_ci_lower']:.4f}, {improved_result['rmse_ci_upper']:.4f}]\n\n")

        f.write("Improvement: Improved vs Vanilla\n")
        f.write("--------------------------------\n")
        f.write(f"MAE improvement mean: {improvement_result['mae_improvement_mean_percent']:.2f}%\n")
        f.write(
            "MAE improvement 95% CI: "
            f"[{improvement_result['mae_improvement_ci_lower_percent']:.2f}%, "
            f"{improvement_result['mae_improvement_ci_upper_percent']:.2f}%]\n"
        )
        f.write(f"RMSE improvement mean: {improvement_result['rmse_improvement_mean_percent']:.2f}%\n")
        f.write(
            "RMSE improvement 95% CI: "
            f"[{improvement_result['rmse_improvement_ci_lower_percent']:.2f}%, "
            f"{improvement_result['rmse_improvement_ci_upper_percent']:.2f}%]\n"
        )

    print("\nSaved confidence interval results to:")
    print("- confidence_interval_results.json")
    print("- confidence_interval_results.txt")


if __name__ == "__main__":
    main()