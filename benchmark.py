
import csv
import statistics

from lab04_starter import train_recall


# Five configurations required by the lab
configs = [
    ("RNN", "rnn", None),
    ("LSTM_Default", "lstm", None),
    ("LSTM_ForgetBias3", "lstm", 3.0),
    ("GRU_Default", "gru", None),
    ("GRU_UpdateBias3", "gru", 3.0),
]

# Sequence lengths and random seeds
sequence_lengths = [10, 30, 60]
seeds = [0, 1, 2]

results = []

for name, kind, gate_bias in configs:
    for seq_len in sequence_lengths:

        accuracies = []

        for seed in seeds:
            print(
                f"Running {name}, length={seq_len}, "
                f"seed={seed}"
            )

            accuracy, history = train_recall(
                kind,
                seq_len=seq_len,
                seed=seed,
                gate_bias=gate_bias
            )

            accuracies.append(accuracy)

            results.append({
                "Configuration": name,
                "Sequence_Length": seq_len,
                "Seed": seed,
                "Accuracy": accuracy
            })

            print(f"Validation accuracy: {accuracy:.3f}")

        mean_accuracy = statistics.mean(accuracies)

        results.append({
            "Configuration": name,
            "Sequence_Length": seq_len,
            "Seed": "Mean",
            "Accuracy": mean_accuracy
        })

        print(
            f"{name}, length={seq_len}, "
            f"mean accuracy={mean_accuracy:.3f}"
        )
        print("-" * 50)


# Save every seed and the mean
with open("benchmark_results.csv", "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Configuration",
            "Sequence_Length",
            "Seed",
            "Accuracy"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

print("Benchmark completed.")
print("Results saved to benchmark_results.csv")
