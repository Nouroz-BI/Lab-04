
import csv
import statistics

from lab04_starter import train_recall, RecallNet


# Use the same three seeds for every experiment
seeds = [0, 1, 2]

results = []


# Function to run one experiment with three seeds
def run_experiment(name, kind, gate_bias=None, clip=1.0):

    accuracies = []

    for seed in seeds:

        # Train the model using the starter settings
        accuracy, history = train_recall(
            kind,
            seq_len=30,
            seed=seed,
            gate_bias=gate_bias,
            clip=clip
        )

        accuracies.append(accuracy)

        # Save the accuracy for each seed
        results.append({
            "Experiment": name,
            "Seed": seed,
            "Accuracy": accuracy
        })

        print(
            f"{name}, seed={seed}, "
            f"accuracy={accuracy:.3f}"
        )

    # Calculate the mean accuracy
    mean_accuracy = statistics.mean(accuracies)

    results.append({
        "Experiment": name,
        "Seed": "Mean",
        "Accuracy": mean_accuracy
    })

    print(f"{name} mean accuracy: {mean_accuracy:.3f}")
    print("-" * 50)


# Experiment 1: LSTM forget-gate bias sweep

for bias in [0.0, 1.0, 2.0, 3.0]:

    # Confirm the intended gate slice before training
    model = RecallNet("lstm", gate_bias=bias)

    hidden_size = model.rnn.hidden_size

    print(f"LSTM forget-gate bias requested: {bias:g}")
    print(
        "Effective forget-gate bias:",
        (
            model.rnn.bias_ih_l0[hidden_size:2 * hidden_size]
            + model.rnn.bias_hh_l0[hidden_size:2 * hidden_size]
        ).detach()[0].item()
    )

    run_experiment(
        name=f"LSTM_Bias_{bias:g}",
        kind="lstm",
        gate_bias=bias
    )

# Experiment 2: RNN clipping comparison

# Gradient clipping enabled
run_experiment(
    name="RNN_Clipping_On",
    kind="rnn",
    clip=1.0
)

# Gradient clipping disabled
run_experiment(
    name="RNN_Clipping_Off",
    kind="rnn",
    clip=None
)


# Save all results in a CSV file

with open("fixes_results.csv", "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["Experiment", "Seed", "Accuracy"]
    )

    writer.writeheader()
    writer.writerows(results)


print("All experiments completed.")
print("Results saved to fixes_results.csv")
