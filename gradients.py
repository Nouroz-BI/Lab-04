
import csv
import os
import statistics

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from lab04_starter import RecallNet, make_recall_batch


# Settings required by the lab
SEQ_LEN = 60
HIDDEN_SIZE = 32
BATCH_SIZE = 16
N_INPUTS = 6
SEEDS = [0, 1, 2, 3, 4]

CONFIGS = [
    ("RNN", "rnn", None),
    ("LSTM_Default", "lstm", None),
    ("LSTM_ForgetBias3", "lstm", 3.0),
]

all_results = {}

os.makedirs("figures", exist_ok=True)

for name, kind, gate_bias in CONFIGS:
    seed_gradients = []

    for seed in SEEDS:
        torch.manual_seed(seed)

        # Create a model with the required hidden size
        model = RecallNet(
            kind,
            hidden=HIDDEN_SIZE,
            gate_bias=gate_bias
        )

        # Create a 60-step input batch
        generator = torch.Generator().manual_seed(seed + 100)
        x, labels = make_recall_batch(
            BATCH_SIZE,
            SEQ_LEN,
            generator
        )

        # Track gradients on the input tensor
        x.requires_grad_(True)

        # Use the loss from the final prediction only
        logits = model(x)
        loss = F.cross_entropy(logits, labels)

        model.zero_grad()
        loss.backward()

        # Gradient norm for each input step
        # Shape of x.grad: (batch, sequence, input features)
        step_norms = torch.linalg.vector_norm(
            x.grad,
            ord=2,
            dim=2
        ).mean(dim=0)

        seed_gradients.append(step_norms.detach().cpu().tolist())

        print(
            f"{name}: seed={seed}, "
            f"loss={loss.item():.4f}"
        )

    # Median gradient norm at each of the 60 steps
    median_curve = []

    for step in range(SEQ_LEN):
        values = [curve[step] for curve in seed_gradients]
        median_curve.append(statistics.median(values))

    all_results[name] = median_curve

    print(f"\n{name}: gradient norms at selected steps")

    for steps_back in [0, 9, 29, 59]:
        input_step = SEQ_LEN - 1 - steps_back
        print(
            f"{steps_back} steps back: "
            f"{median_curve[input_step]:.8e}"
        )


# Save median gradient measurements
with open("gradient_results.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Configuration",
        "Input_Step",
        "Steps_Back_From_Loss",
        "Median_Gradient_Norm"
    ])

    for name, curve in all_results.items():
        for input_step, norm in enumerate(curve):
            steps_back = SEQ_LEN - 1 - input_step

            writer.writerow([
                name,
                input_step,
                steps_back,
                norm
            ])


# Plot the three curves on a logarithmic scale
plt.figure(figsize=(10, 6))

for name, curve in all_results.items():
    # Avoid zero values on the logarithmic axis
    safe_curve = [max(value, 1e-12) for value in curve]

    steps_back = list(range(SEQ_LEN - 1, -1, -1))

    plt.plot(
        steps_back,
        safe_curve,
        label=name
    )

plt.yscale("log")
plt.xlabel("Steps Back From Final Loss")
plt.ylabel("Median Input Gradient Norm (log scale)")
plt.title("Gradient Through Time: 60-Step Sequence")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "figures/gradient_through_time.png",
    dpi=150
)

plt.show()

print("\nGradient measurement completed.")
print("Saved: gradient_results.csv")
print("Saved: figures/gradient_through_time.png")
