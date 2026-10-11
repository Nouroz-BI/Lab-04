
import time
import torch

from lab04_starter import RecallNet, make_recall_batch

# Settings for the timing experiment
sequence_lengths = [50, 100, 200, 400]
hidden_size = 32
batch_size = 16
repeats = 3

device = torch.device("cpu")
results = []

# Test each sequence length
for seq_len in sequence_lengths:
    print(f"\nTesting sequence length: {seq_len}")

    # Create the LSTM model
    model = RecallNet("lstm", hidden=hidden_size).to(device)

    # Create a generator for reproducible input data
    generator = torch.Generator().manual_seed(42)

    # Create one batch of input data
    x, y = make_recall_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        generator=generator
    )

    x = x.to(device)
    y = y.to(device)

    # Warm up the model before measuring
    for _ in range(2):
        model.zero_grad(set_to_none=True)

        output = model(x)

        # The model returns a 2-dimensional output
        loss = torch.nn.functional.cross_entropy(output, y)

        loss.backward()

    # Measure forward and backward time three times
    times = []

    for repeat in range(repeats):
        model.zero_grad(set_to_none=True)

        start_time = time.perf_counter()

        # Forward pass
        output = model(x)

        # Calculate the loss
        loss = torch.nn.functional.cross_entropy(output, y)

        # Backward pass
        loss.backward()

        elapsed = time.perf_counter() - start_time
        times.append(elapsed)

        print(
            f"Repeat {repeat + 1}: "
            f"{elapsed:.4f} seconds"
        )

    # Keep the best time from three repeats
    best_time = min(times)
    results.append((seq_len, best_time))

    print(f"Best time: {best_time:.4f} seconds")
    print("-" * 40)

# Display the final results
print("\nFinal Timing Results")
print("Sequence Length | Best Time (seconds)")

for seq_len, best_time in results:
    print(f"{seq_len:15} | {best_time:.4f}")

print("\nTiming experiment completed.")
