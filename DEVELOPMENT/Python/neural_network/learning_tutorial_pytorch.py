# =============================================================================
# TUTORIAL: Understanding a Tiny Neural Network (MLP) from scratch
# Run this script top to bottom. Each section builds on the previous one.
# You will SEE exactly what is happening at every step.
#
# !! TUTORIAL IS FOR LEARNING PURPOSES AND HAS BEEN MADE BY THE AI LLM CLAUDE
# Some THINGS ARE A BIT USELESS OTHERS ARE VERY USEFUL TO BE USED WITH CAUTION
#
# =============================================================================
# To run this script:
#   pip install torch numpy matplotlib pillow
#   python mlp_tutorial.py
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# PART 1: THE PROBLEM WE ARE SOLVING
# =============================================================================
# Forget the drone for a second. Let's build the simplest possible version
# of the same problem:
#
#   INPUT:  A tiny 4x4 black and white image (16 pixels)
#           White pixels (1.0) = safe ground
#           Black pixels (0.0) = obstacle
#
#   OUTPUT: Is there danger on the LEFT side? Is there danger on the RIGHT side?
#           Two outputs, each between 0 and 1.
#           0.0 = totally safe, 1.0 = definite danger
#
# This is EXACTLY the same structure as your drone problem, just smaller.
# =============================================================================

print("=" * 60)
print("PART 1: THE PROBLEM")
print("=" * 60)

# Let's manually create 4 example images and their labels
# Each image is 4x4 pixels, flattened to 16 values
# 1.0 = white (ground/safe), 0.0 = black (obstacle)

# Image 1: obstacle on the LEFT side
img1 = np.array([
    [0, 0, 1, 1],   # row 0: left side black (obstacle), right side white (safe)
    [0, 0, 1, 1],   # row 1
    [0, 0, 1, 1],   # row 2
    [0, 0, 1, 1],   # row 3
], dtype=np.float32)
label1 = np.array([1, 0], dtype=np.float32)  # danger LEFT=1, danger RIGHT=0

# Image 2: obstacle on the RIGHT side
img2 = np.array([
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 0, 0],
], dtype=np.float32)
label2 = np.array([0, 1], dtype=np.float32)  # danger LEFT=0, danger RIGHT=1

# Image 3: obstacles on BOTH sides
img3 = np.array([
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
], dtype=np.float32)
label3 = np.array([1, 1], dtype=np.float32)  # danger LEFT=1, danger RIGHT=1

# Image 4: completely clear
img4 = np.array([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
], dtype=np.float32)
label4 = np.array([0, 0], dtype=np.float32)  # danger LEFT=0, danger RIGHT=0

print("\nHere are our 4 training images (1=white/safe, 0=black/obstacle):")
print("\nImage 1 (obstacle LEFT) - label: danger_left=1, danger_right=0")
print(img1)
print("\nImage 2 (obstacle RIGHT) - label: danger_left=0, danger_right=1")
print(img2)
print("\nImage 3 (obstacles BOTH) - label: danger_left=1, danger_right=1")
print(img3)
print("\nImage 4 (completely clear) - label: danger_left=0, danger_right=0")
print(img4)

# =============================================================================
# PART 2: PREPARING THE DATA
# =============================================================================
# A neural network cannot take a 2D image directly.
# We need to FLATTEN it - turn it from a grid into a single long list.
# 4x4 image -> 16 numbers in a row
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: PREPARING THE DATA")
print("=" * 60)

print("\nBefore flattening, img1 has shape:", img1.shape)
img1_flat = img1.flatten()
print("After flattening, img1 has shape:", img1_flat.shape)
print("Flattened img1:", img1_flat)
print("\nThis is just the rows laid out one after another.")
print("The neural network sees this 1D list, not the 2D grid.")

# Stack all images and labels into arrays
images = np.array([img1.flatten(),
                   img2.flatten(),
                   img3.flatten(),
                   img4.flatten()])   # shape: (4, 16)

labels = np.array([label1, label2, label3, label4])  # shape: (4, 2)

print(f"\nAll images stacked: shape = {images.shape}")
print("(4 images, each with 16 pixel values)")
print(f"\nAll labels stacked: shape = {labels.shape}")
print("(4 labels, each with 2 values: [danger_left, danger_right])")

# Convert to PyTorch tensors
# A tensor is just PyTorch's version of a numpy array - it can track gradients
X = torch.tensor(images)   # shape: (4, 16)
y = torch.tensor(labels)   # shape: (4, 2)

print("\nConverted to PyTorch tensors.")
print("X (inputs):\n", X)
print("\ny (labels):\n", y)

# =============================================================================
# PART 3: WHAT IS A NEURAL NETWORK ACTUALLY DOING?
# =============================================================================
# A single layer of the network does this:
#
#   output = ReLU(input × weights + bias)
#
# Let's break that down:
#   - "input" is our 16 pixel values
#   - "weights" is a matrix of numbers the network LEARNS
#   - "bias" is an extra number added (like the y-intercept in y=mx+b)
#   - "ReLU" means: if the number is negative, make it 0. Otherwise keep it.
#
# The network starts with RANDOM weights, makes bad predictions,
# then slowly adjusts the weights to make better predictions.
# This adjustment process is called TRAINING.
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: WHAT IS THE NETWORK DOING?")
print("=" * 60)

print("\nLet's manually do what ONE neuron does:")
print("A neuron takes all 16 inputs, multiplies each by a weight, adds them up.")

# Simulate one neuron manually
np.random.seed(42)
example_weights = np.random.randn(16).astype(np.float32) * 0.1
example_bias = 0.0
example_input = img1.flatten()

weighted_sum = np.dot(example_input, example_weights) + example_bias
print(f"\nInput (16 pixels): {example_input}")
print(f"Weights (16 random numbers): {example_weights.round(3)}")
print(f"Weighted sum (dot product + bias): {weighted_sum:.4f}")

# Apply ReLU
relu_output = max(0, weighted_sum)
print(f"After ReLU (max(0, x)): {relu_output:.4f}")
print("\nOne neuron = one number output.")
print("Our hidden layer has 8 neurons, so it produces 8 numbers.")
print("Those 8 numbers get passed to the output layer.")

# =============================================================================
# PART 4: BUILDING THE ACTUAL NETWORK
# =============================================================================

print("\n" + "=" * 60)
print("PART 4: BUILDING THE NETWORK")
print("=" * 60)

class TinyMLP(nn.Module):
    def __init__(self):
        super(TinyMLP, self).__init__()
        # Layer 1: 16 inputs -> 8 hidden neurons
        self.fc1 = nn.Linear(16, 8)
        # Layer 2: 8 hidden neurons -> 2 outputs
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        # Step 1: first layer + ReLU
        x = self.fc1(x)        # matrix multiply + add bias
        x = torch.relu(x)      # apply ReLU (negative -> 0)
        # Step 2: second layer + Sigmoid
        x = self.fc2(x)        # matrix multiply + add bias
        x = torch.sigmoid(x)   # squish output to be between 0 and 1
        return x

model = TinyMLP()
print("\nNetwork structure:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal learnable parameters: {total_params}")
print("Layer 1: 16 inputs × 8 neurons = 128 weights + 8 biases = 136")
print("Layer 2: 8 inputs × 2 outputs = 16 weights + 2 biases = 18")
print(f"Total: 136 + 18 = {total_params}")

# =============================================================================
# PART 5: WHAT DOES THE UNTRAINED NETWORK OUTPUT?
# =============================================================================
# Before training, the weights are random, so the predictions are random.
# Let's see what it outputs RIGHT NOW before any training.
# =============================================================================

print("\n" + "=" * 60)
print("PART 5: PREDICTIONS BEFORE TRAINING (random weights)")
print("=" * 60)

model.eval()
with torch.no_grad():  # don't track gradients during prediction
    predictions_before = model(X)

print("\nInputs (4 images)         True labels    Network output (BEFORE training)")
print("-" * 70)
label_names = ["LEFT obstacle", "RIGHT obstacle", "BOTH obstacles", "CLEAR"]
for i in range(4):
    true = y[i].numpy()
    pred = predictions_before[i].numpy()
    print(f"{label_names[i]:<20}  true={true}  pred=[{pred[0]:.2f}, {pred[1]:.2f}]")

print("\nAs you can see, the predictions are basically random (around 0.5).")
print("The network has no idea what it's looking at yet.")

# =============================================================================
# PART 6: TRAINING - HOW THE NETWORK LEARNS
# =============================================================================
# Training works like this:
#
# 1. Show the network an image
# 2. It makes a prediction
# 3. We calculate the LOSS (how wrong it was)
#    - Loss = 0 means perfect prediction
#    - Loss = high means very wrong prediction
# 4. We calculate GRADIENTS (which direction to nudge each weight)
# 5. We nudge the weights a tiny bit in the right direction
# 6. Repeat thousands of times
#
# The loss function we use is BCELoss (Binary Cross Entropy):
#   - For each output, it measures how far the prediction is from the truth
#   - If truth=1 and prediction=0.9, loss is small (good!)
#   - If truth=1 and prediction=0.1, loss is large (bad!)
# =============================================================================

print("\n" + "=" * 60)
print("PART 6: TRAINING THE NETWORK")
print("=" * 60)

criterion = nn.BCELoss()   # our loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)  # the optimizer that nudges weights

print("\nLet's manually do ONE training step so you can see what happens:")

model.train()
# Forward pass: make a prediction
output = model(X)
print(f"\nPredictions: \n{output.detach().numpy().round(3)}")
print(f"True labels: \n{y.numpy()}")

# Calculate loss
loss = criterion(output, y)
print(f"\nLoss (how wrong we are): {loss.item():.4f}")
print("This number will go DOWN as the network learns.")

# Backward pass: calculate gradients
optimizer.zero_grad()   # reset gradients from last step
loss.backward()         # calculate gradients (backpropagation)
optimizer.step()        # nudge the weights

print("\nWeights have been nudged. Loss should be slightly lower now.")

# Check new loss
output = model(X)
new_loss = criterion(output, y)
print(f"Loss after one step: {new_loss.item():.4f}  (was {loss.item():.4f})")

# =============================================================================
# PART 7: FULL TRAINING LOOP
# Now let's train properly for many epochs and watch the loss go down
# =============================================================================

print("\n" + "=" * 60)
print("PART 7: FULL TRAINING LOOP (500 steps)")
print("=" * 60)

# Reset the model to fresh random weights
model = TinyMLP()
optimizer = optim.Adam(model.parameters(), lr=0.01)

loss_history = []

print("\nEpoch  | Loss   | What's happening")
print("-" * 50)

for epoch in range(500):
    model.train()

    # Forward pass
    output = model(X)

    # Calculate loss
    loss = criterion(output, y)
    loss_history.append(loss.item())

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress every 50 epochs
    if (epoch + 1) % 50 == 0:
        if epoch < 100:
            note = "Still learning..."
        elif epoch < 300:
            note = "Getting better!"
        else:
            note = "Converging..."
        print(f"  {epoch+1:3d}  | {loss.item():.4f} | {note}")

print("\nTraining complete!")

# =============================================================================
# PART 8: PREDICTIONS AFTER TRAINING
# =============================================================================

print("\n" + "=" * 60)
print("PART 8: PREDICTIONS AFTER TRAINING")
print("=" * 60)

model.eval()
with torch.no_grad():
    predictions_after = model(X)

print("\nInputs                True labels    Network output (AFTER training)")
print("-" * 70)
for i in range(4):
    true = y[i].numpy()
    pred = predictions_after[i].numpy()
    # Threshold at 0.5: above 0.5 = predicts danger, below = predicts safe
    pred_binary = (pred > 0.5).astype(int)
    correct = "✓ CORRECT" if np.array_equal(pred_binary, true.astype(int)) else "✗ WRONG"
    print(f"{label_names[i]:<20}  true={true.astype(int)}  "
          f"pred=[{pred[0]:.2f}, {pred[1]:.2f}]  -> {pred_binary}  {correct}")

print("\nThe network has learned to distinguish the patterns!")
print("Values close to 1.0 = 'I'm pretty sure there's danger here'")
print("Values close to 0.0 = 'I'm pretty sure it's safe here'")

# =============================================================================
# PART 9: TESTING ON A NEW IMAGE THE NETWORK HAS NEVER SEEN
# =============================================================================

print("\n" + "=" * 60)
print("PART 9: TESTING ON A NEW UNSEEN IMAGE")
print("=" * 60)

# New image: mostly clear but slight obstacle on the left
new_image = np.array([
    [0, 1, 1, 1],   # small obstacle top-left
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
], dtype=np.float32)

print("\nNew image (network has never seen this):")
print(new_image)
print("There's a small obstacle in the top-left corner.")
print("Expected: some left danger, no right danger")

new_tensor = torch.tensor(new_image.flatten()).unsqueeze(0)  # add batch dimension

model.eval()
with torch.no_grad():
    prediction = model(new_tensor)[0]

danger_left = prediction[0].item()
danger_right = prediction[1].item()

print(f"\nNetwork output:")
print(f"  Danger LEFT:  {danger_left:.3f}  -> {'DANGER!' if danger_left > 0.5 else 'safe'}")
print(f"  Danger RIGHT: {danger_right:.3f} -> {'DANGER!' if danger_right > 0.5 else 'safe'}")

# Decision
if danger_left > 0.5 and danger_right <= 0.5:
    decision = "Turn RIGHT (left side is dangerous)"
elif danger_right > 0.5 and danger_left <= 0.5:
    decision = "Turn LEFT (right side is dangerous)"
elif danger_left > 0.5 and danger_right > 0.5:
    decision = "Go UP or STOP (both sides dangerous)"
else:
    decision = "Go STRAIGHT (all clear!)"

print(f"\nDrone decision: {decision}")

# =============================================================================
# PART 10: VISUALIZING EVERYTHING WITH PLOTS
# =============================================================================

print("\n" + "=" * 60)
print("PART 10: CREATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("MLP Tutorial: Drone Obstacle Avoidance", fontsize=14, fontweight='bold')

# --- Top row: the 4 training images ---
images_2d = [img1, img2, img3, img4]
for i, (img, lbl, name) in enumerate(zip(images_2d, [label1, label2, label3, label4], label_names)):
    ax = axes[0, i]
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f"{name}\nlabel: L={int(lbl[0])}, R={int(lbl[1])}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    # Draw dividing line in middle
    ax.axvline(x=1.5, color='red', linewidth=2, linestyle='--', alpha=0.7)

axes[0, 0].set_ylabel("Training Images\n(white=safe, black=obstacle)", fontsize=9)

# Label the dividing line
for ax in axes[0]:
    ax.text(0.5, -0.15, "L | R", transform=ax.transAxes,
            ha='center', fontsize=8, color='red')

# --- Bottom row: loss curve + predictions comparison ---

# Loss curve
ax_loss = axes[1, 0]
ax_loss.plot(loss_history, color='blue', linewidth=2)
ax_loss.set_title("Training Loss Over Time", fontsize=10)
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss")
ax_loss.grid(True, alpha=0.3)
ax_loss.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Perfect = 0')
ax_loss.legend(fontsize=8)

# Before vs After predictions
ax_compare = axes[1, 1]
x_positions = np.arange(4)
width = 0.35
before_left = predictions_before[:, 0].numpy()
after_left = predictions_after[:, 0].numpy()
true_left = y[:, 0].numpy()

bars1 = ax_compare.bar(x_positions - width/2, before_left, width,
                        label='Before training', color='red', alpha=0.6)
bars2 = ax_compare.bar(x_positions + width/2, after_left, width,
                        label='After training', color='blue', alpha=0.6)
ax_compare.plot(x_positions, true_left, 'g*', markersize=12,
                label='True label', zorder=5)
ax_compare.set_title("LEFT danger predictions\n(Before vs After training)", fontsize=9)
ax_compare.set_xticks(x_positions)
ax_compare.set_xticklabels(['LEFT\nobs', 'RIGHT\nobs', 'BOTH\nobs', 'CLEAR'], fontsize=7)
ax_compare.set_ylabel("Predicted danger (0-1)")
ax_compare.set_ylim(0, 1.1)
ax_compare.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Decision threshold')
ax_compare.legend(fontsize=7)
ax_compare.grid(True, alpha=0.3)

# New image prediction visualization
ax_new = axes[1, 2]
ax_new.imshow(new_image, cmap='gray', vmin=0, vmax=1)
ax_new.set_title(f"New unseen image\nL danger: {danger_left:.2f} | R danger: {danger_right:.2f}",
                  fontsize=9)
ax_new.axvline(x=1.5, color='red', linewidth=2, linestyle='--', alpha=0.7)
ax_new.set_xticks([])
ax_new.set_yticks([])
# Color the background based on danger
left_color = 'red' if danger_left > 0.5 else 'green'
right_color = 'red' if danger_right > 0.5 else 'green'
ax_new.add_patch(mpatches.Rectangle((0, 0), 1.5, 4,
    alpha=0.2, color=left_color, transform=ax_new.transData))
ax_new.add_patch(mpatches.Rectangle((1.5, 0), 2, 4,
    alpha=0.2, color=right_color, transform=ax_new.transData))

# Network diagram
ax_net = axes[1, 3]
ax_net.set_xlim(0, 10)
ax_net.set_ylim(0, 10)
ax_net.set_title("Network Architecture", fontsize=10)
ax_net.axis('off')

# Draw simplified network diagram
layer_x = [1.5, 4, 6.5, 9]
layer_labels = ['Input\n(16px)', 'Hidden\n(8)', 'Output\n(2)', '']
layer_colors = ['lightblue', 'lightyellow', 'lightgreen', 'white']
layer_sizes = [6, 4, 2, 0]

for lx, ll, lc, ls in zip(layer_x[:-1], layer_labels[:-1], layer_colors[:-1], layer_sizes[:-1]):
    # Draw neurons
    spacing = 8 / (ls + 1)
    for n in range(ls):
        y_pos = 1 + spacing * (n + 1)
        circle = plt.Circle((lx, y_pos), 0.3, color=lc, ec='black', linewidth=1)
        ax_net.add_patch(circle)

    ax_net.text(lx, 0.3, ll, ha='center', fontsize=8, fontweight='bold')

# Draw arrows between layers
ax_net.annotate('', xy=(3.3, 5), xytext=(2.2, 5),
                arrowprops=dict(arrowstyle='->', color='gray'))
ax_net.annotate('', xy=(5.8, 5), xytext=(4.7, 5),
                arrowprops=dict(arrowstyle='->', color='gray'))

ax_net.text(3.5, 9.5, 'ReLU', ha='center', fontsize=8, color='orange',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax_net.text(6, 9.5, 'Sigmoid', ha='center', fontsize=8, color='purple',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))

plt.tight_layout()
plt.savefig('tutorial_output.png', dpi=120, bbox_inches='tight')
print("Plot saved to tutorial_output.png")
plt.show()

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY: WHAT JUST HAPPENED")
print("=" * 60)
print("""
1. We created 4 tiny 4x4 images with known obstacle positions

2. We FLATTENED them from 2D grids into 1D lists of 16 numbers

3. We built a tiny network:
   - Layer 1: 16 inputs -> 8 neurons (learns features like "is left side dark?")
   - ReLU: kills negative values (adds non-linearity, helps learning)
   - Layer 2: 8 neurons -> 2 outputs (one per danger zone)
   - Sigmoid: squashes output to 0-1 range (probability)

4. BEFORE training: random weights -> random (bad) predictions

5. TRAINING LOOP (500 times):
   - Make prediction
   - Calculate loss (BCELoss: how wrong are we?)
   - Backpropagation: figure out which weights caused the error
   - Nudge weights slightly in the right direction (Adam optimizer)

6. AFTER training: network correctly identifies danger zones

7. On a NEW unseen image: network generalizes what it learned

For your drone project:
   - Replace the 4x4 images with your real B&W masks
   - Replace 2 output zones with 4 zones (Left/CL/CR/Right)
   - Everything else stays the same!
""")
