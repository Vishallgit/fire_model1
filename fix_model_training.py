import os
import json
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Configuration
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

SPLITS_DIR = "splits"
SPLIT_FILE = "split_info.json"
SAMPLES_DIR = "Stage1_Samples"
BATCH_SIZE = 64
EPOCHS = 5
STEPS_PER_EPOCH = 50
VAL_STEPS = 10
TEST_STEPS = 10
MAX_RAM_USAGE = 10_000_000_000

VAL_CM_STEPS = 10
TEST_CM_STEPS = 10


def load_splits(splits_dir=SPLITS_DIR, split_file=SPLIT_FILE):
    path = os.path.join(splits_dir, split_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found: {path}")
    with open(path, "r") as f:
        info = json.load(f)
    return info


split_info = load_splits()
train_files = split_info["splits"]["train"]
val_files = split_info["splits"]["validation"]
test_files = split_info["splits"]["test"]
print("[INFO] Loaded split info")


# ----------------------------------------------------------------------------
# Generator
# ----------------------------------------------------------------------------

def chunk_file_generator(file_list, samples_dir, batch_size, oversample=True, shuffle_files=True):
    files = list(file_list)
    if shuffle_files:
        random.shuffle(files)
    for fname in files:
        fpath = os.path.join(samples_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Chunk file not found: {fpath}")
        with np.load(fpath, allow_pickle=True) as data:
            x_data = data["x_data"]
            y_data = data["y_label"]
        approx_mem = x_data.size * x_data.itemsize + y_data.size * y_data.itemsize
        if approx_mem > MAX_RAM_USAGE:
            print(f"[WARNING] chunk {fname} ~{approx_mem/1e9:.2f} GB > limit {MAX_RAM_USAGE/1e9:.2f} GB")
        x_data = np.nan_to_num(x_data, nan=0.0)
        if oversample:
            pos_idx = np.where(y_data == 1)[0]
            neg_idx = np.where(y_data == 0)[0]
            if len(pos_idx) > 0 and len(neg_idx) > 0:
                if len(pos_idx) < len(neg_idx):
                    factor = len(neg_idx) // len(pos_idx)
                    remainder = len(neg_idx) % len(pos_idx)
                    replicated = np.concatenate(
                        [pos_idx] * factor + ([np.random.choice(pos_idx, remainder, replace=False)] if remainder > 0 else [])
                    )
                    new_idx = np.concatenate([neg_idx, replicated])
                else:
                    factor = len(pos_idx) // len(neg_idx)
                    remainder = len(pos_idx) % len(neg_idx)
                    replicated = np.concatenate(
                        [neg_idx] * factor + ([np.random.choice(neg_idx, remainder, replace=False)] if remainder > 0 else [])
                    )
                    new_idx = np.concatenate([pos_idx, replicated])
                np.random.shuffle(new_idx)
                x_data = x_data[new_idx]
                y_data = y_data[new_idx]
        idxs = np.arange(x_data.shape[0])
        np.random.shuffle(idxs)
        start = 0
        while start < len(idxs):
            end = min(start + batch_size, len(idxs))
            b_idx = idxs[start:end]
            start = end
            yield x_data[b_idx], y_data[b_idx]


def make_dataset(file_list, samples_dir, batch_size, oversample=True, shuffle_files=True):
    def gen():
        return chunk_file_generator(file_list, samples_dir, batch_size, oversample, shuffle_files)
    out_types = (tf.float32, tf.int32)
    out_shapes = (
        tf.TensorShape([None, None, 3, 128, 128]),
        tf.TensorShape([None])
    )
    return tf.data.Dataset.from_generator(gen, output_types=out_types, output_shapes=out_shapes)


# ----------------------------------------------------------------------------
# Model Definition (safe names)
# ----------------------------------------------------------------------------

def build_transformer_model(channels, seq_len=3):
    inp = layers.Input(shape=(channels, seq_len, 128, 128), name="InputLayer")
    x = layers.Permute((2, 3, 4, 1), name="PermuteLayer")(inp)

    def make_cnn_block():
        block = models.Sequential(name="TimeCnnBlock")
        block.add(layers.Conv2D(16, (3, 3), padding="same", activation="relu"))
        block.add(layers.MaxPooling2D((2, 2)))
        block.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
        block.add(layers.MaxPooling2D((2, 2)))
        block.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
        block.add(layers.MaxPooling2D((2, 2)))
        block.add(layers.Flatten())
        block.add(layers.Dense(256, activation="relu"))
        return block

    time_cnn = make_cnn_block()
    x = layers.TimeDistributed(time_cnn, name="TimeDistCnnLayer")(x)

    embed_dim = 256
    num_heads = 4
    ff_dim = 256

    attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, name="MHA")(x, x)
    x = layers.Add(name="AttnAdd")([x, attn_out])
    x = layers.LayerNormalization(name="AttnLN")(x)

    ff = layers.Dense(ff_dim, activation="relu", name="FF1")(x)
    ff = layers.Dense(embed_dim, name="FF2")(ff)
    x = layers.Add(name="FFAdd")([x, ff])
    x = layers.LayerNormalization(name="FFLN")(x)

    x = layers.GlobalAveragePooling1D(name="TimeGAP")(x)
    x = layers.Dense(128, activation="relu", name="Dense128")(x)
    x = layers.Dropout(0.3, name="Dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="Output")(x)

    return models.Model(inp, out, name="CnnTransformerModel")


# ----------------------------------------------------------------------------
# Build, Train, Evaluate
# ----------------------------------------------------------------------------
if len(train_files) == 0:
    raise ValueError("No train files found in 'train' split_info")

first_train_file = os.path.join(SAMPLES_DIR, train_files[0])
with np.load(first_train_file, allow_pickle=True) as data:
    detected_channels = data["x_data"].shape[1]
print(f"[INFO] Detected channels = {detected_channels}")

model = build_transformer_model(detected_channels, seq_len=3)
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4, clipnorm=5.0),
    loss=lambda yt, yp: tf.reduce_mean(tf.keras.losses.binary_crossentropy(yt, yp)),
    metrics=["accuracy"],
)
model.summary()

train_ds = make_dataset(train_files, SAMPLES_DIR, BATCH_SIZE, oversample=True, shuffle_files=True)
val_ds = make_dataset(val_files, SAMPLES_DIR, BATCH_SIZE, oversample=False, shuffle_files=False)
test_ds = make_dataset(test_files, SAMPLES_DIR, BATCH_SIZE, oversample=False, shuffle_files=False)

print(f"\n>>> Starting training for {EPOCHS} epochs ...")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_ds,
    validation_steps=VAL_STEPS,
)

print("\n>>> Evaluating on test dataset ...")
test_metrics = model.evaluate(test_ds, steps=TEST_STEPS, return_dict=True)
print("Test metrics:", test_metrics)
