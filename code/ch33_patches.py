"""Chapter 33. An image is a sequence once you cut it up.

Nothing in the architecture mentions language. Attention operates on a
set of vectors. Anything that can be turned into a set of vectors can be
put through the same stack, and this file does the turning.
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


def patchify(image, patch):
    """Cut an image into square patches and flatten each one."""
    height, width, channels = image.shape
    assert height % patch == 0 and width % patch == 0, "patch must divide"
    rows, cols = height // patch, width // patch
    out = image.reshape(rows, patch, cols, patch, channels)
    out = out.transpose(0, 2, 1, 3, 4)
    return out.reshape(rows * cols, patch * patch * channels)


def patch_embedding_parameters(patch, channels, d_model):
    return patch * patch * channels * d_model + d_model


def sequence_length(size, patch):
    return (size // patch) ** 2


def main():
    size, patch, channels, d_model = 224, 16, 3, 768
    image = np.random.default_rng(0).random((size, size, channels))

    print("ONE IMAGE BECOMES A SEQUENCE")
    tokens = patchify(image, patch)
    print(f"  image        {image.shape}")
    print(f"  patches      {tokens.shape}, one row per patch")
    print(f"  each row is {patch} by {patch} by {channels} flattened")
    print(f"  a linear layer maps {tokens.shape[1]} to {d_model}")
    print(f"  what the stack sees: ({tokens.shape[0]}, {d_model})")
    print("  which is the same shape a sentence of 196 tokens gives")

    print("\nTHE PATCH SIZE DECIDES THE SEQUENCE LENGTH")
    print("  patch   tokens   attention cost, relative")
    base = sequence_length(size, 16) ** 2
    for p in [8, 14, 16, 32]:
        n = sequence_length(size, p)
        print(f"  {p:<7} {n:<8} {n * n / base:>8.2f}")
    print("  halving the patch quadruples the tokens and multiplies")
    print("  the attention cost by sixteen")

    print("\nTHE PATCH EMBEDDING IS ONE SMALL MATRIX")
    count = patch_embedding_parameters(patch, channels, d_model)
    print(f"  {patch}x{patch}x{channels} into {d_model}: {count:,} "
          f"parameters")
    print(f"  one transformer block at this width: "
          f"{4 * d_model ** 2 + 8 * d_model ** 2:,}")
    print("  the part that handles images at all is the smallest part")

    print("\nWHAT CHANGES AND WHAT DOES NOT")
    print("  changes    how the input is cut up, and the positions")
    print("  unchanged  attention, the feedforward network, the norms,")
    print("             the residuals, the block, the stack")

    print("\nOTHER KINDS OF INPUT, SAME TREATMENT")
    others = [("audio", "25ms frame of a spectrogram", 3000),
              ("video", "patch of one frame", 9408),
              ("protein", "amino acid", 350)]
    for kind, unit, count_ in others:
        print(f"  {kind:<9} one token per {unit:<28} about {count_}")


if __name__ == "__main__":
    main()
