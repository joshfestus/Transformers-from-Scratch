"""Chapter 1. A neuron, a layer, and a network, in plain Python.

No libraries. Everything here is lists and loops, so that the word
"model" becomes a concrete object before any array library appears.
"""


def neuron(inputs, weights, bias):
    """Multiply each input by its weight, add them up, add the bias."""
    total = bias
    for value, weight in zip(inputs, weights):
        total += value * weight
    return total


def layer(inputs, all_weights, biases):
    """A layer is several neurons reading the same inputs."""
    return [neuron(inputs, weights, bias)
            for weights, bias in zip(all_weights, biases)]


def relu(values):
    """Keep the positives, flatten the negatives to nought."""
    return [value if value > 0 else 0.0 for value in values]


def network(inputs):
    """Two inputs, two hidden neurons, one output."""
    hidden = layer(inputs,
                   all_weights=[[0.5, 1.0], [-0.5, 0.5]],
                   biases=[0.0, 0.5])
    active = relu(hidden)
    output = layer(active, all_weights=[[1.0, -1.0]], biases=[0.5])
    return hidden, active, output[0]


def main():
    print("ONE NEURON")
    print("  inputs [1.0, 2.0], weights [0.5, 1.0], bias 0.0")
    print("  output", neuron([1.0, 2.0], [0.5, 1.0], 0.0))

    print("\nTHE SAME NEURON, WITH A DIFFERENT INPUT")
    print("  inputs [2.0, -1.0]")
    print("  output", neuron([2.0, -1.0], [0.5, 1.0], 0.0))

    print("\nA WHOLE NETWORK")
    hidden, active, output = network([1.0, 2.0])
    print("  hidden before relu", hidden)
    print("  hidden after relu ", active)
    print("  output            ", output)

    print("\nTHE SAME NETWORK, ONE INPUT CHANGED")
    hidden, active, output = network([1.0, -2.0])
    print("  hidden before relu", hidden)
    print("  hidden after relu ", active)
    print("  output            ", output)


if __name__ == "__main__":
    main()
