"""Kapitel 1. Ein Neuron, eine Schicht und ein Netz, in reinem Python.

Keine Bibliotheken. Alles hier sind Listen und Schleifen, damit das Wort
„Modell“ ein greifbares Objekt wird, bevor Array-Bibliotheken auftauchen.
"""


def neuron(inputs, weights, bias):
    """Jede Eingabe mal ihr Gewicht, alles aufaddieren, Bias dazu."""
    total = bias
    for value, weight in zip(inputs, weights):
        total += value * weight
    return total


def layer(inputs, all_weights, biases):
    """Eine Schicht: mehrere Neuronen, die dieselben Eingaben lesen."""
    return [neuron(inputs, weights, bias)
            for weights, bias in zip(all_weights, biases)]


def relu(values):
    """Positive Werte behalten, negative auf null setzen."""
    return [value if value > 0 else 0.0 for value in values]


def network(inputs):
    """Zwei Eingaben, zwei verdeckte Neuronen, eine Ausgabe."""
    hidden = layer(inputs,
                   all_weights=[[0.5, 1.0], [-0.5, 0.5]],
                   biases=[0.0, 0.5])
    active = relu(hidden)
    output = layer(active, all_weights=[[1.0, -1.0]], biases=[0.5])
    return hidden, active, output[0]


def main():
    print("EIN NEURON")
    print("  Eingaben [1.0, 2.0], Gewichte [0.5, 1.0], Bias 0.0")
    print("  Ausgabe", neuron([1.0, 2.0], [0.5, 1.0], 0.0))

    print("\nDASSELBE NEURON, MIT EINER ANDEREN EINGABE")
    print("  Eingaben [2.0, -1.0]")
    print("  Ausgabe", neuron([2.0, -1.0], [0.5, 1.0], 0.0))

    print("\nEIN GANZES NETZ")
    hidden, active, output = network([1.0, 2.0])
    print("  verdeckt vor ReLU ", hidden)
    print("  verdeckt nach ReLU", active)
    print("  Ausgabe           ", output)

    print("\nDASSELBE NETZ, EINE EINGABE GEÄNDERT")
    hidden, active, output = network([1.0, -2.0])
    print("  verdeckt vor ReLU ", hidden)
    print("  verdeckt nach ReLU", active)
    print("  Ausgabe           ", output)


if __name__ == "__main__":
    main()
