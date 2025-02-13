package io.github.karolbystrek.reader;

import io.github.karolbystrek.core.Tensor;

public class Cifar10DataPoint implements DataPoint {
    private static final int NUM_LABELS = 10;

    private final Tensor input;
    private final Tensor expectedOutput;

    public Cifar10DataPoint(Tensor image, int label) {
        this.input = image;
        this.expectedOutput = labelToTensor(label);
    }

    private Tensor labelToTensor(int label) {
        if (label < 0 || label > NUM_LABELS - 1) {
            throw new IllegalArgumentException("Invalid label: " + label);
        }

        Tensor tensor = new Tensor(1, 1, NUM_LABELS);
        tensor.setValue(0, 0, label, 1.0f);
        return tensor;
    }

    @Override
    public Tensor getExpectedOutput() {
        return expectedOutput;
    }

    @Override
    public Tensor getInput() {
        return input;
    }
}
