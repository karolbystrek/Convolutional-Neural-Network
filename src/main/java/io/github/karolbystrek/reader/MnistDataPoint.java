package io.github.karolbystrek.reader;

import io.github.karolbystrek.core.Tensor;

public class MnistDataPoint implements DataPoint {

    private static final int NUM_LABELS = 10;

    private final Tensor input;
    private final Tensor expectedOutput;

    public MnistDataPoint(Tensor image, int label) {
        this.input = image;
        this.expectedOutput = labelToTensor(label);
    }

    private Tensor labelToTensor(int label) {
        if (label < 0 || label > NUM_LABELS - 1) {
            throw new IllegalArgumentException("Invalid label: " + label);
        }

        float[][][] tensorData = new float[1][1][NUM_LABELS];
        tensorData[0][0][label] = 1.0f;

        return new Tensor(tensorData);
    }

    @Override
    public Tensor getInput() {
        return input;
    }

    @Override
    public Tensor getExpectedOutput() {
        return expectedOutput;
    }
}
