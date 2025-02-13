package io.github.karolbystrek.layers;

import io.github.karolbystrek.core.Tensor;

public class OutputLayer implements Layer {

    private final int numNodesIn;
    private final int numNodesOut;

    private Tensor lastInput;
    private final float[] lastWeightedInput;

    private float[][] weightsGradient;
    private float[] biasesGradient;

    private float[][] weights;
    private float[] biases;

    public OutputLayer(int numNodesIn, int numNodesOut) {
        this.numNodesIn = numNodesIn;
        this.numNodesOut = numNodesOut;

        this.lastWeightedInput = new float[numNodesOut];

        initializeWeights();
        initializeBiases();
    }

    @Override
    public Tensor forward(Tensor input) {
        if (input.getWidth() != numNodesIn) {
            throw new IllegalArgumentException("Input width must be the number of input nodes");
        }

        lastInput = input;
        Tensor output = new Tensor(1, 1, numNodesOut);

        float maxLogit = Float.NEGATIVE_INFINITY;
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            lastWeightedInput[nodeOut] = biases[nodeOut];
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                lastWeightedInput[nodeOut] += weights[nodeOut][nodeIn] * input.getValue(0, 0, nodeIn);
            }

            if (lastWeightedInput[nodeOut] > maxLogit) {
                maxLogit = lastWeightedInput[nodeOut];
            }
        }

        float sumExp = 0.0f;
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            float value = (float) Math.exp(lastWeightedInput[nodeOut] - maxLogit);
            output.setValue(0, 0, nodeOut, value);
            sumExp += value;
        }

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            float value = output.getValue(0, 0, nodeOut);
            value /= sumExp;
            output.setValue(0, 0, nodeOut, value);
        }

        return output;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        float[][][] gradOutData = gradOutput.getData();
        float[] delta = new float[numNodesOut];
        float[] gradInputFlat = new float[numNodesIn];

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            delta[nodeOut] = gradOutData[0][0][nodeOut];
            biasesGradient[nodeOut] += delta[nodeOut];

            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                float inputValue = lastInput.getValue(0, 0, nodeIn);
                weightsGradient[nodeOut][nodeIn] += delta[nodeOut] * inputValue;

                gradInputFlat[nodeIn] += weights[nodeOut][nodeIn] * delta[nodeOut];
            }
        }

        Tensor gradInput = new Tensor(1, 1, numNodesIn);
        for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
            gradInput.setValue(0, 0, nodeIn, gradInputFlat[nodeIn]);
        }

        return gradInput;
    }

    @Override
    public void updateParameters(float learningRate) {
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            biases[nodeOut] -= learningRate * biasesGradient[nodeOut];
            biasesGradient[nodeOut] = 0.0f;

            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                weights[nodeOut][nodeIn] -= learningRate * weightsGradient[nodeOut][nodeIn];
                weightsGradient[nodeOut][nodeIn] = 0.0f;
            }
        }
    }

    private void initializeWeights() {
        weights = new float[numNodesOut][numNodesIn];
        weightsGradient = new float[numNodesOut][numNodesIn];

        float limit = (float) Math.sqrt(1.0 / numNodesIn);
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                weights[nodeOut][nodeIn] = (float) ((Math.random() * 2 - 1) * limit);
                weightsGradient[nodeOut][nodeIn] = 0.0f;
            }
        }
    }

    private void initializeBiases() {
        biases = new float[numNodesOut];
        biasesGradient = new float[numNodesOut];

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            biases[nodeOut] = 0.0f;
            biasesGradient[nodeOut] = 0.0f;
        }
    }
}
