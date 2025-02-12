package io.github.karolbystrek.layers;

import io.github.karolbystrek.core.Tensor;

public class OutputLayer implements Layer {

    private final int numNodesIn;
    private final int numNodesOut;

    private Tensor lastInput;
    private final double[] lastWeightedInput;

    private double[][] weightsGradient;
    private double[] biasesGradient;

    private double[][] weights;
    private double[] biases;

    public OutputLayer(int numNodesIn, int numNodesOut) {
        this.numNodesIn = numNodesIn;
        this.numNodesOut = numNodesOut;

        this.lastWeightedInput = new double[numNodesOut];

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

        double maxLogit = Double.NEGATIVE_INFINITY;
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            lastWeightedInput[nodeOut] = biases[nodeOut];
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                lastWeightedInput[nodeOut] += weights[nodeOut][nodeIn] * input.getValue(0, 0, nodeIn);
            }

            if (lastWeightedInput[nodeOut] > maxLogit) {
                maxLogit = lastWeightedInput[nodeOut];
            }
        }

        double sumExp = 0.0;
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            double value = Math.exp(lastWeightedInput[nodeOut] - maxLogit);
            output.setValue(0, 0, nodeOut, value);
            sumExp += value;
        }

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            double value = output.getValue(0, 0, nodeOut);
            value /= sumExp;
            output.setValue(0, 0, nodeOut, value);
        }

        return output;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        double[][][] gradOutData = gradOutput.getData();
        double[] delta = new double[numNodesOut];
        double[] gradInputFlat = new double[numNodesIn];

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            delta[nodeOut] = gradOutData[0][0][nodeOut];
            biasesGradient[nodeOut] += delta[nodeOut];

            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                double inputValue = lastInput.getValue(0, 0, nodeIn);
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
    public void updateParameters(double learningRate) {
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            biases[nodeOut] -= learningRate * biasesGradient[nodeOut];
            biasesGradient[nodeOut] = 0.0;

            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                weights[nodeOut][nodeIn] -= learningRate * weightsGradient[nodeOut][nodeIn];
                weightsGradient[nodeOut][nodeIn] = 0.0;
            }
        }
    }

    private void initializeWeights() {
        weights = new double[numNodesOut][numNodesIn];
        weightsGradient = new double[numNodesOut][numNodesIn];

        double limit = Math.sqrt(1.0 / numNodesIn);
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                weights[nodeOut][nodeIn] = (Math.random() * 2 - 1) * limit;
                weightsGradient[nodeOut][nodeIn] = 0.0;
            }
        }
    }

    private void initializeBiases() {
        biases = new double[numNodesOut];
        biasesGradient = new double[numNodesOut];

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            biases[nodeOut] = 0.0;
            biasesGradient[nodeOut] = 0.0;
        }
    }
}
