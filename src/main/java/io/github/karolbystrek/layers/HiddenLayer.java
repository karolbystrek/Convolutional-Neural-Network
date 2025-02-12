package io.github.karolbystrek.layers;

import io.github.karolbystrek.core.Tensor;

public class HiddenLayer implements Layer {

    private final int numNodesIn;
    private final int numNodesOut;

    private Tensor lastInput;
    private final double[] lastWeightedInput;

    private double[][] weightsGradient;
    private double[] biasesGradient;

    private double[][] weights;
    private double[] biases;

    public HiddenLayer(int numNodesIn, int numNodesOut) {
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

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            lastWeightedInput[nodeOut] = biases[nodeOut];
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                lastWeightedInput[nodeOut] += weights[nodeOut][nodeIn] * input.getValue(0, 0, nodeIn);
            }

            output.setValue(0, 0, nodeOut, activation(lastWeightedInput[nodeOut]));
        }

        return output;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        double[][][] gradOutData = gradOutput.getData();
        double[] delta = new double[numNodesOut];
        double[] gradInputFlat = new double[numNodesIn];

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            double dActivation = (lastWeightedInput[nodeOut] > 0) ? 1.0 : 0.0;
            delta[nodeOut] = gradOutData[0][0][nodeOut] * dActivation;
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

        double scale = Math.sqrt(2.0 / numNodesIn);

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                weights[nodeOut][nodeIn] = (Math.random() * 2 - 1) * scale;
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

    private double activation(double weightedInput) {
        return Math.max(0.0, weightedInput);
    }
}
