package io.github.karolbystrek.core;

public class Kernel {

    private final double[][][] weights;
    private double bias;

    public double[][][] weightsGradient;
    public double biasGradient;

    public Kernel(int depth, int height, int width) {
        weights = new double[depth][height][width];
        bias = 0.0;

        weightsGradient = new double[depth][height][width];
        biasGradient = 0.0;

        initializeWeights(depth, height, width);
    }

    public void updateParameters(double learningRate) {
        bias -= learningRate * biasGradient;
        biasGradient = 0.0;

        for (int d = 0; d < weights.length; d++) {
            for (int h = 0; h < weights[d].length; h++) {
                for (int w = 0; w < weights[d][h].length; w++) {
                    weights[d][h][w] -= learningRate * weightsGradient[d][h][w];
                    weightsGradient[d][h][w] = 0.0;
                }
            }
        }
    }

    private void initializeWeights(int depth, int height, int width) {
        double scale = Math.sqrt(2.0 / (depth * height * width));

        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    weights[d][h][w] = (Math.random() * 2 - 1) * scale;
                    weightsGradient[d][h][w] = 0.0;
                }
            }
        }
    }

    public double[][][] getWeights() {
        return weights;
    }

    public double getBias() {
        return bias;
    }
}
