package io.github.karolbystrek.core;

public class Kernel {

    private final float[][][] weights;
    private float bias;

    public float[][][] weightsGradient;
    public float biasGradient;

    public Kernel(int depth, int height, int width) {
        weights = new float[depth][height][width];
        bias = 0.0f;

        weightsGradient = new float[depth][height][width];
        biasGradient = 0.0f;

        initializeWeights(depth, height, width);
    }

    public void updateParameters(float learningRate) {
        bias -= learningRate * biasGradient;
        biasGradient = 0.0f;

        for (int d = 0; d < weights.length; d++) {
            for (int h = 0; h < weights[d].length; h++) {
                for (int w = 0; w < weights[d][h].length; w++) {
                    weights[d][h][w] -= learningRate * weightsGradient[d][h][w];
                    weightsGradient[d][h][w] = 0.0f;
                }
            }
        }
    }

    private void initializeWeights(int depth, int height, int width) {
        float scale = (float) Math.sqrt(2.0 / (depth * height * width));

        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    weights[d][h][w] = (float) ((Math.random() * 2 - 1) * scale);
                    weightsGradient[d][h][w] = 0.0f;
                }
            }
        }
    }

    public float[][][] getWeights() {
        return weights;
    }

    public float getBias() {
        return bias;
    }

    public void setBias(float bias) {
        this.bias = bias;
    }
}
