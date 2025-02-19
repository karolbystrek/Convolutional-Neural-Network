package io.github.karolbystrek.layers;

import io.github.karolbystrek.core.Tensor;

import java.util.stream.IntStream;

public class PoolingLayer implements Layer{

    private final int poolSize;
    private final int stride;

    private Tensor lastInput;

    public PoolingLayer(int poolSize, int stride) {
        this.poolSize = poolSize;
        this.stride = stride;
    }

    @Override
    public Tensor forward(Tensor input) {
        this.lastInput = input;
        float[][][] inputData = input.getData();

        int depth = input.getDepth();
        int inputHeight = input.getHeight();
        int inputWidth = input.getWidth();

        int outputHeight = (inputHeight - poolSize) / stride + 1;
        int outputWidth = (inputWidth - poolSize) / stride + 1;

        float[][][] outputData = new float[depth][outputHeight][outputWidth];

        IntStream.range(0, depth).parallel().forEach(d -> {
            for (int outY = 0; outY < outputHeight; outY++) {
                for (int outX = 0; outX < outputWidth; outX++) {
                    float maxVal = Float.NEGATIVE_INFINITY;

                    for (int pY = 0; pY < poolSize; pY++) {
                        for (int pX = 0; pX < poolSize; pX++) {
                            int inY = outY * stride + pY;
                            int inX = outX * stride + pX;

                            if (inY < inputHeight && inX < inputWidth) {
                                float value = inputData[d][inY][inX];
                                if (value > maxVal) {
                                    maxVal = value;
                                }
                            }
                        }
                    }

                    outputData[d][outY][outX] = maxVal;
                }
            }
        });

        return new Tensor(outputData);
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        int depth = lastInput.getDepth();
        int inputHeight = lastInput.getHeight();
        int inputWidth = lastInput.getWidth();

        int outputHeight = gradOutput.getHeight();
        int outputWidth = gradOutput.getWidth();

        float[][][] gradOutputData = gradOutput.getData();
        float[][][] lastInputData = lastInput.getData();

        float[][][] gradInputData = new float[depth][inputHeight][inputWidth];

        IntStream.range(0, depth).parallel().forEach(d -> {
            for (int outY = 0; outY < outputHeight; outY++) {
                for (int outX = 0; outX < outputWidth; outX++) {
                    float maxValue = Float.NEGATIVE_INFINITY;
                    int maxInY = -1, maxInX = -1;

                    for (int pY = 0; pY < poolSize; pY++) {
                        for (int pX = 0; pX < poolSize; pX++) {
                            int inY = outY * stride + pY;
                            int inX = outX * stride + pX;

                            if (inY < inputHeight && inX < inputWidth) {
                                float value = lastInputData[d][inY][inX];
                                if (value > maxValue) {
                                    maxValue = value;
                                    maxInY = inY;
                                    maxInX = inX;
                                }
                            }
                        }
                    }

                    gradInputData[d][maxInY][maxInX] += gradOutputData[d][outY][outX];
                }
            }
        });

        return new Tensor(gradInputData);
    }

    @Override
    public void updateParameters(float learningRate) {}
}
