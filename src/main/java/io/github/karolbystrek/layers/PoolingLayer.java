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

        int depth = input.getDepth();
        int inputHeight = input.getHeight();
        int inputWidth = input.getWidth();

        int outputHeight = (inputHeight - poolSize) / stride + 1;
        int outputWidth = (inputWidth - poolSize) / stride + 1;

        Tensor output = new Tensor(depth, outputHeight, outputWidth);

        IntStream.range(0, depth).parallel().forEach(d -> {
            for (int outY = 0; outY < outputHeight; outY++) {
                for (int outX = 0; outX < outputWidth; outX++) {
                    float maxVal = Float.NEGATIVE_INFINITY;

                    for (int pY = 0; pY < poolSize; pY++) {
                        for (int pX = 0; pX < poolSize; pX++) {
                            int inY = outY * stride + pY;
                            int inX = outX * stride + pX;

                            if (inY < inputHeight && inX < inputWidth) {
                                float value = input.getValue(d, inY, inX);
                                if (value > maxVal) {
                                    maxVal = value;
                                }
                            }
                        }
                    }

                    output.setValue(d, outY, outX, maxVal);
                }
            }
        });

        return output;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        int depth = lastInput.getDepth();
        int inputHeight = lastInput.getHeight();
        int inputWidth = lastInput.getWidth();

        int outputHeight = gradOutput.getHeight();
        int outputWidth = gradOutput.getWidth();

        Tensor gradInput = new Tensor(depth, inputHeight, inputWidth);

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
                                float value = lastInput.getValue(d, inY, inX);
                                if (value > maxValue) {
                                    maxValue = value;
                                    maxInY = inY;
                                    maxInX = inX;
                                }
                            }
                        }
                    }

                    float grad = gradOutput.getValue(d, outY, outX);
                    float currentGrad = gradInput.getValue(d, maxInY, maxInX);
                    gradInput.setValue(d, maxInY, maxInX, currentGrad + grad);
                }
            }
        });

        return gradInput;
    }

    @Override
    public void updateParameters(float learningRate) {}
}
