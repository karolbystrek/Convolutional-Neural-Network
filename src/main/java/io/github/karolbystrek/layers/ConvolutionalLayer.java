package io.github.karolbystrek.layers;

import io.github.karolbystrek.core.Kernel;
import io.github.karolbystrek.core.Tensor;

import java.util.stream.IntStream;

public class ConvolutionalLayer implements Layer {
    private final Kernel[] kernels;
    private final int stride;
    private final int padding;

    private Tensor lastInput;
    private Tensor lastWeightedInput;

    public ConvolutionalLayer(Kernel[] kernels, int stride, int padding) {
        this.kernels = kernels;
        this.stride = stride;
        this.padding = padding;
    }

    public ConvolutionalLayer(int numKernels, int kernelDepth, int kernelSize, int stride, int padding) {
        this.kernels = new Kernel[numKernels];
        this.stride = stride;
        this.padding = padding;

        for (int k = 0; k < kernels.length; k++) {
            this.kernels[k] = new Kernel(kernelDepth, kernelSize, kernelSize);
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        this.lastInput = input;

        int inputDepth = input.getDepth();
        int inputHeight = input.getHeight();
        int inputWidth = input.getWidth();
        int numKernels = kernels.length;

        int kernelHeight = kernels[0].getWeights()[0].length;
        int kernelWidth = kernels[0].getWeights()[0][0].length;

        int outputHeight = (inputHeight + 2 * padding - kernelHeight) / stride + 1;
        int outputWidth = (inputWidth + 2 * padding - kernelWidth) / stride + 1;

        lastWeightedInput = new Tensor(numKernels, outputHeight, outputWidth);
        Tensor output = new Tensor(numKernels, outputHeight, outputWidth);

        IntStream.range(0, numKernels).parallel().forEach( k -> {
            Kernel kernel = kernels[k];
            double[][][] kernelWeights = kernel.getWeights();
            double bias = kernel.getBias();

            for (int outY = 0; outY < outputHeight; outY++) {
                for (int outX = 0; outX < outputWidth; outX++) {
                    double sum = 0.0;
                    for (int d = 0; d < inputDepth; d++) {
                        for (int kY = 0; kY < kernelHeight; kY++) {
                            for (int kX = 0; kX < kernelWidth; kX++) {
                                int inY = outY * stride - padding + kY;
                                int inX = outX * stride - padding + kX;
                                if (inY >= 0 && inY < inputHeight && inX >= 0 && inX < inputWidth) {
                                    sum += input.getValue(d, inY, inX) * kernelWeights[d][kY][kX];
                                }

                            }
                        }
                    }
                    sum += bias;
                    lastWeightedInput.setValue(k, outY, outX, sum);
                    double activated = activation(sum);
                    output.setValue(k, outY, outX, activated);
                }
            }
        });

        return output;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        int inputDepth = lastInput.getDepth();
        int inputHeight = lastInput.getHeight();
        int inputWidth = lastInput.getWidth();

        int numKernels = kernels.length;
        int kernelHeight = kernels[0].getWeights()[0].length;
        int kernelWidth = kernels[0].getWeights()[0][0].length;

        int outputHeight = gradOutput.getHeight();
        int outputWidth = gradOutput.getWidth();

        Tensor gradInput = new Tensor(inputDepth, inputHeight, inputWidth);

        double[][][] gradOutData = gradOutput.getData();
        double[][][] lastInputData = lastInput.getData();
        double[][][] lastWeightedInputData = lastWeightedInput.getData();

        for (int k = 0; k < numKernels; k++) {
            Kernel kernel = kernels[k];
            double[][][] kernelWeights = kernel.getWeights();

            for (int outY = 0; outY < outputHeight; outY++) {
                for (int outX = 0; outX < outputWidth; outX++) {
                    double dActivation = (lastWeightedInputData[k][outY][outX] > 0) ? 1.0 : 0.0;
                    double delta = gradOutData[k][outY][outX] * dActivation;

                    kernel.biasGradient += delta;

                    for (int d = 0; d < inputDepth; d++) {
                        for (int kY = 0; kY < kernelHeight; kY++) {
                            for (int kX = 0; kX < kernelWidth; kX++) {
                                int inY = outY * stride - padding + kY;
                                int inX = outX * stride - padding + kX;

                                if (inY >= 0 && inY < inputHeight && inX >= 0 && inX < inputWidth) {
                                    double inputValue = lastInputData[d][inY][inX];

                                    kernel.weightsGradient[d][kY][kX] += inputValue * delta;

                                    double previousGrad = gradInput.getValue(d, inY, inX);
                                    gradInput.setValue(d, inY, inX, previousGrad + kernelWeights[d][kY][kX] * delta);
                                }
                            }
                        }
                    }
                }
            }
        }

        return gradInput;
    }

    @Override
    public void updateParameters(double learningRate) {
        for (Kernel kernel : kernels) {
            kernel.updateParameters(learningRate);
        }
    }

    private double activation(double input) {
        return Math.max(0.0, input);
    }
}
