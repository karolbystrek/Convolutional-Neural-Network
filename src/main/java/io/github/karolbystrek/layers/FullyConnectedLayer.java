package io.github.karolbystrek.layers;

import io.github.karolbystrek.core.Tensor;

public class FullyConnectedLayer implements Layer {

    private final Layer[] layers;

    public FullyConnectedLayer(int[] layerSizes) {
        if (layerSizes == null || layerSizes.length < 2) {
            throw new IllegalArgumentException("Fully connected layer requires at least two layers");
        }

        layers = new Layer[layerSizes.length - 1];

        int layerIndex = 0;
        for (; layerIndex < layers.length - 1; layerIndex++) {
            layers[layerIndex] = new HiddenLayer(layerSizes[layerIndex], layerSizes[layerIndex + 1]);
        }
        layers[layerIndex] = new OutputLayer(layerSizes[layerIndex], layerSizes[layerIndex + 1]);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor output = input;
        for (Layer layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    @Override
    public Tensor backward(Tensor gradientOutput) {
        Tensor grad = gradientOutput;
        for (int layerIndex = layers.length - 1; layerIndex >= 0; layerIndex--) {
            grad = layers[layerIndex].backward(grad);
        }
        return grad;
    }

    @Override
    public void updateParameters(double learningRate) {
        for (Layer layer : layers) {
            layer.updateParameters(learningRate);
        }
    }

}
