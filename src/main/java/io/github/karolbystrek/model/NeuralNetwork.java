package io.github.karolbystrek.model;

import io.github.karolbystrek.layers.Layer;
import io.github.karolbystrek.core.Tensor;
import io.github.karolbystrek.reader.DataPoint;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class NeuralNetwork {
    private final List<Layer> layers = new ArrayList<>();

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    public Tensor predict(DataPoint dataPoint) {
        return forward(dataPoint.getInput());
    }

    public Tensor forward(Tensor input) {
        Tensor output = input;
        for (Layer layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    public void backward(Tensor output) {
        for (int i = layers.size() - 1; i >= 0; i--) {
            output = layers.get(i).backward(output);
        }
    }

    public void updateParameters(double learningRate) {
        for (Layer layer : layers) {
            layer.updateParameters(learningRate);
        }
    }

    public double train(List<? extends DataPoint> trainingData, int batchSize, double learningRate) {
        double totalCost = 0.0;
        int batchIndex = 0;

        for (DataPoint dataPoint : trainingData) {
            Tensor output = forward(dataPoint.getInput());
            Tensor expectedOutput = dataPoint.getExpectedOutput();

            totalCost += cost(output, expectedOutput);
            Tensor gradOutput = calculateOutputGradient(output, expectedOutput);

            backward(gradOutput);

            batchIndex++;
            if (batchIndex >= batchSize) {
                updateParameters(learningRate);
                batchIndex = 0;
            }
        }

        if (batchIndex > 0) {
            updateParameters(learningRate);
        }

        return totalCost / trainingData.size();
    }

    public void fit(List<? extends DataPoint> trainingData, int maxEpochs, int batchSize, double learningRate) {
        System.out.println("Beginning training...");

        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            long startTime = System.nanoTime();
            System.out.print("Epoch: " + (epoch + 1) + ", ");

            Collections.shuffle(trainingData);
            double averageCost = train(trainingData, batchSize, learningRate);
            System.out.print("Average cost: " + averageCost + ", ");

            long endTime = System.nanoTime();
            System.out.println("Total execution time: " + (endTime - startTime) / 1.0e9 + "s");
        }
    }

    private double cost(Tensor output, Tensor expectedOutput) {
        double cost = 0.0;
        double EPSILON = 1.0e-15;

        for (int w = 0; w < output.getWidth(); w++) {
            double value = output.getValue(0, 0, w);
            double expectedValue = expectedOutput.getValue(0, 0, w);

            cost -= expectedValue * Math.log(value + EPSILON);
        }

        return cost;
    }

    private Tensor calculateOutputGradient(Tensor output, Tensor expectedOutput) {
        Tensor gradOutput = new Tensor(1, 1, output.getWidth());

        for (int w = 0; w < output.getWidth(); w++) {
            double value = output.getValue(0, 0, w);
            double expectedValue = expectedOutput.getValue(0, 0, w);
            gradOutput.setValue(0, 0, w, value - expectedValue);
        }

        return gradOutput;
    }
}
