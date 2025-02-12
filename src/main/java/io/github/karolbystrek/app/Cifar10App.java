package io.github.karolbystrek.app;

import io.github.karolbystrek.core.Kernel;
import io.github.karolbystrek.core.Tensor;
import io.github.karolbystrek.layers.ConvolutionalLayer;
import io.github.karolbystrek.layers.FlattenLayer;
import io.github.karolbystrek.layers.FullyConnectedLayer;
import io.github.karolbystrek.layers.PoolingLayer;
import io.github.karolbystrek.model.NeuralNetwork;
import io.github.karolbystrek.reader.Cifar10DataPoint;
import io.github.karolbystrek.reader.Cifar10DataReader;
import io.github.karolbystrek.reader.DataPoint;

import java.io.IOException;
import java.util.List;

public class Cifar10App {

    private static final String[] trainingFiles = {
            "data/cifar-10/data_batch_1.bin",
            "data/cifar-10/data_batch_2.bin",
            "data/cifar-10/data_batch_3.bin",
            "data/cifar-10/data_batch_4.bin",
            "data/cifar-10/data_batch_5.bin"
    };
    private static final String testFile = "data/cifar-10/test_batch.bin";

    private static final int MAX_EPOCHS = 10;
    private static final int BATCH_SIZE = 64;
    private static final double LEARNING_RATE = 0.001;

    public static void main(String[] args) {
        try {
            Cifar10DataReader dataReader = new Cifar10DataReader();
            List<Cifar10DataPoint> trainingData = dataReader.readTrainingData(trainingFiles);
            List<Cifar10DataPoint> testData = dataReader.readTestData(testFile);

            NeuralNetwork model = new NeuralNetwork();

            model.addLayer(new ConvolutionalLayer(16, 3, 3, 1, 1));
            model.addLayer(new PoolingLayer(2, 2));

            model.addLayer(new ConvolutionalLayer(16, 32, 3, 1, 1));
            model.addLayer(new PoolingLayer(2, 2));

            model.addLayer(new FlattenLayer());

            model.addLayer(new FullyConnectedLayer(new int[] {1024, 512, 10}));

            model.fit(trainingData, MAX_EPOCHS, BATCH_SIZE, LEARNING_RATE);

            evaluate(model, testData);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void evaluate(NeuralNetwork model, List<? extends DataPoint> testData) {
        int correct = 0;

        for (DataPoint dataPoint : testData) {
            Tensor outputTensor = model.predict(dataPoint);
            double[] output = outputTensor.getData()[0][0];

            int predicted = 0;
            for (int i = 1; i < output.length; i++) {
                if (output[i] > output[predicted]) {
                    predicted = i;
                }
            }

            Tensor expectedOutputTensor = dataPoint.getExpectedOutput();
            double[] expectedOutput = expectedOutputTensor.getData()[0][0];
            int expected = 0;
            for (int i = 1; i < expectedOutput.length; i++) {
                if (expectedOutput[i] > expectedOutput[expected]) {
                    expected = i;
                }
            }

            if (predicted == expected) {
                correct++;
            }
        }

        double accuracy = 100.0 * correct / testData.size();
        System.out.println("Test accuracy: " + accuracy + "%");
    }

}
