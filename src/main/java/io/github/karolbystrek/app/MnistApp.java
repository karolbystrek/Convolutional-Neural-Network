package io.github.karolbystrek.app;

import io.github.karolbystrek.core.Kernel;
import io.github.karolbystrek.core.Tensor;
import io.github.karolbystrek.layers.ConvolutionalLayer;
import io.github.karolbystrek.layers.FlattenLayer;
import io.github.karolbystrek.layers.FullyConnectedLayer;
import io.github.karolbystrek.layers.PoolingLayer;
import io.github.karolbystrek.model.NeuralNetwork;
import io.github.karolbystrek.reader.DataPoint;
import io.github.karolbystrek.reader.MnistDataPoint;
import io.github.karolbystrek.reader.MnistDataReader;

import java.io.IOException;
import java.util.List;

public class MnistApp {
    private static final int MAX_EPOCHS = 30;
    private static final int BATCH_SIZE = 64;
    private static final double LEARNING_RATE = 0.001;

    public static void main(String[] args) {
        try {
            MnistDataReader reader = new MnistDataReader();
            List<MnistDataPoint> trainingData = reader.readData("data/mnist/train-images.idx3-ubyte", "data/mnist/train-labels.idx1-ubyte");
            List<MnistDataPoint> testData = reader.readData("data/mnist/t10k-images.idx3-ubyte", "data/mnist/t10k-labels.idx1-ubyte");

            NeuralNetwork model = new NeuralNetwork();

            Kernel kernel1 = new Kernel(1, 5, 5);
            Kernel kernel2 = new Kernel(1, 5, 5);
            Kernel kernel3 = new Kernel(1, 5, 5);
            model.addLayer(new ConvolutionalLayer(new Kernel[]{kernel1, kernel2, kernel3}, 1, 2));

            model.addLayer(new PoolingLayer(2, 2));

            Kernel kernel4 = new Kernel(3, 5, 5);
            Kernel kernel5 = new Kernel(3, 5, 5);
            Kernel kernel6 = new Kernel(3, 5, 5);
            model.addLayer(new ConvolutionalLayer(new Kernel[]{kernel4, kernel5, kernel6}, 1, 2));

            Kernel kernel7 = new Kernel(3, 3, 3);
            Kernel kernel8 = new Kernel(3, 3, 3);
            Kernel kernel9 = new Kernel(3, 3, 3);
            model.addLayer(new ConvolutionalLayer(new Kernel[]{kernel7, kernel8, kernel9}, 1, 1));

            model.addLayer(new PoolingLayer(2, 2));

            model.addLayer(new FlattenLayer());

            model.addLayer(new FullyConnectedLayer(new int[]{147, 64, 10}));

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
