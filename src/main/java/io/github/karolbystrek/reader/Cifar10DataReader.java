package io.github.karolbystrek.reader;

import io.github.karolbystrek.core.Tensor;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Cifar10DataReader {
    private static final int NUM_DATA_POINTS = 10000;

    private static final int NUM_CHANNELS = 3;
    private static final int IMAGE_HEIGHT = 32;
    private static final int IMAGE_WIDTH = 32;

    public List<Cifar10DataPoint> readTestData(String testFilePath) throws IOException {
        return readData(testFilePath);
    }

    public List<Cifar10DataPoint> readTrainingData(String[] trainingFilePaths) throws IOException {
        List<Cifar10DataPoint> trainingData = new ArrayList<>();
        for (String filePath : trainingFilePaths) {
            List<Cifar10DataPoint> batchData = readData(filePath);
            trainingData.addAll(batchData);
        }
        return trainingData;
    }

    private List<Cifar10DataPoint> readData(String dataFilePath) throws IOException {
        try (DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(dataFilePath)))) {

            List<Cifar10DataPoint> data = new ArrayList<>(NUM_DATA_POINTS);

            for (int i = 0; i < NUM_DATA_POINTS; i++) {
                int label = dataInputStream.readUnsignedByte();

                double[][][] imageData = new double[NUM_CHANNELS][IMAGE_HEIGHT][IMAGE_WIDTH];
                for (int c = 0 ; c < NUM_CHANNELS ; c++) {
                    for (int h = 0 ; h < IMAGE_HEIGHT ; h++) {
                        for (int w = 0 ; w < IMAGE_WIDTH ; w++) {
                            int pixelValue = dataInputStream.readUnsignedByte();
                            imageData[c][h][w] = pixelValue / 255.0;
                        }
                    }
                }
                Tensor tensor = new Tensor(imageData);
                data.add(new Cifar10DataPoint(tensor, label));
            }

            return data;
        }
    }
}
