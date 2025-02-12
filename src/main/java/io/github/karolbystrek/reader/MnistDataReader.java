package io.github.karolbystrek.reader;

import io.github.karolbystrek.core.Tensor;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class MnistDataReader {

    public List<MnistDataPoint> readData(String dataFilePath, String labelFilePath) throws IOException {
        try (DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(dataFilePath)));
             DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFilePath)))) {

            if (dataInputStream.readInt() != 2051) {
                throw new IOException("Invalid MNIST image file: " + dataFilePath);
            }
            if (labelInputStream.readInt() != 2049) {
                throw new IOException("Invalid MNIST label file: " + labelFilePath);
            }

            int numDataPoints = dataInputStream.readInt();
            int numRows = dataInputStream.readInt();
            int numCols = dataInputStream.readInt();
            int numLabels = labelInputStream.readInt();

            if (numLabels != numDataPoints) {
                throw new IOException("Mismatch between image count and number of labels");
            }

            List<MnistDataPoint> data = new ArrayList<>(numDataPoints);

            for (int i = 0; i < numDataPoints; i++) {
                double[][][] imageData = new double[1][numRows][numCols];

                for (int row = 0; row < numRows; row++) {
                    for (int col = 0; col < numCols; col++) {
                        int pixelValue = dataInputStream.readUnsignedByte();
                        imageData[0][row][col] = pixelValue / 255.0;
                    }
                }

                int label = labelInputStream.readUnsignedByte();

                Tensor imageTensor = new Tensor(imageData);
                data.add(new MnistDataPoint(imageTensor, label));
            }

            return data;
        }
    }
}
