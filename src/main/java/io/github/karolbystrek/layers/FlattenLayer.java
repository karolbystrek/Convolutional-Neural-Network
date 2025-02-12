package io.github.karolbystrek.layers;

import io.github.karolbystrek.core.Tensor;

public class FlattenLayer implements Layer {

    private int originalDepth;
    private int originalHeight;
    private int originalWidth;

    @Override
    public Tensor forward(Tensor input) {
        this.originalDepth = input.getDepth();
        this.originalHeight = input.getHeight();
        this.originalWidth = input.getWidth();

        double[][][] data = input.getData();
        double[][][] flatData = new double[1][1][originalDepth * originalHeight * originalWidth];

        int index = 0;
        for (int d = 0; d < originalDepth; d++) {
            for (int h = 0; h < originalHeight; h++) {
                for (int w = 0; w < originalWidth; w++) {
                    flatData[0][0][index++] = data[d][h][w];
                }
            }
        }
        return new Tensor(flatData);
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        double[][][] gradData = gradOutput.getData();
        double[][][] unflatData = new double[originalDepth][originalHeight][originalWidth];

        int index = 0;
        for (int d = 0; d < originalDepth; d++) {
            for (int h = 0; h < originalHeight; h++) {
                for (int w = 0; w < originalWidth; w++) {
                    unflatData[d][h][w] = gradData[0][0][index++];
                }
            }
        }
        return new Tensor(unflatData);
    }

    @Override
    public void updateParameters(double learningRate) {}

}
