package io.github.karolbystrek.core;

public class Tensor {

    private final double[][][] data;
    private final int depth;
    private final int height;
    private final int width;

    public Tensor(int depth, int height, int width) {
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.data = new double[depth][height][width];
    }

    public Tensor(double[][][] data) {
        this.depth = data.length;
        this.height = data[0].length;
        this.width = data[0][0].length;
        this.data = data;
    }

    public int getDepth() {
        return depth;
    }

    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }

    public double[][][] getData() {
        return data;
    }

    public double getValue(int d, int h, int w) {
        return data[d][h][w];
    }

    public void setValue(int d, int h, int w, double value) {
        data[d][h][w] = value;
    }
}
