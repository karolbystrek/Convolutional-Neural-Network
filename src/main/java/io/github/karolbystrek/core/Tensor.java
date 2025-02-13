package io.github.karolbystrek.core;

public class Tensor {

    private final float[][][] data;
    private final int depth;
    private final int height;
    private final int width;

    public Tensor(int depth, int height, int width) {
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.data = new float[depth][height][width];
    }

    public Tensor(float[][][] data) {
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

    public float[][][] getData() {
        return data;
    }

    public float getValue(int d, int h, int w) {
        return data[d][h][w];
    }

    public void setValue(int d, int h, int w, float value) {
        data[d][h][w] = value;
    }
}
