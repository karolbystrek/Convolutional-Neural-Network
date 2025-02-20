package io.github.karolbystrek.io;

import io.github.karolbystrek.core.Kernel;
import io.github.karolbystrek.layers.*;
import io.github.karolbystrek.model.NeuralNetwork;

import java.io.*;
import java.util.List;
import java.util.StringTokenizer;

public class NeuralNetworkFileIO {

    public static NeuralNetwork loadNetwork(String filePath) throws IOException {
        try (BufferedReader in = new BufferedReader(new FileReader(filePath))) {
            String line = in.readLine();
            int numLayers = Integer.parseInt(line);

            NeuralNetwork model = new NeuralNetwork();
            for (int layerIndex = 0; layerIndex < numLayers; layerIndex++) {
                line = in.readLine();

                StringTokenizer tokenizer = new StringTokenizer(line);
                String layerType = tokenizer.nextToken();

                Layer newLayer = switch (layerType) {
                    case "ConvolutionalLayer" -> loadConvolutionalLayer(in, tokenizer);
                    case "PoolingLayer" -> loadPoolingLayer(tokenizer);
                    case "FlattenLayer" -> loadFlattenLayer();
                    case "FullyConnectedLayer" -> loadFullyConnectedLayer(in, tokenizer);
                    default -> null;
                };

                model.addLayer(newLayer);
            }
            return model;
        }
    }

    public static void saveNetwork(NeuralNetwork model, String filePath) throws IOException {
        try (PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(filePath)))) {
            List<Layer> layers = model.getLayers();
            out.println(layers.size());

            for (Layer layer : layers) {
                if (layer instanceof ConvolutionalLayer) {
                    saveConvolutionalLayer(out, (ConvolutionalLayer) layer);
                } else if (layer instanceof PoolingLayer) {
                    savePoolingLayer(out, (PoolingLayer) layer);
                } else if (layer instanceof FlattenLayer) {
                    saveFlattenLayer(out);
                } else if (layer instanceof FullyConnectedLayer) {
                    saveFullyConnectedLayer(out, (FullyConnectedLayer) layer);
                }
            }
        }
    }

    private static FullyConnectedLayer loadFullyConnectedLayer(BufferedReader in, StringTokenizer tokenizer) throws IOException {
        int sizesLength = Integer.parseInt(tokenizer.nextToken());
        int[] sizes = new int[sizesLength];
        for (int i = 0; i < sizesLength; i++) {
            sizes[i] = Integer.parseInt(tokenizer.nextToken());
        }

        FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(sizes);
        Layer[] layers = fullyConnectedLayer.getLayers();
        for (Layer layer : layers) {
            float[][] weights = new float[0][0];
            float[] bias = new float[0];

            if (layer instanceof HiddenLayer hiddenLayer) {
                weights = hiddenLayer.getWeights();
                bias = hiddenLayer.getBiases();
            } else if (layer instanceof OutputLayer outputLayer) {
                weights = outputLayer.getWeights();
                bias = outputLayer.getBiases();
            }

            int numNodesOut = weights.length;
            int numNodesIn = weights[0].length;

            for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
                String line = in.readLine();
                tokenizer = new StringTokenizer(line);

                bias[nodeOut] = Float.parseFloat(tokenizer.nextToken());
                for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                    weights[nodeOut][nodeIn] = Float.parseFloat(tokenizer.nextToken());
                }
            }
        }

        return fullyConnectedLayer;
    }

    private static FlattenLayer loadFlattenLayer() {
        return new FlattenLayer();
    }

    private static PoolingLayer loadPoolingLayer(StringTokenizer tokenizer) {
        int poolSize = Integer.parseInt(tokenizer.nextToken());
        int stride = Integer.parseInt(tokenizer.nextToken());
        return new PoolingLayer(poolSize, stride);
    }

    private static ConvolutionalLayer loadConvolutionalLayer(BufferedReader in, StringTokenizer tokenizer) throws IOException {
        int numKernels = Integer.parseInt(tokenizer.nextToken());
        int kernelDepth = Integer.parseInt(tokenizer.nextToken());
        int kernelSize = Integer.parseInt(tokenizer.nextToken());
        int stride = Integer.parseInt(tokenizer.nextToken());
        int padding = Integer.parseInt(tokenizer.nextToken());

        ConvolutionalLayer convolutionalLayer = new ConvolutionalLayer(numKernels, kernelDepth, kernelSize, stride, padding);

        Kernel[] kernels = convolutionalLayer.getKernels();

        for (int k = 0; k < numKernels; k++) {
            String line = in.readLine();
            tokenizer = new StringTokenizer(line);

            Kernel kernel = kernels[k];
            float[][][] kernelWeights = kernel.getWeights();

            kernel.setBias(Float.parseFloat(tokenizer.nextToken()));
            for (int d = 0; d < kernelDepth; d++) {
                for (int h = 0; h < kernelSize; h++) {
                    for (int w = 0; w < kernelSize; w++) {
                        kernelWeights[d][h][w] = Float.parseFloat(tokenizer.nextToken());
                    }
                }
            }

        }

        return convolutionalLayer;
    }

    private static void saveFullyConnectedLayer(PrintWriter out, FullyConnectedLayer fullyConnectedLayer) {
        int[] layerSizes = fullyConnectedLayer.getLayerSizes();

        out.print("FullyConnectedLayer " + layerSizes.length);
        for (int size : layerSizes) {
            out.print(" " + size);
        }
        out.println();

        Layer[] layers = fullyConnectedLayer.getLayers();
        for (Layer layer : layers) {
            float[][] weights = new float[0][0];
            float[] biases = new float[0];

            if (layer instanceof HiddenLayer hiddenLayer) {
                weights = hiddenLayer.getWeights();
                biases = hiddenLayer.getBiases();
            } else if (layer instanceof OutputLayer outputLayer) {
                weights = outputLayer.getWeights();
                biases = outputLayer.getBiases();
            }

            int numNodesOut = weights.length;
            int numNodesIn = weights[0].length;

            StringBuilder stringBuilder = new StringBuilder();
            for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
                stringBuilder.append(biases[nodeOut]).append(" ");
                for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                    stringBuilder.append(weights[nodeOut][nodeIn]).append(" ");
                }
                out.println(stringBuilder);
                stringBuilder.setLength(0);
            }
        }
    }

    private static void saveFlattenLayer(PrintWriter out) {
        out.println("FlattenLayer");
    }

    private static void savePoolingLayer(PrintWriter out, PoolingLayer poolingLayer) {
        out.println("PoolingLayer " + poolingLayer.getPoolSize() + " " + poolingLayer.getStride());
    }

    private static void saveConvolutionalLayer(PrintWriter out, ConvolutionalLayer convolutionalLayer) {
        Kernel[] kernels = convolutionalLayer.getKernels();
        int numKernels = kernels.length;
        int kernelDepth = kernels[0].getWeights().length;
        int kernelSize = kernels[0].getWeights()[0].length;

        int stride = convolutionalLayer.getStride();
        int padding = convolutionalLayer.getPadding();

        out.println("ConvolutionalLayer " + numKernels + " " + kernelDepth + " " + kernelSize + " " + stride + " " + padding);

        for (Kernel kernel : kernels) {
            float[][][] kernelWeights = kernel.getWeights();
            StringBuilder stringBuilder = new StringBuilder();

            stringBuilder.append(kernel.getBias()).append(" ");
            for (int d = 0; d < kernelDepth; d++) {
                for (int h = 0; h < kernelSize; h++) {
                    for (int w = 0; w < kernelSize; w++) {
                        stringBuilder.append(kernelWeights[d][h][w]).append(" ");
                    }
                }
            }
            out.println(stringBuilder);
        }
    }
}
