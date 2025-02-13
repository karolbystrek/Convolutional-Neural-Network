package io.github.karolbystrek.layers;

import io.github.karolbystrek.core.Tensor;

public interface Layer {

    Tensor forward(Tensor input);

    Tensor backward(Tensor gradientOutput);

    void updateParameters(float learningRate);
}
