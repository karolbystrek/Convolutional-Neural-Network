package io.github.karolbystrek.reader;

import io.github.karolbystrek.core.Tensor;

public interface DataPoint {

    Tensor getInput();

    Tensor getExpectedOutput();
}
