# Quantum Natural Language Processing (QNLP) with lambeq (Quantinuum, Inc.)

This repository is the submission entry for the "Quantum Natural Language Processing (QNLP) with lambeq" challenge provided by Quantinuum, Inc. during the first Womanium Quantum Hackathon 2022. The details for the team are as follows:

Team name: neiljdo
Team members: Neil John D. Ortega (discord_id, github_id, neiljohn.ortega@gmail.com)
Presenter: Neil John D. Ortega
Name of the challenge: Quantum Natural Language Processing with lambeq - Quantinuum

## Available files

### `QNLP.ipynb`

This notebook contains exploratory work done by the team to explore the challenge problem and, more importantly, to explore the lambeq library to be able to perform the challenge task(s).

### `QNLP Experiments.ipynb`

This notebook contains the experiment runner function that combines the learnings from the above notebook into a single function that can be parametrized to run different experiments. For asynchronous inspection of the results, the experiment runner writes its (partial) results into a `*.json` file that can be analyzed on its own. This is to maximize efficiency as the experiment trials can take same time, even using the `jax`-optimized `NumpyModel` class.

### `QNLP Experiment Results.ipynb`

This notebook contains the logic for processing the results from the experiment runs so that they can be rendered as plots. The main results we display are the following:

1. Learning curve i.e. train/validation loss vs epochs
2. Metrics curve i.e. train/validation accuracy vs epochs

## Submission Overview

### Problem Statement

Input data: 2 sentences, each sentence can be classified as exclusively either about "IT" or "Food".
Output: Output "1" if the two sentences are of the same topic, i.e. both about "IT" or both about "Food". Output "0", otherwise.

Can we use QNLP for this task?

### Solution & Approach

We implemented the following extensions to the existing `lambeq` functionality to be able to perform the task:

#### Conversion of labels

We have the following mapping for the labels:

* "1" gets mapped to the vector [1, 0]
* "0" gets mapped to the vector [0, 1]

We explain the rationale for this mapping in the next session as this is opposite the convention for the basis states $|0>$ and $|1>$.

#### Modify the forward pass to `QuantumModel` subclasses:

The custom forward pass takes in a pair of diagrams and does the following:

```
class CustomNumpyModel(NumpyModel):
    def forward(self, x: list[[Diagram, Diagram]]) -> np.ndarray:
        # The forward pass takes x with 2 circuits
        # for each of the sentence being compared
        s1_diagrams = []
        s2_diagrams = []
        n_rows = len(x)
        for s1d, s2d in x:
            s1_diagrams.append(s1d)
            s2_diagrams.append(s2d)
        
        s1_output = self.get_diagram_output(s1_diagrams)
        s2_output = self.get_diagram_output(s2_diagrams)
        s1_output = s1_output.reshape((n_rows, -1))[:,:2]
        s2_output = s2_output.reshape((n_rows, -1))[:,:2]
        
        s1_output_norm = np.sqrt(np.sum(s1_output * s1_output, axis=1))
        s2_output_norm = np.sqrt(np.sum(s2_output * s2_output, axis=1))
        denom = s1_output_norm * s2_output_norm
        s1_dot_s2 = np.sum(s1_output[:,:2] * s2_output[:,:2], axis=1) / denom

        complement = np.ones_like(s1_dot_s2) - s1_dot_s2
        out = np.array([s1_dot_s2,
                        complement]).T

        return out
```

#### Implement a different word ansatz

#### Implement an experiment runner

### Implementation & Scaling

* Limitations
* Full experiment setup
    * 5-Fold validation due to small dataset size, repeated 4 times for robustness/variance mitigation
    * 2 ansatze tried - IQPAnsatz and the custom one from <insert paper here>
    * Several ansatz parameter configuration for each of the 2 ansatze:
        * qn := # of qubits for nouns, either 1 or 2
        * qs := # of qubits for sentences, either 1 or 2
        * pn := # of parameters in word ansatz, either 1 or 3
        * d := depth of word ansatz, either 1 or 2
