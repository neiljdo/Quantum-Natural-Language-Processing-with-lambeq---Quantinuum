# Quantum Natural Language Processing (QNLP) with lambeq (Quantinuum, Inc.)

This repository is the submission entry for the "Quantum Natural Language Processing (QNLP) with lambeq" challenge provided by Quantinuum, Inc. during the first Womanium Quantum Hackathon 2022 by the team:

Team name: neiljdo
Team members: Neil John D. Ortega (neiljdo#9361, [@neiljdo](https://github.com/neiljdo), neiljohn.ortega@gmail.com)
Presenter: Neil John D. Ortega
Name of the challenge: Quantum Natural Language Processing with lambeq (Quantinuum)

## Available files

1. `ntbks/QNLP Experiments.ipynb` - this notebook contains the experiment runner function that combines the learnings from the above notebook into a single function that can be parametrized to run different experiments. For asynchronous inspection of the results, the experiment runner writes its (partial) results into a JSON file that can be analyzed on its own. This is to maximize efficiency as the experiment trials can take same time, even using the `jax`-optimized `NumpyModel` class.
2. `ntbks/QNLP Experiment Results.ipynb` - this notebook contains the logic for processing the results from the experiment runs so that they can be rendered as plots. The main results we display are the learning curve, i.e. train/validation loss vs epochs, and the metrics curve, i.e. train/validation accuracy vs epochs.
3. `ntbks/QNLP.ipynb` - this notebook contains exploratory work done by the team to explore the challenge problem and, more importantly, to explore the `lambeq` library to be able to perform the challenge task(s).

## Problem Statement

For the input, we have 2 sentences, where each sentence can be classified as exclusively either about "IT" or "Food". For the output, we are asked to determine if the two sentences are about the same topic, i.e. both about "IT" or both about "Food". More specifically, the label "1" means that both sentences are about the same topic while the label "0" means otherwise.

Can we use QNLP for this task?

## Solution & Approach

We implemented the following extensions to the existing `lambeq` functionality to be able to perform the task.

### Implement a different word ansatz

In addition to the IQP Ansatz by Havlíček et al. (2019), we use the ansatz introduced by Samuel Yen-Chi Chen et al. (2019) for deep reinforcement learning. Because only the former in included in the default `lambeq` package, we implemented a `Customansatz` class extending `discopy.quantum.circuit.Circuit`. We visualize a sample 3-qubit, 2-layer word ansatz below.

<img width="594" alt="image" src="https://user-images.githubusercontent.com/1657332/186183542-86bd5005-1184-4704-8d99-2c6bfbf128f5.png">

In addition to the custom class above, we implemented a `CustomAnsatz` class extending `lambeq.CircuitAnsatz`. This class is responsible for converting diagrams into quantum circuits. We depict a sample quantum circuit that uses the custom ansatz in the image below. It corresponds to the sentence _"John cooks delicious food."_

<img width="753" alt="image" src="https://user-images.githubusercontent.com/1657332/186186313-267a8742-9c36-417f-8e3b-eab740630e5c.png">

### Experiment setup

#### Ansatz parametrization
We parametrized the ansatze with the 4-tuple (qsn, qss, pn, d), similar to the scheme by Lorenz et al. (2021). The meaning of each parameter and allowed values are as follows.

* qsn is the number of qubits for the noun atomic type, either 1 or 2
* qss is the number of qubits for the sentence atomic type, either 1 or 2
* pn is the number of parameters in the Euler decomposition, either 1 or 3
* d is the depth of the word ansatz, either 1 or 2

The values are restricted to keep the number of qubits as low as possible. For both the IQP and the custom ansatz, we try all the possible 4-tuple of parameters, resulting in 32 different model/ansatz parametrizations.

#### Dataset preparation

The entire dataset consists of 100 sentence pairs, each with a binary label. We created a held-out test dataset of 10 sentences to be used for final verification. The remaining 90 sentences were used in 4-repeated 5-fold cross validation scheme to control the variance of any of the results that we collected from the experiment. 

#### Model pipeline

We trained each parametrization of the quantum circuit model on all the 20 folds for 500 epochs each. Our pipeline closely followed the pipeline of Lorenz et al. (2021) which we show below.

<img width="285" alt="image" src="https://user-images.githubusercontent.com/1657332/186206312-a38f3508-08b0-4711-96bd-507a9bfa3f54.png">
(Lorenz et al., 2021)

We implemented the following changes and/or restrictions to the pipeline to accomodate the problem at hand:

1. Preprocess two sentences at a time instead of a single sentence
2. Used a custom `NumpyModel` class combined with `jit` for reasonable training times. This class implements a different forward pass to make sure that we get a probability vector for each sentence pair input. We used _cosine similarity_ between the circuit outputs to generate these probability vectors.
3. Fixed the optimizer type and hyperparameters
4. Logged training and validation costs, and training and validation accuracies for the entire experiment incrementally into a JSON file

We summarized the entire model pipeline into a fully parametrized and customizable `run_experiment` function - please refer to `ntbks/QNLP Experiments.ipynb` for the full implementation and usage.
