# Dual-Network Feature Selection

## Description
This repository contains the source code to reproduce the results stated in the paper "Embedded Feature Selection Using Dual-Network Architecture".

## Installation
Since we have only tested the implementation on Linux systems, we recommend using one if you intend to reproduce the results yourself. If you are using a virtual environment with pip, simply install the 'req.txt,' via

```bash
pip install -r req.txt
```

and you should be ready to go. On Windows, you may need to change the optimizer from

```python
    keras.optimizers.legacy.Adam()
```

inside `feature_selector.py` to

```python
    keras.optimizers.Adam()
```

## Usage
The file `feature_selector.py` contains the implementation based on the working principles outlined in the paper. If you intend to reproduce the results, you need to execute the files corresponding to the respective datasets. If you are using Spyder, you can directly execute the code blocks step by step.

For the Ames Housing dataset, you must specify the path to the AmesHousing.csv file.

The following code initializes the selector:

```python
# Generate Selector instance
selector = FeatureSelector(selector_nodes = [276,256,276],
               task_nodes = [276, 128, 64, 32, 1])

# Compile selector
selector.compile(factor_loss = factor_loss)

# Perform selection. 
history = selector.fit(epochs, X_train, y_train, validation_data=(X_test, y_test), verbose=2)
```

The process is similar for every dataset. For dataset-specific information, please refer to the comments inside the respective files, as you may need to change some parameters if you want to switch to non-linear datasets, for example.
