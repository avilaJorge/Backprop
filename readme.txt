RUN INSTRUCTIONS
---
To run the main neural network for classification on fashionMNIST dataset:
  1. Ensure you have python3.5+ installed.
  2. Ensure you have installed the numpy, matplotlib, and yaml python modules.
  3. Open the config.yaml file in your plaintext text editor and set the parameters you desire.
  4. Run the following command in a linux shell:

    $ python3 neuralnet.py

---
To perform the weight gradient check:
  1. Perform steps 1, 2 from above.
  2. Open the config_original.yaml file and change any setting except the number of values in the 'layer_specs' parameter list.
  3. Run the following command in a linux shell:

    $ python3 test.py
