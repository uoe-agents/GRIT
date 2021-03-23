# GRIT
Implementation for the paper:

[GRIT: Verifiable Goal Recognition for Autonomous Driving using Decision Trees](https://arxiv.org/abs/2103.06113)

#Setup
Make sure you are using Python 3.6 or later.

Install Lanelet2 following the instructions [here](https://github.com/fzi-forschungszentrum-informatik/Lanelet2).

Clone this repository:
```
git clone https://github.com/uoe-agents/GRIT.git
```
Install with pip:
```
cd GRIT
pip install -e .
```

Extract the [inD](https://www.ind-dataset.com/) and [rounD](https://www.round-dataset.com/) datasets into the `GRIT/data` directory.

Apply patches to the lanelet2 maps:

```
cd lanelet_map_patches
python patch_lanelet_maps.py
```

Preprocess the data and Extract features:

```
cd ../core
python data_processing.py
```

Train the decision trees:

```
cd ../decisiontree
python train_decision_tree.py
```

Calculate evaluation metrics on the test set:

```
cd ../evaluation/
python evaluate_models_from_features.py
```

Show animation of the dataset along with inferred goal probabilities:

```
python run_track_visualization.py --scenario heckstrasse --goal_recogniser trained_trees
```