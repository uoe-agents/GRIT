# GRIT
This repo contains the implementation of the method described 
in the paper:

["GRIT: Fast, Interpretable, and Verifiable Goal Recognition with Learned Decision Trees for Autonomous Driving"](https://arxiv.org/abs/2103.06113)
by Brewitt, et al. [1] accepted at IROS 2021

In the paper described above, GRIT was compared to another method named [IGP2](https://arxiv.org/abs/2002.02277), for which code is available here: https://github.com/uoe-agents/IGP2 [2]

# Please cite:
If you use this code, please cite
"GRIT: Fast, Interpretable, and Verifiable Goal Recognition with Learned Decision Trees for Autonomous Driving"
```
@inproceedings{brewitt2021grit,
                title={GRIT: Fast, Interpretable, and Verifiable Goal Recognition with Learned Decision Trees for Autonomous Driving},
                author={Cillian Brewitt and Balint Gyevnar and Samuel Garcin and Stefano V. Albrecht},
                booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2021)},
                year={2021}
            }
```

The files "evalutation/run_track_visualisation.py", "core/tracks_import.py",  and "core/track_visualizer.py" are based on the inD Dataset Python Tools available at https://github.com/ika-rwth-aachen/drone-dataset-tools


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

##References
[1] C Brewitt, B Gyvenar, S Garcin, SV Albrecht, "GRIT: Fast, Interpretable, and Verifiable Goal Recognition with Learned Decision Trees for Autonomous Driving", in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2021

[2] SV Albrecht, C Brewitt, J Wilhelm, B Gyvenar, F Eiras, M Dobre, S Ramamoorthy, "Integrating planning and interpretable goal recognition for autonomous driving", in Proc. of the IEEE International Conference on Robotics and Automation (ICRA), 2021
