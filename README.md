# Neural Feature Combiner

This project was developed by [Cesare Bernardis](https://scholar.google.it/citations?user=9fzJj_AAAAAJ), 
Ph.D. candidate at Politecnico di Milano, and it is based on the recommendation framework used by our [research group](http://recsys.deib.polimi.it/).
The code allows reproducing the results of "NFC: a Deep and Hybrid Item-based Model for Item Cold-Start Recommendation", 
published on the UMUAI Special Issue on Dynamic Recommender Systems and User Models.

Please cite our article (coming soon!) if you use this repository or a part of it in your work.


## Installation

---

To run the experiments, we suggest creating an ad hoc virtual environment using [Anaconda](https://www.anaconda.com/).
 
_You can also use other types of virtual environments (e.g. virtualenv). 
Just ensure to install the packages listed in 'requirements.txt' in the environment used to run the experiments._

To install Anaconda you can follow the instructions on the [website](https://www.anaconda.com/products/individual).


To create the Anaconda environment with all the required python packages, run the following command:

```console
bash create_env.sh
```

This script also installs other dependencies and libraries that are not included in the _requirements.txt_ file (e.g., xlearn, tensorflow-gpu, similaripy, deeplift), and extracts the model hyperparameters from the _data.zip_ archive.
If you are not using the script to install the environment, please be sure to follow the steps it performs for the proper functioning of the framework.

The default name for the environment is _recsys-nfc_.
You can change it by replacing the string _recsys-nfc_ in 'create_env.sh' with the name you prefer.
All the following commands have to be modified according to the new name.

Note that this framework includes some [Cython](https://cython.readthedocs.io/en/latest/index.html) implementations that need to be compiled.
Before continuing with next steps, ensure to have a C/C++ compiler installed.
On several Linux distributions (e.g. Ubuntu, Debian) it is enough to run the following:

```console
sudo apt-get install build-essential
```

Finally, to install the framework you can simply activate the created virtual environment and execute the 'install.sh' script, using the following commands:

```console
conda activate recsys-nfc
sh install.sh
```

## Run the experiments

---

First of all, you should activate the environment where you installed the framework.
If you followed the installation guide above, the command is:

```console
conda activate recsys-nfc
```

Then, the first step to run the experiments consists in downloading the data and generating the splits.
These actions are performed by the 'create_splits.py' script that can be executed with the following command:

```console
python create_splits.py
```

Note that Amazon Video Games, BookCrossing and Movielens are publicly available and are automatically downloaded by the framework.
YahooMovies, instead, has to be manually downloaded, since it is not publicly available and requires a (free) subscription to access the datasets.

###Top-n recommendation

To perform the evaluation on the splits created, run:

```console
python evaluate_on_test.py --cold #runs the experiments in the cold-start scenario
python evaluate_on_test.py --rampup #runs the experiments in the ramp-up scenario
python evaluate_on_test.py --hybrid #runs the experiments in the cold-warm hybrid scenario
```

It is also possible to specify multiple arguments at the same time (among "--cold", "--rampup", and "--hybrid"). 
The script runs all the required experiments.

Automatically, the script employs the best hyper-parameter configurations found by the optimization procedure we performed in our experiments.

If you want to perform a new hyper-parameter optimization, you can use:

```console
python hyperparameter_optimization.py
```

The script takes a very long time to run.
For this reason, we provide the optimal configurations found during our experiments, so you don't have to run this script again.
If you want to rerun everything from scratch, open the script and set the "resume_from_saved" variable to False (line 72).

By default, the repository runs the experiments only on Amazon Video Games.
You can run the experiments on all the datasets by uncommenting lines 41-70 in file RecSysFramework/ExperimentalConfig.py

###Qualitative analysis

If you want to run the qualitative study on YahooMovies, run the following command: 
```console
python test_Embeddings.py"
```
If you want to run the qualitative study on YahooMovies without synopsis, run the following command:
```console
python test_Embeddings.py --nosynopsis"
```
If you want to run the qualitative study on YahooMovies in a cold scenario, replace "test_Embeddings.py" with "test_Embeddings_cold.py" in the previous two commands.

## IMPORTANT!
In order to find the data, the experiments have to be run from the main folder of the project (i.e. the folder that contains this file).
