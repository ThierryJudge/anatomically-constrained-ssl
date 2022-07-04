# Anatomically Constrained Semi-supervised Learning

This project also uses the [vital submodule](https://github.com/ThierryJudge/vital).

## How to run
First, install dependencies
```bash
git clone --recurse-submodules git@github.com:ThierryJudge/anatomically-constrained-ssl.git

# install echo-segmentation-uncertainty
cd anatomically-constrained-ssl
conda env create -f requirements/environment.yml
 ```
You are now ready to import modules and run code.

If the submodule is not loaded correctly run 
```bash
git submodule init
git submodule update
 ```
You can also change the submodule branch.

### Important dependencies
These are the main dependencies for this project. All dependencies are listed in the [requirements file](requirements/requirements.txt)
* [Hydra](https://hydra.cc/) To handle configurations and CLI
* [Pytorch](https://pytorch.org/) and [PyTorch Lightning](https://www.pytorchlightning.ai/) To handle training and tesing 
* [dotenv](https://pypi.org/project/python-dotenv/) to handle environment variables.  
* [Comet.ml](https://www.comet.ml/) to experiment tracking.  

#### Hydra
This project uses [Hydra](https://hydra.cc/) to handle configurations and CLI. Uses a hierarchy of configuration files. 
This hierarchy is located the [config directory](config). Hydra has many features that help to launch code such as:  
* You can edit and existing parameter with by adding   ```param=32```.
* You can add an non-existing parameter with ```+param=32```
* Resolvers allow defining more complex parameters at runtime such as accessing environment variables (```param=${oc.env:ENV_VARAIBLE}```)

#### Pytorch Lightning
Pytorch Lightning is a library built on top of Pytorch to simplify training. The library is 

* Datamodule: This class handles loading the dataset and generating the 3 dataloaders (train, validation, test)
* LightningModule: This class defines the model and the defines the training, validation and test step. 
* Trainer: The pytorch-lightning trainer handles the training loop given a system and datamodule


### Vital Project structure 
The vital submodule builds upon Pytorch Lightning to avoid boiler plate code. Two classes of 
* [VitalSystem](vital/vital/systems/system.py): This class inherits from pytorch_lighting.LightningModule and is the base for all systems.
* [VitalRunner](vital/vital/runner.py): This class loads the hydra configuration, instantiates the Trainer, Datamodule and System and handles the training and testing.


### Environment setup

To setup the environment, copy the `.env.example` file and name it `.env`. Add the paths to the relevant datasets.

### Training and testing
All training and evaluation is ran through the [runner.py](runner.py) script. This script reads the 
[default configuration file](config/default.yaml) and loads the appropriate datamodule, system and trainer. 

All options in the configuration can be modified. To see all the possible options:
```bash
python runner.py -h 
 ```
The defaults are the Camus dataset, the enet network and segmentation system. 

A few example running commands:
```bash
# Train baseline
python runner.py data.label_split=0.5

# Train semi supervised 
python runner.py task=acssl data.label_split=0.5 weights=<path/to/weights>
 ```


### Contribution 

#### Adding a dataset 
To add a new dataset, the following must be done: 
1. Create a new directory in [vital/vital/data](vital/vital/data) or [echo_segmentation_uncertainty/data](echo_segmentation_uncertainty/data)
2. Create the dataset class.
3. Add a config files with the necessary data structure (optional)   
4. Create the datamodule class
5. Add the corresponding data configuration to [vital/vital/config/data](vital/vital/config/data).
6. Add the dataset path to the [env example file](.env.example) and your [env file]((.env))