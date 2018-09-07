# Cross-Entropy metaheuristic method for Reinforcement Learning (Pytorch)

This project includes the code for a reinforcement learning agent that learns via cross entropy metaheuristic method. The agent can use multiprocessing to run multiple workers for parallel evaluations.

For more information on the implementation refer to "Cross-Entropy method for finding the weights in a Reinforcement Learning Problem (Pytorch Version).ipynb". The notebook includes a summary of all essential concepts used in the code. It also contains two examples where the algorithm is used to solve Open AI gym environments.

### Examples


[//]: # (Image References)

[image1]: https://raw.githubusercontent.com/cpow-89/Cross-Entropy-metaheuristic-method-for-Reinforcement-Learning-Pytorch-/master/images/MountainCarContinuous-v0.gif "Trained Agents1"

#### MountainCarContinuous-v0

![Trained Agents1][image1]

[image2]: https://raw.githubusercontent.com/cpow-89/Cross-Entropy-metaheuristic-method-for-Reinforcement-Learning-Pytorch-/master/images/CartPole-v1.gif "Trained Agents2"

#### CartPole_v1

![Trained Agents2][image2]

### Dependencies

1. Create (and activate) a new environment with Python 3.6.

> conda create --name env_name python=3.6<br>
> source activate env_name

2. Install OpenAi Gym

> git clone https://github.com/openai/gym.git<br>
> cd gym<br>
> pip install -e .<br>
> pip install -e '.[box2d]'<br>
> pip install -e '.[classic_control]'<br>
> sudo apt-get install ffmpeg<br>

3. Install Sourcecode dependencies

> conda install -c rpi matplotlib <br>
> conda install -c pytorch pytorch <br>
> conda install -c anaconda numpy <br>

### Instructions

You can run the project via Extended_Deep_Q_Learning_for_Multilayer_Perceptron.ipynb or running the main.py file through the console.



open the console and run: python main.py -c "your_config_file".json 
optional arguments:

-h, --help

    - show help message
    
-c , --config

    - Config file name - file must be available as .json in ./configs
    
Example: python main.py -c "MountainCarContinuous_v0.json" 

