## Overview
This tutorial provides an efficient guide to configuring your Python environment for multi-agent reinforcement learning (MARL) experiment. We recommend [Anaconda](https://www.anaconda.com/) for its robust package management capabilities.

For installing Anaconda on different platforms:
- Windows: Refer to [Windows Installation Guide](https://docs.anaconda.com/free/anaconda/install/windows/).
- MAC: Refer to [MAC Installation Guide](https://docs.anaconda.com/free/anaconda/install/mac-os/).
- Linux: Refer to [Linux Installation Guide](https://docs.anaconda.com/free/anaconda/install/linux/).

After installing Anaconda, you can get started with conda by referring to the [guide](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#starting-conda).

## Step 1, install python requirement:
### Create New "MARL" Environment:
macOS/Linux/Windows (using Anaconda Prompt)
- ``conda create -y -n MARL python=2.7``
### Activate the "MARL" Environment:
- ``conda activate MARL``
### Install Dependencies:
- ``cd xxx/MARL``
- ``pip install -r requirement.txt``

    For Windows only:
    
     ``pip install tensorflow-1.5.0-cp27-cp27m-win_amd64.whl``

## Step 2, run the experiment:
- ``python main.py``
    - Additional customization parameters (the utilization of argparse refer to the [guid](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwipxLjZouOCAxX6EEQIHSCGC4wQFnoECBcQAQ&url=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Fargparse.html&usg=AOvVaw1fTZQF8-wScb7NRlIUGaMF&opi=89978449)): 
        - ``--MARLAlgorithm``, choice in 'IQL', 'VDN' (to be completed) and 'QMIX',
        - ``--agents-number``, an int number, which represents the number of agents in experiment,
        - For other parameters, please refer to ``main.py``.
    - Example
        - ``python main.py --agents-number 3 --MARLAlgorithm IQL``, where you set three agents and train them with IQL algorithm.
