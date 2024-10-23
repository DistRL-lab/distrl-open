# 🤖 DistRL Setup Guide


This is the code repo for Paper **DistRL: An Asynchronous Distributed Reinforcement Learning Framework for On-Device Control Agents**

Paper link [https://arxiv.org/pdf/2410.14803](https://arxiv.org/pdf/2410.14803)

We will release the Model Weights and Dataset later.

## 🍩 Features

### Framework Capabilities

- **Flexible Integration of Agents and Models**
- **Scalable Data Collection Tools**
- **Efficient Online Training with Multi-Machine Environment Support** (utilizing heterogeneous workers and GPUs)

### Methodology Highlights

- **Supported Training Module** as detailed in the paper:
  - **DistRL**: An efficient reinforcement learning algorithm tailored for distributed environments.
  - **Baseline**: **DigiRL** (features automatic curriculum learning with doubly robust estimator filtering).
  - **Baseline**: **Filtered Behavior Cloning** (employs reward-based filtering).

- **Agent Support**:
  - **[AutoUI](https://arxiv.org/abs/2309.11436)**: Comprehensive support for both training and evaluation phases.

- **Android-in-the-Wild Task Sets**:
  - **AitW General**: Tasks involving general browsing and app launching.
  - **AitW Web Shopping**: Tasks centered around shopping on popular e-commerce websites.
  - **Additional Evaluations**: We've assessed generalization capabilities on other [AitW subsets](https://github.com/google-research/google-research/tree/master/android_in_the_wild), such as App Install, although the training environments for these were not meticulously configured.

- **DDP Multi-GPU Training Support**:
  - Multi-GPU training is facilitated through `accelerate`. If you're operating with a single GPU, this feature can be disabled. Running AutoUI with the DistRL algorithm requires only **15GB** of GPU memory. This support is provided should you wish to experiment with more extensive setups.





## ✅ Quick Start

### A. Dependencies

Please check the `requirements.txt` file for all necessary dependencies.

### B. Before You Start

1. **Create Necessary Directories**: Set up the required directories as specified in the configuration `.yaml` files (e.g., `Tmp` path, `agg_traj` path, `save_path`, etc.).

2. **Update Tokens**: Replace placeholders with your actual tokens in the configuration files in `scripts/config`:
   - `huggingface_token`
   - `wandb_token`
   - `gemini_token`
   - 	... etc.

3. **Review and Enhance Prompts**:  Clear and well-structured prompts are essential for improving the evaluator's performance in assessing task completion. By crafting precise and detailed prompts, we can guide the model to produce more accurate and reliable evaluations. We have provided demonstration examples in `data/environment/android/prompts.txt` for your reference. Please do adjust `data/environment/android/evaluate.py` based on our hints and comments.

   
### C. Android Environment Setup

To set up the Android environment for the DistRL to interact with, refer to [the environment](./env_setup.md). Before moving on, you should be able to view [this script](./scripts/screenshot.py).

### D. Running Experiments

#### D.1 Weights & Biases (Wandb) Setup

[Weights & Biases](https://wandb.ai/site) is a tool for tracking machine learning experiments. To integrate Wandb into our framework:

- **Create an Account**: If you don't already have one, sign up for a free Wandb account.
- **Install Wandb**: Ensure Wandb is installed by running `pip install wandb`.
- **Login**: Authenticate your Wandb account by running `wandb login` and entering your API key when prompted.
- **Configure Wandb in the Framework**: Update your `wandb_token` in the configuration files with your API key.

For more detailed instructions, refer to the [Wandb Quickstart Guide](https://docs.wandb.ai/quickstart).

#### D.2 Entry Point of the Framework

The main entry point of the program is the `run.py` script. You can specify different experiments by passing the configuration name. Configuration files are located in the `scripts/config/` directory.

Setup Steps:

1. Set Up Conda Environment:
   - Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
   - Create a new environment named **distrl**:
   
     ```bash
     conda create -n distrl python=3.8
     conda activate distrl
     ```

2. Clone the Repository:
   - Clone the repository and check out the master branch:

     ```bash
     git clone <repository_url>
     cd <repository_directory>
     git checkout master
     ```

   - Install the package:

     ```bash
     pip install -e .
     ```

3. Set Up the Environment:
   - Follow the [Environment Setup Guide](./env_setup.md) to configure the environment for Android emulator.

4. Test the Setup:
   - Run the `run.py` script with the worker configuration to test:

    ```bash
    CUDA_VISIBLE_DEVICES=0 python scripts/run.py --config-path config/multimachine --config-name worker +thread_id=0
    ```
5. Run from host machine:

    <!-- - Multi GPU: -->

    ```bash
    accelerate launch --config_file config/accelerate_config/default_config.yaml scripts/run.py --config-path config/multimachine --config-name host
    ```



