# AOCP
Autonomous Optimization Controller Placement (AOCP) is a design of a Markov game suitable to formulate the problem of controller placement in SDN and provide a system model for it ,the model has been trained using the MuZero model-based reinforcement learning algorithm.
Firstly we worked on a initial random network wich then we used k-means to apply clustring approch which is represented by the code initial clustering.py.
Secondaly to train our model using the Muzero we followed the following steps:
Step 1: MuZero Environment Configuration

-We Used the Python version 3.10 installed on the system.
The MuZero source code was downloaded from the DeepMind GitHub repository: GitHub - werner-duvaud/muzero-general: MuZero
We configured our development environment by following the instructions provided in the MuZero documentation.
-On CMD we used the following commands: 
1-git clone https://github.com/werner-duvaud/muzero-general.git then
2-cd muzero-general
3-pip install -r requirements.lock
4-python muzero.py

Step 2: Define the Game
We modeled our game to train MuZero as mentionned in the file gameAOCP.py, we implemented the game rules in the format required by MuZero.

Step 3: Training MuZero
We used MuZero to train our model. Training MuZero involves having the algorithm play our game, collecting training data from these games.
MuZero will learn both how to play the game and how to model the underlying rules.

To obtain the evaluation of MuZero's performance metrics, we run
tensorboard dev upload --logdir C:\Users\hp\muzero-general\results\these14\2023-05-21--21-26-57 --name " My latest experiment" 
-To evaluate our AOCP system in the large network ,we created a script with python creating a large network automatically as shown in code "script creating a nodes group automatically" then we aplied k-means for clusteing presented in code "cluster72nodes"
for 72 nodes
tensorboard dev upload --logdir C:\Users\hp\muzero-general\results\these14 -72nodes\2023-10-11--21-17-13--name " My latest experiment72nodes"   
