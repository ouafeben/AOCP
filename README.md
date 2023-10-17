# AOCP
Step 1: MuZero Environment Configuration

Using the Python installed on the system.
The MuZero source code was downloaded from the DeepMind GitHub repository: GitHub - werner-duvaud/muzero-general: MuZero
We configured our development environment by following the instructions provided in the MuZero documentation.
To obtain the evaluation of MuZero's performance metrics, run
tensorboard dev upload --logdir C:\Users\hp\muzero-general\results\these14\2023-05-21--21-26-57 --name " My latest experiment"   
for 72 nodes
tensorboard dev upload --logdir C:\Users\hp\muzero-general\results\these14 -72nodes\2023-10-11--21-17-13--name " My latest experiment72nodes"   

Step 2: Define the Game
We modeled our game to train MuZero.
We implemented the game rules in the format required by MuZero.

Step 3: Training MuZero
We used MuZero to train our model. Training MuZero involves having the algorithm play our game, collecting training data from these games.
MuZero will learn both how to play the game and how to model the underlying rules.
