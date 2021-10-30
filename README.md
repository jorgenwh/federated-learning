## Explanation
This is an (overly) simple federated deep learning simulation that simulates creating an initial model and distributing this model to several edge nodes. Then, each edge node will train the model on training data that is only known to this particular node, before sending the model back to the main server. The main server will update its own model by averaging all the parameters of the models it has received from the edge nodes. Finally, the updated model is tested and loss and accuracy is reported.

## Installing dependencies
To install all necessary dependencies to run the sim.py script, run
```bash
pip install -r requirements.txt
```
## How to run the sim.py script
The sim.py script can be ran without providing any arguments, as the necessary arguments have default settings.

The arguments accepted by the script are
```bash
python sim.py --num_edge_nodes <num_edge_nodes> --training_epochs <training_epochs> --training_lr <training_lr> --training_batch_size <training_batch_size>
```