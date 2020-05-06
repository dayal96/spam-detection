ProjectReport.pdf is the final report on the mini project.

results directory contains the results of the experiment
as output of the python script. These results are reported
in tabular form in the project report.

data directory contains the datasets used for the project.


To run the experiment use the command:
'python3 main.py'

This will print all the results to std:out.
Be wary that this will take a long time and can consume
upto 10 GB of RAM when testing neural network models.

To change which models are tested, change the CONSTRS
constant in main.py. For example, to run the experiment
with only the models for k-means clustering change

CONSTRS = CLUSTERS + FORESTS + NEURAL_NETWORKS

to

CONSTRS = CLUSTERS

then run main.py as before.
