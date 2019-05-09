The file OK_rank_r_unbiased.py contains a tensorflow implementation of the OK algorithm for a Recurrent highway network.

The folder cp_runs contains 5 sample runs on the copy task for 2 million training steps, with batch size 16, 
learning rate 10^(-3.5), rank 16 and network size 128. Further experiments can be done by setting the experiment
variable to 'cp' in OK_rank_r_unbiased.py. Other parameters and what is printed can be changed there as well

The folder ptb_runs contains 5 sample runs on character level language modelling on the PennTreeBank dataset
for 1 million training steps, with batch size 32, learning rate 10^(-3.5), rank 8 and network size 256. Further
experiments can be done by setting the experiment variable to 'ptb' in OK_rank_r_unbiased.py. Other parameters
and what is printed can be changed there as well. For the test performance, save the model and call the file 
OK-test-ptb.py.

The files ptb.train.npy, ptb.valid.npy, ptb.test.npy are required for running tests on the PennTreeBank dataset.
The copy task data is generated in the file OK_rank_r_unbiased.py and no extra files are necessary.
