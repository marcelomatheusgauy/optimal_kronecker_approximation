



The file KTP_rank_r_approximation.py contains a tensorflow implementation of the KTP algorithm for a
Recurrent highway network.

The folder cp_runs contains 5 sample runs on the copy task for 2 million training steps, with batch size 256, 
learning rate 10^(-3.5), rank 16 and network size 128. Further experiments can be done by setting the experiment 
variable to 'cp' in OK_rank_r_unbiased.py. Other parameters and what is printed can be changed there as well


Experiments on the PennTreeBank dataset can be done by setting the experiment variable to 'ptb' in 
KTP_rank_r_approximation.py. The files ptb.train.npy, ptb.valid.npy, ptb.test.npy found in the OK folder are
required for running tests on the PennTreeBank dataset. Other parameters and what is printed can be changed
there as well.

The copy task data is generated in the file KTP_rank_r_approximation.py and no extra files are necessary.
