import tensorflow as tf
import numpy as np
import time                                                                                                                 
import sys      


import scipy

                                                                                                                            
from tensorflow.python.client import timeline                                                                               

from tensorflow.python import debug as tf_debug

###################### TO MAKE SURE TENSORFLOW ONLY USES ONE GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


n = 256
n_ = 7
steps = 10 * 6000000#1000000 - 3
delay = 3
updates = []
avg_steps = 1
b_size = 32

cur_lr = 10**(-3.5)
reset_every_n = 100
experiment = "ptb" #"cp" or "ptb"



val_steps = 2
val_batch = 2


if experiment == "cp":
    vocab = 4
elif experiment == "ptb":
    vocab = 50

rank_approximation = 8

#RTRL input
G1 = tf.get_variable('G1', shape=(b_size, rank_approximation, 1, vocab + n + 1), initializer=tf.zeros_initializer())
G2 = tf.get_variable('G2', shape=(b_size, rank_approximation, n, 2 * n), initializer=tf.zeros_initializer())

#RNN input
x_ = tf.placeholder(tf.int32, shape=[b_size])
x = tf.one_hot(x_, vocab)

h = tf.get_variable('h', shape=(b_size, n), initializer=tf.zeros_initializer())
b = tf.ones(shape=(b_size, 1))
target_ = tf.placeholder(tf.int32, shape=[b_size, 1])
target = tf.one_hot(target_, vocab)
lr = tf.placeholder(tf.float32)

#Neccesary steps for running RNN
W = tf.get_variable('W_hx', shape=(n + vocab + 1, 2 * n), initializer=tf.random_normal_initializer(stddev=0.01))
concat = tf.concat([x, h, b], axis=1)


#Calculate new hidden state
z_t = tf.split(tf.matmul(concat, W), 2, axis=1)
c, f = tf.nn.sigmoid(z_t[0]), tf.nn.sigmoid(z_t[1])
c_pre = c
c = 2 * c - 1
h_next = f * h + (1 - f) * c 

#Get derivatives and run RTRL
c_, f_ = (1 - c_pre) * c_pre, (1 - f) * f
c_ = 2 * c_


D = tf.concat([tf.matrix_diag((1 - f) * c_), tf.matrix_diag((h - c) * f_)], axis=2)

F = tf.concat([concat[0, i] * D[0] for i in range(n + vocab + 1)], axis=1)


c2h = tf.expand_dims((1 - f) * c_, axis=2) * tf.transpose(W[vocab:n+vocab, :n])
f2h = tf.expand_dims((h - c) * f_, axis=2) * tf.transpose(W[vocab:n+vocab, n:])
h2h = tf.matrix_diag(f)

H = c2h + f2h + h2h

h_bias = tf.expand_dims(concat, axis=1)



#compute norms of u, A, obtain h_ort and D_ort and compute their norms
epsilon = 0#0.0000000001 #so norms cannot be zero
u_norm = tf.norm(G1, axis=(2, 3), keepdims=True)# + epsilon


H_low_rank = tf.expand_dims(H, axis = 1)

H_low_rank = tf.tile(H_low_rank, [1,rank_approximation,1,1])

A = tf.matmul(H_low_rank, G2)

A_norm = tf.norm(A, axis=(2, 3), keepdims=True)# + epsilon

h_bias_low_rank = tf.expand_dims(h_bias, axis = 1)






if 1==1:
    vectors_ort_span = tf.concat([G1, h_bias_low_rank], axis = 1)

    gram_schmidt_aux_vec = tf.ones(shape = [b_size, rank_approximation+1, 1, 1])
    identidade_vec = tf.eye(rank_approximation+1, batch_shape=[b_size])
    identidade_vec = tf.expand_dims(identidade_vec, axis = 3)
    
    gram_schmidt_vec_coeff = tf.zeros(shape = [b_size, rank_approximation+1, rank_approximation+1])

    for s in range(rank_approximation+1):

        dot_products_vec_s = tf.reduce_sum(vectors_ort_span[:,s:s+1,:,:]*vectors_ort_span, axis=(2,3), keepdims = True)#b_s*rank+1*1*1
        vector_s_squared_norm = dot_products_vec_s[:,s:s+1,:,:]#b_s*1*1*1

        gram_schmidt_aux_vec_s = gram_schmidt_aux_vec - identidade_vec[:,:,s:s+1,:]#b_s*rank+1*1*1
        vector_s_squared_norm_tiled = tf.tile(vector_s_squared_norm, [1,rank_approximation+1,1,1])
        dot_products_vec_s = tf.where(tf.equal(vector_s_squared_norm_tiled, tf.zeros([b_size,(rank_approximation+1), 1, 1])), tf.zeros([b_size,(rank_approximation+1), 1, 1]), dot_products_vec_s/vector_s_squared_norm)
        

        matrix_auxiliar = tf.tile(identidade_vec[:,s:s+1,:,0], [1,rank_approximation+1, 1])

        

        gram_schmidt_vec_coeff = gram_schmidt_vec_coeff + tf.sqrt(vector_s_squared_norm[:,:,:,0])*dot_products_vec_s[:,:,:,0]*matrix_auxiliar
        

        dot_products_vec_s = dot_products_vec_s*gram_schmidt_aux_vec_s

        vector_s = vectors_ort_span[:,s:s+1,:,:]#take the s matrix

        orthogonalizer_factor_vec = dot_products_vec_s*vector_s

        vectors_ort_span = vectors_ort_span - orthogonalizer_factor_vec



    #need to make all row norms 1!
    vectors_ort_span_norms = tf.norm(vectors_ort_span, axis=(2,3), keepdims=True)# + epsilon
    vectors_ort_span_norms_inverted = tf.where(tf.equal(vectors_ort_span_norms, tf.zeros([b_size, rank_approximation+1, 1, 1])), tf.zeros([b_size, rank_approximation+1, 1, 1]), 1/vectors_ort_span_norms)
    vectors_ort_span = vectors_ort_span*vectors_ort_span_norms_inverted


    D_low_rank = tf.expand_dims(D, axis = 1)


    A_matrices_ort_span = tf.concat([A, D_low_rank], axis = 1)

    gram_schmidt_aux = tf.ones(shape = [b_size, rank_approximation+1, 1, 1])
    identidade = tf.eye(rank_approximation+1, batch_shape=[b_size])
    identidade = tf.expand_dims(identidade, axis = 3)
    
    gram_schmidt_mat_coeff = tf.zeros(shape = [b_size, rank_approximation+1, rank_approximation+1])

    
    for s in range(rank_approximation+1):

        dot_products_s = tf.reduce_sum(A_matrices_ort_span[:,s:s+1,:,:]*A_matrices_ort_span, axis=(2,3), keepdims = True)#b_s*rank+1*1*1
        matrix_s_squared_norm = dot_products_s[:,s:s+1,:,:]#b_s*1*1*1
        gram_schmidt_aux_s = gram_schmidt_aux - identidade[:,:,s:s+1,:]#b_s*rank+1*1*1
        matrix_s_squared_norm_tiled = tf.tile(matrix_s_squared_norm, [1,rank_approximation+1,1,1])
        dot_products_s = tf.where(tf.equal(matrix_s_squared_norm_tiled, tf.zeros([b_size,(rank_approximation+1), 1, 1])), tf.zeros([b_size,(rank_approximation+1), 1, 1]), dot_products_s/matrix_s_squared_norm)

        

        matrix_auxiliar_mat = tf.tile(identidade[:,s:s+1,:,0], [1,rank_approximation+1, 1])

        gram_schmidt_mat_coeff = gram_schmidt_mat_coeff + tf.sqrt(matrix_s_squared_norm[:,:,:,0])*dot_products_s[:,:,:,0]*matrix_auxiliar_mat
        
        
        dot_products_s = dot_products_s*gram_schmidt_aux_s

        matrix_s = A_matrices_ort_span[:,s:s+1,:,:]#take the s matrix
        orthogonalizer_factor = dot_products_s*matrix_s

        A_matrices_ort_span = A_matrices_ort_span - orthogonalizer_factor


    #need to make all row norms 1!
    A_matrices_ort_span_norms = tf.norm(A_matrices_ort_span, axis=(2,3), keepdims=True)
    A_matrices_ort_span_norms_inverted = tf.where(tf.equal(A_matrices_ort_span_norms, tf.zeros([b_size, rank_approximation+1, 1, 1])), tf.zeros([b_size, rank_approximation+1, 1, 1]), 1/A_matrices_ort_span_norms)
    A_matrices_ort_span = A_matrices_ort_span*A_matrices_ort_span_norms_inverted

 
    vector_coeff_dot_products = tf.transpose(gram_schmidt_vec_coeff, perm=[0,2,1])


    matrix_coeff_dot_products = tf.transpose(gram_schmidt_mat_coeff, perm=[0,2,1])

 


matrix_ort_basis = tf.matmul(vector_coeff_dot_products, tf.transpose(matrix_coeff_dot_products, perm = [0,2,1]))


if 1 == 1:
    coeff_ortnorm_u_h, singular_values, right_singular_values = tf.py_func(np.linalg.svd, [matrix_ort_basis], [tf.float32, tf.float32, tf.float32])

    singular_values = tf.reshape(singular_values, [b_size, rank_approximation + 1])
    coeff_ortnorm_u_h = tf.reshape(coeff_ortnorm_u_h, [b_size, rank_approximation + 1, rank_approximation + 1])
    right_singular_values = tf.reshape(right_singular_values, [b_size, rank_approximation + 1, rank_approximation + 1])

    right_singular_values = tf.transpose(right_singular_values, perm = [0,2,1])


if 0 == 1:
    coeff_ortnorm_u_h = tf.zeros(shape = [1, rank_approximation+1, rank_approximation + 1])
    singular_values = tf.zeros(shape = [1, rank_approximation+1])
    right_singular_values = tf.zeros(shape = [1, rank_approximation+1, rank_approximation + 1])
    for i in range(b_size):
        coeff_ortnorm_u_h_batch, singular_values_batch, right_singular_values_batch = tf.py_func(scipy.linalg.svd, [matrix_ort_basis[i]], [tf.float32, tf.float32, tf.float32])
        
        coeff_ortnorm_u_h_batch = tf.expand_dims(coeff_ortnorm_u_h_batch, axis = 0)
        coeff_ortnorm_u_h = tf.concat([coeff_ortnorm_u_h, coeff_ortnorm_u_h_batch], axis = 0)
        singular_values_batch = tf.expand_dims(singular_values_batch, axis = 0)
        singular_values = tf.concat([singular_values, singular_values_batch], axis = 0)
        right_singular_values_batch = tf.expand_dims(right_singular_values_batch, axis = 0)
        right_singular_values = tf.concat([right_singular_values, right_singular_values_batch], axis = 0)

    coeff_ortnorm_u_h = coeff_ortnorm_u_h[1:b_size+1]
    singular_values = singular_values[1:b_size+1]
    right_singular_values = right_singular_values[1:b_size+1]
    

diagonal_singular_values = tf.matrix_diag(singular_values)


coeff_ortnorm_A_D = tf.matmul(right_singular_values, diagonal_singular_values)


coeff_ortnorm_u_h = tf.reverse(coeff_ortnorm_u_h, [2])


coeff_ortnorm_A_D = tf.reverse(coeff_ortnorm_A_D, [2])


diagonal_entries_ord_target = tf.reverse(singular_values, [1])


diagonal_entries_target = diagonal_entries_ord_target 

diagonal_entries_average = tf.reduce_sum(diagonal_entries_target)/(b_size*(rank_approximation+1))




index_1 = tf.cast(tf.floor((n+vocab+1)*tf.random_uniform([1])),tf.int32)[0]
index_2 = tf.cast(tf.floor(n*tf.random_uniform([1])),tf.int32)[0]
index_3 = tf.cast(tf.floor(2*n*tf.random_uniform([1])),tf.int32)[0]

original_kronecker = tf.reduce_sum(G1[:,:,0,index_1]*A[:,:,index_2,index_3], axis = 1) + h_bias_low_rank[:,0,0, index_1]*D_low_rank[:,0,index_2,index_3]




correct_batch_index = -1*tf.ones(shape=[b_size], dtype=tf.int32)
diagonal_entries_initial_sum = tf.zeros(shape = [b_size])

boolean_matrix = tf.zeros(shape = [b_size, rank_approximation+1])

for s in range(rank_approximation+1):
    initial_eigen_sum = tf.reduce_sum(diagonal_entries_target[:,0:s+1], axis = 1, keepdims = True)

    boolean_vector = tf.less_equal(s*diagonal_entries_target[:,s], initial_eigen_sum[:,0])
    
    boolean_partial_matrix = tf.expand_dims(boolean_vector, axis = 0)
    boolean_partial_matrix = tf.scatter_nd([[s]], tf.cast(boolean_partial_matrix, tf.int32), [rank_approximation + 1,b_size])
    boolean_partial_matrix = tf.transpose(boolean_partial_matrix, perm = [1,0])
    boolean_matrix = boolean_matrix + tf.cast(boolean_partial_matrix, dtype = tf.float32)

    correct_batch_index = correct_batch_index + tf.cast(boolean_vector, tf.int32)
    diagonal_entries_initial_sum = diagonal_entries_initial_sum + tf.cast(boolean_vector, tf.float32)*diagonal_entries_target[:,s]

    
correct_batch_average = tf.reduce_sum(correct_batch_index)/b_size

correct_batch_index = tf.expand_dims(correct_batch_index,axis = 1, name = 'correct_batch_index')
diagonal_entries_initial_sum = tf.expand_dims(diagonal_entries_initial_sum,axis = 1)


diagonal_entries_mixed_average = tf.reduce_sum(diagonal_entries_target*boolean_matrix)/(b_size*(rank_approximation+1))


diagonal_entries_mixed_sum = tf.reduce_sum(diagonal_entries_target*boolean_matrix, axis = 1)

noise_step_true_value = diagonal_entries_mixed_sum*diagonal_entries_mixed_sum/tf.cast(correct_batch_index[:,0], dtype = tf.float32) - tf.reduce_sum(diagonal_entries_target*diagonal_entries_target*boolean_matrix, axis = 1)

noise_step_true_value_batch_average = tf.reduce_sum(noise_step_true_value)/b_size

diagonal_entries_initial_sum_inverted = tf.where(tf.equal(diagonal_entries_initial_sum[:,0], tf.zeros([b_size])), 1/(tf.cast(correct_batch_index,tf.float32)+1), 1/diagonal_entries_initial_sum)
diagonal_entries_scale_factor = tf.cast(correct_batch_index,tf.float32)*diagonal_entries_initial_sum_inverted

# produces a diagonal with entries summing to 1 proportional to those d_s satisfying *:s*d_s\leq sum_i<=s d_i where s is between 0 and rank
# for the other entries we have d_i rescaled by the factor
diagonal_entries_rescaled_sum_0 = diagonal_entries_scale_factor*(tf.ones([b_size, rank_approximation+1])*boolean_matrix) + diagonal_entries_scale_factor*(diagonal_entries_target)
diagonal_entries_rescaled = tf.where(tf.equal(diagonal_entries_initial_sum[:,0], tf.zeros([b_size])), diagonal_entries_rescaled_sum_0, diagonal_entries_scale_factor*(diagonal_entries_target))



# this orthogonal vector is what we will use to build the orthogonal basis which we later take to build the optimal terms
# note that this vector has entries 0 when s*d_s> sum_i<=s d_i 
test_error_1 = tf.ones(shape=[b_size,rank_approximation+1]) - diagonal_entries_rescaled
test_error_2 = (tf.ones(shape=[b_size,rank_approximation+1]) - diagonal_entries_rescaled)*boolean_matrix

#numerical imprecision of reduce sum forces us to take out the values that are less than 0.
vector_ort_keep_signal_squared = tf.where(tf.less(test_error_2, tf.zeros([b_size, rank_approximation+1])), tf.zeros([b_size, rank_approximation+1]), test_error_2)

vector_ort_keep_signal = tf.sqrt(vector_ort_keep_signal_squared, name = 'vector_ort_keep_signal')




# below we get the initial vector candidates where we use the diagonals to initialize it, but replace first coordinate by the vector w_i = sqrt(1-d_i)
initial_vector_candidates = tf.eye(rank_approximation+1, batch_shape=[b_size])
initial_vector_candidates = tf.matrix_set_diag(initial_vector_candidates, tf.sqrt(diagonal_entries_rescaled))

replacing_vector = vector_ort_keep_signal - initial_vector_candidates[:,0,:]
replacing_vector = tf.expand_dims(replacing_vector, axis = 0)
replace_first_coordinate = tf.scatter_nd([[0]], replacing_vector, [rank_approximation + 1, b_size, rank_approximation + 1])
replace_first_coordinate = tf.transpose(replace_first_coordinate, perm = [1,0,2])
initial_vector_candidates = initial_vector_candidates + replace_first_coordinate

orthogonal_vector_candidates = initial_vector_candidates


#gram schmidt orthogonalization
gram_schmidt_aux_matrix = tf.ones(shape = [b_size, rank_approximation+1, rank_approximation+1])
identity = tf.eye(rank_approximation+1, batch_shape=[b_size])
gram_schmidt_aux_matrix = gram_schmidt_aux_matrix - identity
for s in range(rank_approximation+1):

    dot_products = tf.matmul(orthogonal_vector_candidates, tf.transpose(orthogonal_vector_candidates, perm = [0,2,1]))
    vector_squared_norms = tf.reduce_sum(orthogonal_vector_candidates*orthogonal_vector_candidates, axis = 2, keepdims = True)# + epsilon
    vector_squared_norms_inverted = tf.where(tf.equal(vector_squared_norms, tf.zeros([b_size, rank_approximation+1, 1])), tf.zeros([b_size, rank_approximation+1, 1]), 1/vector_squared_norms)
    dot_products = dot_products*vector_squared_norms_inverted
    dot_products = dot_products*gram_schmidt_aux_matrix
    dot_products = tf.transpose(dot_products, perm = [0,2,1])
    relevant_dot_products = dot_products[:,:,s]
    relevant_dot_products = tf.expand_dims(relevant_dot_products, axis = 2)
    relevant_dot_products = tf.tile(relevant_dot_products, [1,1,rank_approximation+1])

    vector_s = orthogonal_vector_candidates[:,s,:] #take the s vector of the matrix
    vector_s = tf.expand_dims(vector_s, axis = 2) #
    vector_s = tf.tile(vector_s, [1,1,rank_approximation+1])
    vector_s = tf.transpose(vector_s, perm = [0,2,1])
    orthogonalizer_matrix = relevant_dot_products*vector_s

    orthogonal_vector_candidates = orthogonal_vector_candidates - orthogonalizer_matrix


#need to make all row norms 1!
orthogonal_vector_candidates_norms = tf.norm(orthogonal_vector_candidates, axis=(2), keepdims=True)# + epsilon
orthogonal_vector_candidates_norms_inverted = tf.where(tf.equal(orthogonal_vector_candidates_norms, tf.zeros([b_size, rank_approximation+1, 1])), tf.zeros([b_size, rank_approximation+1, 1]), 1/orthogonal_vector_candidates_norms)

orthogonal_vector_candidates = orthogonal_vector_candidates*orthogonal_vector_candidates_norms_inverted




#but recover those from s+1 to rank
boolean_matrix_broadcast = tf.expand_dims(boolean_matrix, axis = 2)
boolean_matrix_broadcast = tf.tile(boolean_matrix_broadcast, [1,1,rank_approximation+1])
recover_diagonal = initial_vector_candidates - initial_vector_candidates*boolean_matrix_broadcast
recover_diagonal_norms = tf.norm(recover_diagonal, axis=(2), keepdims=True)# + epsilon

recover_diagonal_norms_inverted = tf.where(tf.equal(recover_diagonal_norms, tf.zeros([b_size, rank_approximation+1, 1])), tf.zeros([b_size, rank_approximation+1, 1]), 1/recover_diagonal_norms)

recover_diagonal = recover_diagonal - recover_diagonal*recover_diagonal_norms_inverted

#this should recover the diagonal entries from s+1 to rank
orthogonal_vector_candidates = orthogonal_vector_candidates + recover_diagonal


#need to rescale what I have back to its size:
diagonal_entries_scale_factor_bootstrap = tf.expand_dims(diagonal_entries_scale_factor, axis = 2)
orthogonal_vector_candidates = orthogonal_vector_candidates/tf.sqrt(diagonal_entries_scale_factor_bootstrap)

sign_vector  = 2 * (tf.round(tf.random_uniform((b_size,1, rank_approximation + 1))) - 0.5)
sign_vector = tf.tile(sign_vector, [1,rank_approximation,1])

random_orthogonal_vector = orthogonal_vector_candidates[:,1:rank_approximation+1]*sign_vector #b_s*rank+1*rank+1*1*1


diagonal_entries_target_aux = tf.expand_dims(diagonal_entries_target, axis = 1)
renormalize_factor_diag = tf.where(tf.equal(diagonal_entries_target_aux, tf.zeros([b_size, 1, rank_approximation + 1])), tf.zeros([b_size, 1, rank_approximation + 1]),  1./tf.sqrt(diagonal_entries_target_aux))

random_orthogonal_vector = random_orthogonal_vector*renormalize_factor_diag


vector_coefficients = tf.matmul(random_orthogonal_vector, tf.transpose(coeff_ortnorm_u_h, perm = [0,2,1]))

vector_coefficients = tf.expand_dims(tf.expand_dims(vector_coefficients, axis = 3), axis = 3)
vectors_ort_span = tf.expand_dims(vectors_ort_span, axis = 1)
G1_opt = tf.reduce_sum(vector_coefficients*vectors_ort_span, axis = 2)#b_s*rank*1*n+vocab+1

matrix_coefficients = tf.matmul(random_orthogonal_vector, tf.transpose(coeff_ortnorm_A_D, perm = [0,2,1]))

matrix_coefficients = tf.expand_dims(tf.expand_dims(matrix_coefficients, axis = 3), axis = 3)
A_matrices_ort_span = tf.expand_dims(A_matrices_ort_span, axis = 1)


G2_opt = tf.reduce_sum(matrix_coefficients*A_matrices_ort_span, axis = 2)#b_s*rank*1*n+vocab+1


u_opt_norm = tf.norm(G1_opt, axis=(2, 3), keepdims=True)
A_opt_norm = tf.norm(G2_opt, axis=(2, 3), keepdims=True)


rescale_factor = tf.where(tf.equal(A_opt_norm*u_opt_norm, tf.zeros([b_size, rank_approximation, 1, 1])), tf.ones([b_size, rank_approximation, 1, 1]), tf.sqrt(A_opt_norm/u_opt_norm))


G1_opt = G1_opt*rescale_factor
G2_opt = G2_opt/rescale_factor

#measure error between old prediction and new one
test_new_kronecker_approx = tf.reduce_sum(G1_opt[:,:,0,index_1]*G2_opt[:,:,index_2,index_3], axis = 1)
test_error_new_prediction = test_new_kronecker_approx - original_kronecker
test_error_average = tf.reduce_sum(test_error_new_prediction)/b_size







#Generate prediction
W_out = tf.get_variable('W_out', shape=(n + 1, vocab), initializer=tf.random_normal_initializer(stddev=0.01))
h_out = tf.concat([h_next, b], axis=1)

y = tf.matmul(h_out, W_out)

y_soft = tf.nn.softmax(y)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=y))
loss2wout = tf.gradients(loss, W_out)[0]



loss2h = tf.expand_dims(tf.gradients(loss, h_next)[0], axis=1)




loss2W_joint = 0

for s in range(rank_approximation):
    loss2W = tf.matmul(loss2h, G2_opt[:,s])
    loss2W = [tf.contrib.kfac.utils.kronecker_product(G1_opt[i,s], loss2W[i]) for i in range(b_size)]
    loss2W = sum(loss2W) / b_size
    loss2W_joint = loss2W_joint + loss2W

loss2W = loss2W_joint    

res__ = tf.reduce_sum(loss2W[0] * loss2W[0]) - tf.reduce_sum(loss2W[0] * loss2W[0])



loss2W_ = tf.reshape(loss2W, (n + vocab + 1, 2 * n))

loss2W_avg    = tf.get_variable("loss2W_avg",    shape=loss2W_.get_shape(), initializer=tf.zeros_initializer())
loss2wout_avg = tf.get_variable("loss2wout_avg", shape=loss2wout.get_shape(), initializer=tf.zeros_initializer())

optimizer = tf.train.AdamOptimizer(learning_rate=lr)
loss2W_avg_ = tf.clip_by_norm(loss2W_avg / avg_steps, 100)

train = optimizer.apply_gradients([(loss2W_avg_,    W),
                                   (loss2wout_avg / avg_steps, W_out)])



clear = [loss2W_avg.assign(   loss2W_avg    * 0),
         loss2wout_avg.assign(loss2wout_avg * 0)]

resets__ = tf.placeholder(tf.float32, shape=[b_size, 1, 1, 1])
clear__ = [G1.assign(G1 * resets__), G2.assign(G2 * resets__), h.assign(h * resets__[:, 0, 0])]#, h.assign(h * resets__[:, 0])]
clear__ptb = [G1.assign(G1 * 0), G2.assign(G2 * 0)]#, h.assign(h * 0)]



debug_grad = tf.gradients(loss, h)[0]

with tf.control_dependencies([H, D, loss2W_, loss2wout, debug_grad]):
    updates.append(loss2W_avg.assign_add(   loss2W_ ))
    updates.append(loss2wout_avg.assign_add(loss2wout))

    updates.append(h.assign(h_next))

    updates.append(G1.assign(G1_opt))
    updates.append(G2.assign(G2_opt))

####VALIDATION-starts#######
############################

x_val_ = tf.placeholder(tf.int32, shape=[val_batch, val_steps])
x_val = tf.one_hot(x_val_, vocab)
b_val = tf.ones(shape=(val_batch, 1))

target_val_ = tf.placeholder(tf.int32, shape=[val_batch, val_steps])
target_val = tf.one_hot(target_val_, vocab)
h1_val = tf.zeros(shape=(val_batch, n))
h_val = h1_val

loss_val = 0
for i in range(val_steps):
    concat_val = tf.concat([x_val[:, i], h_val, b_val], axis=1)
    z_t_val = tf.split(tf.matmul(concat_val, W), 2, axis=1)
    
    c_val, f_val = tf.nn.sigmoid(z_t_val[0]), tf.nn.sigmoid(z_t_val[1])
    c_val = 2 * c_val - 1
    h_val = f_val * h_val + (1 - f_val) * c_val

    h_out_val = tf.concat([h_val, b_val], axis=1)

    y_val = tf.matmul(h_out_val, W_out)
    #y_softb = tf.nn.softmax(yb)

    loss_val += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_val[:, i], logits=y_val))
        
loss_val = loss_val / val_steps

########################
####VALIDATION-end######

        
T_ = 1

if experiment == "ptb":
    inp_ptb = np.load('ptb.train.npy')[:, None]
    data_s = inp_ptb.shape[0]

    out_ptb = np.random.randint(vocab-3, size=(data_s,))[:, None]
    out_ptb[:-1] = inp_ptb[1:]

    idx = np.linspace(0, data_s-1, b_size, dtype=np.int32)
    
    #validation
    inp_val = np.load('ptb.test.npy')
    #print(inp_val[:10])
    data_s_val = inp_val.shape[0]
    out_val = np.random.randint(vocab-3, size=(data_s_val,))
    out_val[:-1] = inp_val[1:]

    chunk_val = val_steps * val_batch
    
    data_s_val = inp_val.shape[0] - (inp_val.shape[0] % chunk_val) 

    inp_val = inp_val[:data_s_val].reshape(val_batch, data_s_val // chunk_val, val_steps)
    out_val = out_val[:data_s_val].reshape(val_batch, data_s_val // chunk_val, val_steps)

    data_s_val = inp_val.shape[1]

elif experiment == "cp":
    idx = np.zeros(b_size, np.int32) - 1
    data_i = [[0]] * b_size
    data_o = [[0]] * b_size

#######Task4######


def cp_data():
    global idx, data_i, data_o, b_size, T_, avg_log_loss
    resets = np.ones((b_size,))
    cp_inp = np.zeros((b_size,))
    cp_out = np.zeros((b_size,))
    
    for i in range(b_size):
        cur_idx = idx[i]
        if cur_idx == -1 or cur_idx == data_i[i].shape[0]:
            if avg_log_loss < 0.15: 
                T_ += 1
                avg_log_loss = 0.3

            T = max(1, T_ - np.random.randint(5))


            
            inp = np.random.randint(2, size=(2 * T + 2))

            inp[0] = 2
            #inp[T] = 2
            inp[T + 1:] = 3
            #inp[2*T + 1] = 2

            out = np.random.randint(2, size=(2 * T + 2))
            #print(out)
            out[:T+1] = 3
            out[T+1 : 2 * T+2] = inp[:T+1]

            

            data_i[i] = inp
            data_o[i] = out
            idx[i] = 0
            cur_idx = idx[i]
            resets[i] = 0
        cp_inp[i] = data_i[i][cur_idx]
        cp_out[i] = data_o[i][cur_idx]
    
    return cp_inp, cp_out, resets
    



avg_loss = 0.1
avg_log_loss = -np.log2(0.5)
prev_clear = 0
avg_clear = 1000
norm = tf.norm(G1) / b_size

cur_time = time.time()
cur_T = 1
gpu_options = tf.GPUOptions(allow_growth=True)

best_val = 99



test_error_sum = 0

correct_index_average = 0
test_error_avg = 0
noise_estimate_running_average = 0
noise_estimate_sum = 0

expected_improvement_average = 0
##to save the session!
saver = tf.train.Saver()


with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    
    #sess.run(tf.global_variables_initializer())
    saver.restore(sess, './model_2.ckpt') #restore saved model to see performance on test data

    steps = 0
    c_ = 0
    while steps < 1:
        steps += 1
        
        if experiment == "ptb" and (steps % 1) == 0:
            cur_loss_val = 0
            cur_h_val = sess.run(h1_val)
            for j in range(data_s_val):
                feed_dict = {x_val_: inp_val[:, j],
                            target_val_: out_val[:, j],
                            h1_val: cur_h_val}
                cur_run = sess.run([loss_val, h_val], feed_dict=feed_dict)
                cur_loss_val += cur_run[0] * np.log2(np.e)
                cur_h_val = cur_run[1]
                #print(i, data_s_val)
            cur_loss_val /= data_s_val
            
            print('VAL BPC: ', cur_loss_val)
            break
            if best_val > cur_loss_val:
                best_val = cur_loss_val
                saver.save(sess, "./model_test.ckpt")
        
            
        if experiment == "ptb":
            idx = idx % data_s        
            feed_dict = {x_: inp_ptb[idx, 0],
                        target_: out_ptb[idx]} 

            resets = np.clip(idx % reset_every_n, 0, 1) 
            sess.run(clear__, feed_dict={resets__: resets[:, None, None, None]})
        elif experiment == "cp":
            cp_inp, cp_out, resets = cp_data()
            feed_dict = {resets__: resets[:, None, None, None]}

            sess.run(clear__, feed_dict=feed_dict)
            feed_dict = {x_: cp_inp, target_: cp_out[:, None]}
            
            
        idx += 1       

        results_1, cur_loss, y__, _, n_, cur_c_, cor_batch_avg, diag_mix_avg, diag_avg, noise_step_batch_avg = sess.run([test_error_average, loss, y_soft, updates, norm, res__, correct_batch_average, diagonal_entries_mixed_average, diagonal_entries_average, noise_step_true_value_batch_average], feed_dict=feed_dict)


        c_ += cur_c_


        correct_index_average = correct_index_average*0.999 + cor_batch_avg*0.001

        test_error_avg = test_error_avg*0.999 + results_1*0.001
        
        noise_estimate_running_average = noise_estimate_running_average*0.999 + results_1*results_1*0.001
        
        noise_estimate_sum = noise_estimate_sum + results_1*results_1
        

        if diag_avg > 0:
            expected_improvement_average = expected_improvement_average*0.999 + (diag_mix_avg/diag_avg)*0.001


        if steps % avg_steps == avg_steps-1:
            sess.run(train, {lr: cur_lr})
            sess.run([clear])
            pass

        avg_log_loss = avg_log_loss * 0.999 + 0.001 * cur_loss * np.log2(np.e)

        if n_ > 50: 

            avg_clear = 0.99 * avg_clear + 0.01 * (steps - prev_clear)
            prev_clear = steps
            print("CLEARED")



        if steps % 1000 == 999: 
            print(test_error_sum, c_, steps, avg_clear, T_, avg_log_loss, n_, correct_index_average)
            

            print((time.time() - cur_time))
            print(expected_improvement_average, test_error_avg, noise_estimate_running_average, noise_estimate_sum, noise_step_batch_avg)

            c_ = 0
            cur_time = time.time()
            sys.stdout.flush()

                                                         

            

