import tensorflow as tf
import numpy as np
import time                                                                                                                 
import sys                                                                                                                  
                                                                                                                            
from tensorflow.python.client import timeline                                                                               

from tensorflow.python import debug as tf_debug

###################### TO MAKE SURE TENSORFLOW ONLY USES ONE GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


n = 128
n_ = 7
steps = 10 * 6000000#1000000 - 3
delay = 3
updates = []
avg_steps = 1
b_size = 256#256 
cur_lr = 10**(-3.5)
reset_every_n = 100
experiment = "cp" #"cp" or "ptb"


val_steps = 2
val_batch = 64

if experiment == "cp":
    vocab = 4
elif experiment == "ptb":
    vocab = 50

#this means we will use rank_approximation factors of form u*a_1*a_2
rank_approximation = 32


#RTRL input
G1 = tf.get_variable('G1', shape=(b_size, 1, 1, vocab + n + 1), initializer=tf.zeros_initializer())

#low rank approximation of G2; write G2 approx matmul(G2_1, G2_2); where G2_1,2 are low rank
G2_1 = tf.get_variable('G2_1', shape=(b_size, rank_approximation, n, 1), initializer=tf.zeros_initializer())
G2_2 = tf.get_variable('G2_2', shape=(b_size, rank_approximation, 1, 2*n), initializer=tf.zeros_initializer())



#RNN input
x_ = tf.placeholder(tf.int32, shape=[b_size])
x = tf.one_hot(x_, vocab)

h = tf.get_variable('h', shape=(b_size, n), initializer=tf.zeros_initializer())
b = tf.ones(shape=(b_size, 1))
target_ = tf.placeholder(tf.int32, shape=[b_size, 1])
target = tf.one_hot(target_, vocab)
lr = tf.placeholder(tf.float32)


#we replace the derivatives to get H by a numerical approximation
#this should use less memory and be faster
if 1==1:
    epsilon = 0.00001
    h_expand = tf.expand_dims(h, axis = 1)
    h_expand_epsilon = h_expand + epsilon*G2_1[:,:,:,0]#b_s,rank,n
    x_expand = tf.expand_dims(x, axis = 1)
    x_expand = tf.tile(x_expand, [1, rank_approximation, 1])
    b_expand = tf.expand_dims(b, axis = 1)
    b_expand = tf.tile(b_expand, [1, rank_approximation, 1])
    #Neccesary steps for running RNN
    W = tf.get_variable('W_hx', shape=(n + vocab + 1, 2 * n), initializer=tf.random_normal_initializer(stddev=0.01))
    W_transpose = tf.transpose(W, perm = [1,0])
    concat = tf.concat([x, h, b], axis=1)
    concat_expand = tf.concat([x_expand, h_expand_epsilon, b_expand], axis=2)
    #Calculate new hidden state
    z_t = tf.split(tf.matmul(concat, W), 2, axis=1)
    c, f = tf.nn.sigmoid(z_t[0]), tf.nn.sigmoid(z_t[1])
    c_pre = c
    c = 2 * c - 1
    h_next = f * h + (1 - f) * c
    #compute next hidden state 'shifted' by epsilon G2_1
    z_t_epsilon = tf.split(tf.einsum('bri,ik->brk', concat_expand, W), 2, axis=2)
    c_epsilon, f_epsilon = tf.nn.sigmoid(z_t_epsilon[0]), tf.nn.sigmoid(z_t_epsilon[1])
    c_epsilon = 2 * c_epsilon - 1
    h_next_epsilon = f_epsilon * h_expand_epsilon + (1 - f_epsilon) * c_epsilon
    h_next_expand = tf.expand_dims(h_next, axis = 1)
    a_1 = (h_next_epsilon - h_next_expand)/epsilon#b_s, rank, n
    a_1 = tf.expand_dims(a_1, axis = 3)
    
    #Get derivatives and run RTRL
    c_, f_ = (1 - c_pre) * c_pre, (1 - f) * f
    c_ = 2 * c_

    D_diag = tf.concat([(1 - f) * c_, (h - c) * f_], axis=1)
    D_diag = tf.expand_dims(D_diag, axis = 2)#b_s*2n*1
    D_diag = tf.expand_dims(D_diag, axis = 1)#b_s*1*2n*1
    
else:
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


    #we do not use a matrix anymore instead just take the entries
    D_diag = tf.concat([(1 - f) * c_, (h - c) * f_], axis=1)
    D_diag = tf.expand_dims(D_diag, axis = 2)#b_s*2n*1
    D_diag = tf.expand_dims(D_diag, axis = 1)#b_s*1*2n*1



    c2h = tf.expand_dims((1 - f) * c_, axis=2) * tf.transpose(W[vocab:n+vocab, :n])
    f2h = tf.expand_dims((h - c) * f_, axis=2) * tf.transpose(W[vocab:n+vocab, n:])
    h2h = tf.matrix_diag(f)

    H = c2h + f2h + h2h

    H_low_rank = tf.expand_dims(H, axis = 1)

    H_low_rank = tf.transpose(H_low_rank, perm=[0,1,3,2])

    a_1 = tf.reduce_sum(H_low_rank*G2_1, axis=2, keepdims=True)

    a_1 = tf.transpose(a_1, perm=[0,1,3,2])


h_bias = tf.expand_dims(concat, axis=1)




h_bias_low_rank = tf.expand_dims(h_bias, axis = 1)
#the vectors u
#u_complete = tf.concat([G1, h_bias_low_rank], axis = 1)

#D is n*2n; we will do an optimal low rank approx of D so we need to use svd; which is trivial by hand

u_coeff = tf.ones([b_size, 1, n, 1])

#diag
D_diag_1 = D_diag[:,:,0:n]
D_diag_2 = D_diag[:,:,n:2*n]
diag_entries_1 = tf.sqrt(D_diag_1*D_diag_1 + D_diag_2*D_diag_2)
diag_entries_2 = tf.zeros([b_size, 1, n, 1])
diag_entries_svd = tf.concat([diag_entries_1,diag_entries_2], axis = 2)

#right sing vect #only keep non-zero entries
diag_entries_inv = tf.where(tf.equal(diag_entries_1, tf.zeros([b_size,1,n,1])), tf.zeros([b_size,1,n,1]), 1./diag_entries_1)
v_coeff_1 = D_diag_1*diag_entries_inv
v_coeff_2 = D_diag_2*diag_entries_inv
v_coeff = tf.concat([v_coeff_1,v_coeff_2], axis = 2)#this is b_size, 1, 2n, 1



diagonal_entries_target_non_ordered = diag_entries_1[:,0,:,0]

diagonal_entries_ord, indices_diag_ord = tf.nn.top_k(diagonal_entries_target_non_ordered, n)
diagonal_entries_ord = tf.reverse(diagonal_entries_ord, [1])
indices_diag_ord = tf.reverse(indices_diag_ord, [1])

diagonal_entries_target = diagonal_entries_ord

auxiliary_matrix_lower_triangular_ones = tf.ones([n,n])
auxiliary_matrix_lower_triangular = tf.matrix_band_part(auxiliary_matrix_lower_triangular_ones, 0, -1)
initial_eigen_sum = tf.matmul(diagonal_entries_target, auxiliary_matrix_lower_triangular)

multiply_factor_term = tf.cast(tf.range(n), dtype = tf.float32) - (n-rank_approximation-1.)*tf.ones([n], dtype=tf.float32)
multiply_factor_term = tf.expand_dims(multiply_factor_term, axis = 0)
boolean_matrix =  tf.less_equal(multiply_factor_term*diagonal_entries_target, initial_eigen_sum)
boolean_matrix = tf.cast(boolean_matrix, dtype = tf.int32)
correct_batch_index = tf.reduce_sum(boolean_matrix, axis = 1) - 1
boolean_matrix = tf.cast(boolean_matrix, dtype = tf.float32)
diagonal_entries_initial_sum = tf.reduce_sum(diagonal_entries_target*boolean_matrix, axis = 1)


#to test how much is mixed
correct_batch_average = tf.reduce_mean(correct_batch_index)

correct_batch_index = tf.expand_dims(correct_batch_index,axis = 1, name = 'correct_batch_index')
diagonal_entries_initial_sum = tf.expand_dims(diagonal_entries_initial_sum,axis = 1)

diagonal_entries_mixed_average = tf.reduce_sum(diagonal_entries_target*boolean_matrix)/(b_size*(rank_approximation+1))

#to estimate the true noise of the experiment
#same as before but should be done per batch element as it should be divided
diagonal_entries_mixed_sum = tf.reduce_sum(diagonal_entries_target*boolean_matrix, axis = 1)

noise_step_true_value = diagonal_entries_mixed_sum*diagonal_entries_mixed_sum/(tf.cast(correct_batch_index[:,0], dtype = tf.float32)*tf.cast(correct_batch_index[:,0], dtype = tf.float32)) - tf.reduce_sum(diagonal_entries_target*diagonal_entries_target*boolean_matrix, axis = 1)

noise_step_true_value_batch_average = tf.reduce_mean(noise_step_true_value)

#rescale so sum of d_i = correct batch index
diagonal_entries_initial_sum_inverted = tf.where(tf.equal(diagonal_entries_initial_sum[:,0], tf.zeros([b_size])), 1/(tf.cast(correct_batch_index,tf.float32)+1), 1/diagonal_entries_initial_sum)
diagonal_entries_scale_factor = (tf.cast(correct_batch_index,tf.float32)-(n-rank_approximation-1.))*diagonal_entries_initial_sum_inverted

#produces a diagonal with entries summing to 1 proportional to those d_s satisfying *:s*d_s\leq sum_i<=s d_i where s is between 0 and rank
#for the other entries we have d_i rescaled by the factor
diagonal_entries_rescaled_sum_0 = diagonal_entries_scale_factor*(tf.ones([b_size, n])*boolean_matrix) + diagonal_entries_scale_factor*(diagonal_entries_target)
diagonal_entries_rescaled = tf.where(tf.equal(diagonal_entries_initial_sum[:,0], tf.zeros([b_size])), diagonal_entries_rescaled_sum_0, diagonal_entries_scale_factor*(diagonal_entries_target))



#################################################################################################
##################################################################################################
#we now need to do the rotations part; get orthogonal vectors that have diagonal in u_1*u_1^t+...+u_r*u_r^t equal to target
#we replace the rotations approach to one that does much less work
diag_entries_partial = diagonal_entries_rescaled*boolean_matrix
diag_entries_partial_sum = tf.matmul(diag_entries_partial, auxiliary_matrix_lower_triangular)#b_size, n
diag_entries_rescale_partial_sum = diag_entries_partial_sum
diag_entries_partial_sum_floor = tf.floor(diag_entries_partial_sum)
#this is to get a tensor with ones exactly at the positions where the sum of the previous coordinates are above an integer
diag_entries_partial_sum_floor_shift = tf.concat([tf.zeros([b_size, 1]), diag_entries_partial_sum_floor[:,0:n-1]], axis = 1)
diag_entries_partial_sum_floor_shift_two = tf.concat([tf.zeros([b_size, 2]), diag_entries_partial_sum_floor[:,0:n-2]], axis = 1)


#we will build a tensor with the entries so we only need to do r rotations at most
extra_mass_values = diag_entries_partial_sum - diag_entries_partial_sum_floor
missing_mass_values = 1. - extra_mass_values
#this vector is only used shifted by 1
missing_mass_values = tf.concat([tf.ones([b_size, 1]), missing_mass_values[:,0:n-1]], axis = 1)

#when the last three positions are equal the entry is picked exactly
#when the last two entries are equal but the one before is not we pick from the extra_mass_values
#when the last two entries are different but the one before is equal, we pick from missing_mass_values
#when all three are different we add a 1 at the position
current_previous_positions_equal = tf.equal(diag_entries_partial_sum_floor, diag_entries_partial_sum_floor_shift)
previous_and_before_equal = tf.equal(diag_entries_partial_sum_floor_shift, diag_entries_partial_sum_floor_shift_two)
current_before_previous_positions_equal = tf.equal(diag_entries_partial_sum_floor, diag_entries_partial_sum_floor_shift_two)

#first is unnecessary; we can just initialize the vector to the diag_entries; we want the fixed entries to be zero as they are added later
entry_comes_from_diagonal = tf.logical_and(current_previous_positions_equal, previous_and_before_equal)
initial_vector_diag_entries = diag_entries_partial

#we replace the ones which should come from the extra_mass
entry_comes_from_extra_mass = tf.logical_and(current_previous_positions_equal, tf.logical_not(previous_and_before_equal))
initial_vector_diag_entries = tf.where(entry_comes_from_extra_mass, extra_mass_values, initial_vector_diag_entries)

#now we replace the entries which come from missing_mass
entry_comes_from_missing_mass = tf.logical_and(tf.logical_not(current_previous_positions_equal), previous_and_before_equal)
initial_vector_diag_entries = tf.where(entry_comes_from_missing_mass, missing_mass_values, initial_vector_diag_entries)

#now we add the coordinates which must be one
entry_is_one = tf.logical_and(tf.logical_not(current_previous_positions_equal), tf.logical_not(previous_and_before_equal))
initial_vector_diag_entries_sqr = tf.where(entry_is_one, tf.ones([b_size, n]), initial_vector_diag_entries)

#we take the square root as the diagonal entries are the vector entries squared
initial_vector_diag_entries = tf.sqrt(initial_vector_diag_entries_sqr)
#the vector above has the entries that are used to non-zero initially (before rotations) in the ort_vec_cand

#this is a tensor mapping i to i. This is the factor we multiply the target to get the ones_ort_vec_partial
auxiliary_identity = tf.range(rank_approximation, dtype = tf.float32)
auxiliary_identity = tf.expand_dims(auxiliary_identity, axis = 0)
auxiliary_identity = tf.tile(auxiliary_identity, [b_size, 1])

#this is so the last non fixed entry is 0 as it should be
target_auxiliary_factor = auxiliary_identity + tf.ones([b_size, rank_approximation])*(1.-boolean_matrix[:,n-rank_approximation:n])
target_auxiliary_factor = tf.expand_dims(target_auxiliary_factor, axis = 2)
target_auxiliary_factor = tf.tile(target_auxiliary_factor, [1, 1, n])

#b_size, rank, n
#vector i below is one until the sum is larger than i
diag_entries_partial_sum = tf.expand_dims(diag_entries_partial_sum, axis = 1)
diag_entries_partial_sum = tf.tile(diag_entries_partial_sum, [1, rank_approximation, 1])#b_size, rank, n
bool_vec_ort_cand = tf.less(diag_entries_partial_sum, target_auxiliary_factor)
#we need to shift the entries by one. So the ones go until the first one above i
float_bool_vec_ort_cand_large_wrong = tf.cast(bool_vec_ort_cand, tf.float32)
float_bool_vec_ort_cand_large = tf.concat([tf.ones([b_size, rank_approximation-1, 1]), float_bool_vec_ort_cand_large_wrong[:, 1:rank_approximation, 0:n-1]], axis = 2)
#we want the ones to be only between the second one above i-1 and first one above i
#so we subtract the vector i by the vector i-1
float_bool_vec_ort_cand_small = tf.concat([tf.zeros([b_size, 1, n]), float_bool_vec_ort_cand_large], axis = 1)
float_bool_vec_ort_cand_large = tf.concat([float_bool_vec_ort_cand_large, tf.ones([b_size, 1, n])], axis = 1)

non_zero_entries_initial_ort_vec_cand = tf.expand_dims(float_bool_vec_ort_cand_large - float_bool_vec_ort_cand_small, axis = 3)


initial_vector_diag_entries_exp = tf.expand_dims(tf.expand_dims(initial_vector_diag_entries, axis = 1), axis = 3)

#these are the r initial vectors. We need to apply r rotations in the places where the partial
#sums crosses the integers. The two entries around are always wrong and we fix the first one by
#moving mass from the one directly below it. After r rotations everything is correct
initial_ort_vec_cand_non_fixed = initial_vector_diag_entries_exp*non_zero_entries_initial_ort_vec_cand



#we get the rotation matrices 11 entries as b_s,1,n vectos with non-one entries at the positions
#where one must transfer mass up
#in the last non fixed entry the mass will always be correct. So no rotation can ever be needed there
boolean_matrix_shift = tf.concat([boolean_matrix[:,1:n], tf.zeros([b_size, 1])], axis = 1)*boolean_matrix
positions_mass_transfer_up = tf.cast(tf.logical_not(current_previous_positions_equal), tf.float32)*boolean_matrix_shift
one_minus_positions_mass_transfer_up = tf.expand_dims(1. - positions_mass_transfer_up, axis = 1)

#the entry when doing the ith rotation is always missing mass values. The entry where one must transfer
#mass from comes from the original initial vector (those entries have not been touched yet)
#the 11 entry of the rotation matrix is (initial_vector_diag_entries_shifted-d_i)/(initial_vector_diag_entries_shifted-initial_vector_diag_entries)
initial_vector_diag_entries_sqr_shift = tf.concat([initial_vector_diag_entries_sqr[:,1:n], tf.ones([b_size,1])], axis = 1)*boolean_matrix
denominator_rotation_11_entry = initial_vector_diag_entries_sqr_shift - missing_mass_values
denominator_rotation_11_entry = tf.where(tf.less(denominator_rotation_11_entry, tf.zeros([b_size,n])), tf.zeros([b_size,n]), denominator_rotation_11_entry)
denominator_rotation_11_inv = tf.where(tf.equal(denominator_rotation_11_entry, tf.zeros([b_size, n])), tf.zeros([b_size, n]), 1./denominator_rotation_11_entry)
numerator_rotation_11_entry = initial_vector_diag_entries_sqr_shift - diag_entries_partial
numerator_rotation_11_entry = tf.where(tf.less(numerator_rotation_11_entry, tf.zeros([b_size,n])), tf.zeros([b_size,n]), numerator_rotation_11_entry)
rotation_matrix_11_entry_sqr = numerator_rotation_11_entry*denominator_rotation_11_inv*positions_mass_transfer_up
rotation_matrix_11_entry_sqr = tf.expand_dims(rotation_matrix_11_entry_sqr, axis = 1)
rotation_matrix_11_entry_sqr = tf.expand_dims(rotation_matrix_11_entry_sqr, axis = 3)
rotation_matrix_11_entry = tf.sqrt(rotation_matrix_11_entry_sqr)
rotation_matrix_11_entry = tf.where(tf.greater(rotation_matrix_11_entry, tf.ones([b_size,1, n, 1])), tf.ones([b_size, 1, n, 1]), rotation_matrix_11_entry)

#at the right coordinates this gives me the right value; at the wrong ones it is the identity (seen as a 2x2 matrix)
rotation_matrix_11_entry = rotation_matrix_11_entry*non_zero_entries_initial_ort_vec_cand
rotation_matrix_11_entry = tf.where(tf.equal(rotation_matrix_11_entry, tf.zeros([b_size, rank_approximation, n, 1])), tf.ones([b_size, rank_approximation, n, 1]), rotation_matrix_11_entry)
rotation_matrix_12_entry = -tf.sqrt(1. - rotation_matrix_11_entry*rotation_matrix_11_entry)
#we shift this to the side as this is what is used when multiplying the vectors
rotation_matrix_21_entry = -tf.concat([tf.zeros([b_size, rank_approximation,1,1]),rotation_matrix_12_entry[:,:,0:n-1]], axis = 2)
#rotation_matrix_22_entry = rotation_matrix_11_entry
#we shift this to the side as this is what is used when multiplying the vectors
rotation_matrix_22_entry = tf.concat([tf.ones([b_size, rank_approximation,1,1]),rotation_matrix_11_entry[:,:,0:n-1]], axis = 2)

ort_vec_cand = initial_ort_vec_cand_non_fixed
#we rotate the vectors now, applying the rotations in order

######################################################################
#this is to test whether the first s rotations make the diagonal entries d_0,...,d_{s-1}
test_ort_vec_cand_all_s_diag = tf.reduce_sum(ort_vec_cand*ort_vec_cand, axis = 1, keepdims = True)
#above will be a vector b_size, rank, n, n

ort_vec_cand_all_s = initial_ort_vec_cand_non_fixed


for s in range(rank_approximation - 1):
#for s in range(1):
    #I do the 2*2 multiplications by 'hand' unclear whether it would be faster with matmul, but probably does not make a big difference
    ort_vec_cand_s_11 = ort_vec_cand[:,0:s+1]*rotation_matrix_11_entry[:,s:s+1]
    ort_vec_cand_s_12 = ort_vec_cand[:,0:s+1]*rotation_matrix_12_entry[:,s:s+1]
    ort_vec_cand_s_12 = tf.concat([tf.zeros([b_size, s+1, 1, 1]), ort_vec_cand_s_12[:,:,0:n-1]], axis = 2)
    ort_vec_cand_s = ort_vec_cand_s_11 + ort_vec_cand_s_12
    ort_vec_cand_s_21 = ort_vec_cand[:,s+1:s+2]*rotation_matrix_21_entry[:,s:s+1]
    ort_vec_cand_s_21 = tf.concat([ort_vec_cand_s_21[:,:,1:n], tf.zeros([b_size, 1, 1, 1])], axis = 2)
    ort_vec_cand_s_22 = ort_vec_cand[:,s+1:s+2]*rotation_matrix_22_entry[:,s:s+1]
    ort_vec_cand_s_1 = ort_vec_cand_s_21 + ort_vec_cand_s_22
    if s < rank_approximation-2:
        ort_vec_cand = tf.concat([ort_vec_cand_s, ort_vec_cand_s_1, ort_vec_cand[:,s+2:rank_approximation]], axis = 1)
    else:
        ort_vec_cand = tf.concat([ort_vec_cand_s, ort_vec_cand_s_1], axis = 1)

    ######################################################################
    #this is to test whether the first s rotations make the diagonal entries d_0,...,d_{s-1}
    ort_vec_cand_all_s = tf.concat([ort_vec_cand_all_s, ort_vec_cand], axis = 3)
    test_ort_vec_cand_first_s_diag = tf.reduce_sum(ort_vec_cand*ort_vec_cand, axis = 1, keepdims = True)#b_size, rank, n, 1
    test_ort_vec_cand_all_s_diag = tf.concat([test_ort_vec_cand_all_s_diag, test_ort_vec_cand_first_s_diag], axis = 3)

######################################################################
#this is to test whether the first s rotations make the diagonal entries d_0,...,d_{s-1}
test_ort_vec_cand_all_s_diag = test_ort_vec_cand_all_s_diag[:,:,:,0:n]
diagonal_entries_target_tiled = tf.tile(diag_entries_1, [1, 1, 1, n-1])



#one needs to rescale ort_vec_cand back to the original scaling and take care with entries that should not be mixed
#the vectors that should not be mixed must be in the last r coordinates
#this makes all entries that are mixed zero
ort_vec_cand_fixed_diag = (tf.sqrt(diagonal_entries_rescaled[:,(n-rank_approximation):n]))*(1. - boolean_matrix[:,(n-rank_approximation):n])#we subtract one because we kept 1 there before
#non-mixed entries must be in last r coord
ort_vec_cand_fixed_diag = tf.matrix_diag(ort_vec_cand_fixed_diag)#creates a diagonal b_size*rank*rank
complete_ort_vec_cand_fixed = tf.zeros([b_size, rank_approximation, n-rank_approximation])
ort_vec_cand_fixed_diag = tf.concat([complete_ort_vec_cand_fixed, ort_vec_cand_fixed_diag], axis = 2) #these vectors are non zero only at the nonmixed entries
ort_vec_cand_fixed_diag = tf.expand_dims(ort_vec_cand_fixed_diag, axis = 3)

#to fix the previous vector we only have to sum them; and then rescale
ort_vec_cand = ort_vec_cand + ort_vec_cand_fixed_diag #b_size,rank,n,1

#################################################################################################
##################################################################################################


#now we rescale the vector
#one has to check all border cases
diagonal_entries_scale_factor_inv = tf.where(tf.equal(diagonal_entries_scale_factor, tf.zeros([b_size, 1])), tf.zeros([b_size, 1]), 1./tf.sqrt(diagonal_entries_scale_factor))
diagonal_entries_scale_factor_inv = tf.expand_dims(tf.expand_dims(diagonal_entries_scale_factor_inv, axis = 2), axis = 3)

ort_vec_cand_wrong_order = ort_vec_cand*diagonal_entries_scale_factor_inv


###############################################################################
#the lines below are to test whether the diagonal entries of test_ort_vec_cand_wrong_order_diag*test_ort_vec_cand_wrong_order_diag^t are the correct ones
test_ort_vec_cand_wrong_order_diag = tf.reduce_sum(ort_vec_cand_wrong_order*ort_vec_cand_wrong_order, axis = 1, keepdims = True)
#the vector below should be zero everywhere (if tests are correct)
test_ort_vec_cand_wrong_order_error = test_ort_vec_cand_wrong_order_diag[:,0,:,0] - diagonal_entries_target
###############################################################################
###############################################################################


#we need to recompose the vectors so the order is the correct one
#above we build the vector as if the diagonal_entries were ordered (increasing)
#however they are not. The correct index of the i-th entry is given by indices_diag_ord
#we use gather to reconstruct (would be easier with batch_gather but I do not have that)
#so I bring the vectors to dimension 1

#we have to invert the permutation so the reordering works
#this is probably slower due to the for loops. So we remove them


indices_diag_ord_inverted = tf.zeros([1,1,n], dtype = tf.int32)

for i in range(b_size):
    indices_diag_ord_batch_inverted = tf.invert_permutation(indices_diag_ord[i])
    indices_diag_ord_batch_inverted = tf.expand_dims(tf.expand_dims(indices_diag_ord_batch_inverted, axis = 0), axis = 1)
    indices_diag_ord_inverted = tf.concat([indices_diag_ord_inverted, indices_diag_ord_batch_inverted], axis = 0)

indices_diag_ord_inverted = indices_diag_ord_inverted[1:(b_size+1)]#b_size,1,n
    
indices_diag_ord_1 = n*tf.range(rank_approximation)
indices_diag_ord_1 = tf.expand_dims(tf.expand_dims(indices_diag_ord_1, axis = 0), axis = 2)
indices_diag_ord_1 = tf.tile(indices_diag_ord_1, [b_size, 1, n])
indices_diag_ord_1 = indices_diag_ord_1 + indices_diag_ord_inverted#b_size, rank,n

indices_diag_ord_onedim = rank_approximation*n*tf.range(b_size)
indices_diag_ord_onedim = tf.expand_dims(tf.expand_dims(indices_diag_ord_onedim, axis = 1), axis = 2)
indices_diag_ord_onedim = tf.tile(indices_diag_ord_onedim, [1, rank_approximation, n])
indices_diag_ord_onedim = indices_diag_ord_onedim + indices_diag_ord_1#b_size, rank,n



ort_vec_cand_onedim = tf.reshape(ort_vec_cand_wrong_order, [b_size*rank_approximation*n])

ort_vec_cand_fixed = tf.gather(ort_vec_cand_onedim, indices_diag_ord_onedim)

ort_vec_cand_fixed = tf.reshape(ort_vec_cand_fixed, [b_size, rank_approximation, n, 1])


#the lines below are to test whether the diagonal entries of ort_vec_cand_fixed*ort_vec_cand_fixed^t are the correct ones
ort_vec_cand_fixed_transpose = tf.transpose(ort_vec_cand_fixed, [0, 1, 3, 2])#b_size, rank, 1, n
test_ort_vec_cand_fixed_full = tf.reduce_sum(ort_vec_cand_fixed*ort_vec_cand_fixed_transpose, axis = 1, keepdims = True)
test_ort_vec_cand_fixed_diag = tf.reduce_sum(ort_vec_cand_fixed*ort_vec_cand_fixed, axis = 1, keepdims = True)

#the vector below should be zero everywhere (if tests are correct)
test_ort_vec_cand_fixed_error = test_ort_vec_cand_fixed_diag - diag_entries_1
tes_diag_scaled = diagonal_entries_scale_factor*diag_entries_1[:,0,:,0]
###############################################################

#sign_vector  = 2 * (tf.round(tf.random_uniform((b_size, rank_approximation, n, 1))) - 0.5)
sign_vector  = 2 * (tf.round(tf.random_uniform((b_size, 1, n, 1))) - 0.5)
sign_vector = tf.tile(sign_vector, [1,rank_approximation,1, 1])

random_ort_vec_cand = sign_vector*ort_vec_cand_fixed

#we now build the optimal factors; we write D as sum u_i*v_i where u_i n,1 and v_i 1,2n
u_i_vec = random_ort_vec_cand #b_s,rank, n, 1; in principle one has to multiply by the coefficients but they are the identity

u_i_vec_concat = tf.concat([u_i_vec, u_i_vec], axis = 2)#b_size, rank, 2n, 1
v_i_vec_transpose = v_coeff*u_i_vec_concat #b_size, rank, 2n, 1 
v_i_vec = tf.transpose(v_i_vec_transpose, perm = [0,1,3,2])#b_size, rank, 1, 2n

#diagonal_orig = D_diag[:,0,index_diag,0]
diagonal_orig = D_diag

#diagonal_approx = tf.reduce_sum(u_i_vec[:,:,index_diag,0]*v_i_vec[:,:,0,index_diag], axis = 1)
diagonal_approx = tf.reduce_sum(u_i_vec_concat*v_i_vec_transpose, axis = 1, keepdims = True)

diagonal_error = diagonal_approx - diagonal_orig

test_diag_error_max = tf.reduce_max(diagonal_error, axis = 2)

test_diag_error = tf.reduce_mean(test_diag_error_max)

#above measures the batch average of the coordinate with the most bias introduced by our approximation
#the bias is a useful test to check bugs because it should be 0 (+numerical error)
#the actual noise introduced by this approximation might be less interesting


#we have obtained an unbiased representation of D as h*u_i*v_i;
#now we have to combine these factors with the original G1*a_1*G2_2
#we do that by making use of the fact that h is always the same and doing (G1+sign*h)(a_1*G2_2+sign*u_i*v_i)
#and combining the 2r factors inside optimally to get a rank r approximator
#we incorporate the sign term into u_i_vec so we have a_1*G2_2+(signxu_i)v_i
sign_vector_uh_kron_product = 2 * (tf.round(tf.random_uniform((b_size,1,1,1))) - 0.5)

#at this point we have to compute the outer sign trick approximation. That also involves a rescaling
#in addition to the sign above. The rescaling should be incorporated in the vectors
#due to lack of orthogonality the norm a_1*G_2 involves all dot products between the a_1(i),G_2(i)
a_1_transpose = tf.transpose(a_1[:,:,:,0], perm=[0,2,1])#b_s, n, rank
a_1_part_norm_sqr = tf.matmul(a_1[:,:,:,0], a_1_transpose)#b_s, rank, rank
G2_2_transpose = tf.transpose(G2_2[:,:,0], perm=[0,2,1])#b_s, 2*n, rank
G2_2_part_norm_sqr = tf.matmul(G2_2[:,:,0], G2_2_transpose)#b_s, rank, rank
a_1_G2_2_prod_full_norm_sqr = tf.reduce_sum(a_1_part_norm_sqr*G2_2_part_norm_sqr, axis = (1,2), keepdims = True)
a_1_G2_2_prod_full_norm_sqr = tf.expand_dims(a_1_G2_2_prod_full_norm_sqr, axis = 3)
a_1_G2_2_prod_full_norm = tf.sqrt(a_1_G2_2_prod_full_norm_sqr)


G1_norm = tf.norm(G1, axis = (2,3), keepdims = True)


u_i_vec_norm = tf.norm(u_i_vec, axis = (2,3), keepdims = True)
v_i_vec_norm = tf.norm(v_i_vec, axis = (2,3), keepdims = True)
u_i_vec_v_i_vec_prod_norm_sqr = u_i_vec_norm*u_i_vec_norm*v_i_vec_norm*v_i_vec_norm
#the vectors u_i, v_i are orthogonal so the total norm should be the sum of the norms
u_i_vec_v_i_vec_prod_full_norm = tf.reduce_sum(u_i_vec_v_i_vec_prod_norm_sqr, axis = (1), keepdims = True)
u_i_vec_v_i_vec_prod_full_norm = tf.sqrt(u_i_vec_v_i_vec_prod_full_norm)

h_bias_low_rank_norm = tf.norm(h_bias_low_rank, axis = (2,3), keepdims = True)

#we now compute the rescaling factors
rescale_factor_1 = tf.where(tf.equal(G1_norm, tf.zeros([b_size, 1, 1, 1])), tf.zeros([b_size, 1, 1, 1]), tf.sqrt(a_1_G2_2_prod_full_norm/G1_norm))
rescale_factor_2 = tf.where(tf.equal(h_bias_low_rank_norm, tf.zeros([b_size, 1, 1, 1])), tf.zeros([b_size, 1, 1, 1]), tf.sqrt(u_i_vec_v_i_vec_prod_full_norm/h_bias_low_rank_norm))
rescale_factor_1_inv = tf.where(tf.equal(rescale_factor_1, tf.zeros([b_size, 1, 1, 1])), tf.zeros([b_size, 1, 1, 1]), 1./rescale_factor_1)
rescale_factor_2_inv = tf.where(tf.equal(rescale_factor_2, tf.zeros([b_size, 1, 1, 1])), tf.zeros([b_size, 1, 1, 1]), 1./rescale_factor_2)

#we compute G1_opt with the sign trick
G1_opt = rescale_factor_1*G1 + rescale_factor_2*sign_vector_uh_kron_product*h_bias_low_rank

#G1_opt = tf.transpose(G1_opt, perm = [0,2,1,3,4,5,6,7])

#this implies that the last two factors are: rescale_factor_1_inv*a_1*G2_2 +sign*rescale_factor_2_inv*u_i_vec*v_i_vec
#so we just rescale a_1 and u_i_vec
a_1_old = a_1

a_1 = rescale_factor_1_inv*a_1

u_i_vec_sign = sign_vector_uh_kron_product*rescale_factor_2_inv*u_i_vec

#we apply sign tricks to the factors 2 to 1 r times
a_1_norm = tf.norm(a_1, axis =(2,3), keepdims = True)
G2_2_norm = tf.norm(G2_2, axis =(2,3), keepdims = True)

#the vectors u_i_vec are orthogonal by construction (norms are not 1 though)
#we make their norms 1 to fix this first
u_i_vec_sign_norm = tf.norm(u_i_vec_sign, axis=(2,3), keepdims=True)#b_s,rank_approximation,1,1
#v_i_vec_norm is known

rescale_factor_3 = tf.where(tf.equal(a_1_norm, tf.zeros([b_size, rank_approximation, 1, 1])), tf.zeros([b_size, rank_approximation, 1, 1]), tf.sqrt(G2_2_norm/a_1_norm))
rescale_factor_4 = tf.where(tf.equal(u_i_vec_sign_norm, tf.zeros([b_size, rank_approximation, 1, 1])), tf.zeros([b_size, rank_approximation, 1, 1]), tf.sqrt(v_i_vec_norm/u_i_vec_sign_norm))
rescale_factor_3_inv = tf.where(tf.equal(rescale_factor_3, tf.zeros([b_size, rank_approximation, 1, 1])), tf.zeros([b_size, rank_approximation, 1, 1]), 1./rescale_factor_3)
rescale_factor_4_inv = tf.where(tf.equal(rescale_factor_4, tf.zeros([b_size, rank_approximation, 1, 1])), tf.zeros([b_size, rank_approximation, 1, 1]), 1./rescale_factor_4)

sign_vector_kron_2 = 2 * (tf.round(tf.random_uniform((b_size,rank_approximation,1,1))) - 0.5)

G2_1_opt = rescale_factor_3*a_1 + rescale_factor_4*sign_vector_kron_2*u_i_vec_sign
G2_2_opt = rescale_factor_3_inv*G2_2 + rescale_factor_4_inv*sign_vector_kron_2*v_i_vec


if 0==1:
    G1_opt_norm = tf.norm(G1_opt, axis=(2, 3), keepdims=True)# + epsilon
    G2_1_opt_norm = tf.norm(G2_1_opt, axis=(2, 3), keepdims=True)# + epsilon
    G2_2_opt_norm = tf.norm(G2_2_opt, axis=(2, 3), keepdims=True)# + epsilon

    prod_factor_inv = tf.where(tf.equal(G1_opt_norm*G2_1_opt_norm*G2_2_opt_norm, tf.zeros([b_size, rank_approximation, 1, 1])), tf.ones([b_size, rank_approximation, 1, 1]), 1./tf.pow(G1_opt_norm*G2_1_opt_norm*G2_2_opt_norm,  1./3.))
    rescale_factor_1 = tf.where(tf.equal(G1_opt_norm, tf.zeros([b_size, rank_approximation, 1, 1])), tf.ones([b_size, rank_approximation, 1, 1]), prod_factor_inv/G1_opt_norm)
    rescale_factor_2 = tf.where(tf.equal(G2_1_opt_norm, tf.zeros([b_size, rank_approximation, 1, 1])), tf.ones([b_size, rank_approximation, 1, 1]), prod_factor_inv/G2_1_opt_norm)
    rescale_factor_3 = tf.where(tf.equal(G2_2_opt_norm, tf.zeros([b_size, rank_approximation, 1, 1])), tf.ones([b_size, rank_approximation, 1, 1]), prod_factor_inv/G2_2_opt_norm)

    G1_opt = G1_opt*rescale_factor_1
    G2_1_opt = G2_1_opt*rescale_factor_2
    G2_2_opt = G2_2_opt*rescale_factor_3


#test error in approximation
#we improve the test to average over many points
if 0==1:
    index_1 = tf.cast(tf.floor((n+vocab+1)*tf.random_uniform([1])),tf.int32)[0]
    index_2 = tf.cast(tf.floor(n*tf.random_uniform([1])),tf.int32)[0]
    index_3 = tf.cast(tf.floor(2*n*tf.random_uniform([1])),tf.int32)[0]

    original_kronecker = tf.reduce_sum(G1[:,:,0,index_1]*a_1[:,:,index_2,0]*G2_2[:,:,0,index_3], axis = 1) + h_bias_low_rank[:,0,0, index_1]*tf.where(tf.equal(index_2, index_3), D_diag[:,0,index_2,0], tf.zeros([b_size]))

    kronecker_approx = tf.reduce_sum(G1_opt[:,:,0,index_1]*G2_1_opt[:,:,index_2,0]*G2_2_opt[:,:,0,index_3], axis = 1)

    kronecker_error = kronecker_approx - original_kronecker
    test_approx_error = tf.reduce_mean(kronecker_error, axis = 0)
else:
    n_points = 4
    index_1 = tf.cast(tf.floor((n+vocab+1)*tf.random_uniform([n_points])),tf.int32)
    index_2 = tf.cast(tf.floor(n*tf.random_uniform([n_points])),tf.int32)
    index_3 = tf.cast(tf.floor(2*n*tf.random_uniform([n_points])),tf.int32)

    G1_short = tf.gather(G1, index_1, axis = 3)
    a_1_short = tf.gather(a_1, index_2, axis = 2)
    G2_2_short = tf.gather(G2_2, index_3, axis = 3)
    h_bias_low_rank_short = tf.gather(h_bias_low_rank, index_1, axis = 3)

    #the diagonal entry is harder to build because we do not store all the values
    D_diag_short = tf.gather(D_diag, index_2, axis = 2)
    #the tilings must be different because the goal is doing all n_points^2 pairwise comparisons
    index_2_tile = tf.expand_dims(tf.expand_dims(tf.expand_dims(index_2, axis = 0), axis = 1), axis = 3)
    index_2_tile = tf.tile(index_2_tile, [b_size, 1, 1, n_points])
    index_2_tile = tf.reshape(index_2_tile, [b_size, 1, n_points*n_points, 1])
    index_3_tile = tf.expand_dims(tf.expand_dims(tf.expand_dims(index_3, axis = 0), axis = 1), axis = 2)
    index_3_tile = tf.tile(index_3_tile, [b_size, 1, n_points, 1])
    index_3_tile = tf.reshape(index_3_tile, [b_size, 1, n_points*n_points, 1])
    D_diag_short_tile = tf.tile(D_diag_short, [1, 1, n_points, 1])
    D_diag_short_full = tf.where(tf.equal(index_2_tile, index_3_tile), D_diag_short_tile, tf.zeros([b_size, 1, n_points*n_points, 1]))

    #n_points^3 values of the kronecker factor
    #n_points^2 values in matrix
    matrix_kron = a_1_short*G2_2_short
    matrix_kron = tf.reshape(matrix_kron, [b_size,rank_approximation,n_points*n_points,1])
    original_kronecker = tf.reduce_sum(G1_short*matrix_kron, axis = 1) + h_bias_low_rank_short*D_diag_short_full

    G1_opt_short = tf.gather(G1_opt, index_1, axis = 3)
    G2_1_opt_short = tf.gather(G2_1_opt, index_2, axis = 2)
    G2_2_opt_short = tf.gather(G2_2_opt, index_3, axis = 3)

    matrix_kron_approx = G2_1_opt_short*G2_2_opt_short
    matrix_kron_approx = tf.reshape(matrix_kron_approx, [b_size,rank_approximation,n_points*n_points,1])

    kronecker_approx = tf.reduce_sum(G1_opt_short*matrix_kron_approx, axis = 1)

    kronecker_error = kronecker_approx - original_kronecker
    test_approx_error = tf.reduce_mean(kronecker_error)

#Generate prediction
W_out = tf.get_variable('W_out', shape=(n + 1, vocab), initializer=tf.random_normal_initializer(stddev=0.01))
h_out = tf.concat([h_next, b], axis=1)

y = tf.matmul(h_out, W_out)

y_soft = tf.nn.softmax(y)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=y))
loss2wout = tf.gradients(loss, W_out)[0]



loss2h = tf.expand_dims(tf.gradients(loss, h_next)[0], axis=1)

###################################################
###################################################
###################################################

#below is slow so we do an alternative
if 0==1:
    loss2W_joint = 0
    for s in range(rank_approximation):
        loss2W = tf.matmul(loss2h, G2_1_opt[:,s])#b_size,1,1
        #loss2W = [tf.contrib.kfac.utils.kronecker_product(G1_opt[i,s], loss2W[i]) for i in range(b_size)]
        loss2W = [tf.contrib.kfac.utils.kronecker_product(G1_opt[i,s],loss2W[i]*G2_2_opt[i,s]) for i in range(b_size)]
        G2_1_opt = tf.transpose(G2_1_opt, perm=[0,2,1,3,4,5,6,7])
        loss2W = sum(loss2W) / b_size
        loss2W_joint = loss2W_joint + loss2W
    loss2W = loss2W_joint    
else:
    loss2h_exp = tf.expand_dims(loss2h, axis = 1)#b_size, 1, 1, n
    G2_1_opt_transpose = tf.transpose(G2_1_opt, perm = [0,1,3,2])#G2_1_opt is b_size, rank, n, 1
    loss2W = tf.reduce_sum(loss2h_exp*G2_1_opt_transpose, axis = 3, keepdims=True)#b_size,rank,1,1
    vector_kron_product = loss2W*G2_2_opt#b_size, rank, 1, 2*n
    #G1_opt_transpose =  tf.transpose(G1_opt, perm = [0,1,3,2])#b_size, rank, n+vocab+1,1
    G1_opt_tile = tf.tile(G1_opt, [1,rank_approximation, 1, 1])
    loss2W = tf.einsum('bri,brj->ij' ,G1_opt_tile[:,:,0,:], vector_kron_product[:,:,0,:])
    #the loss should be averaged over the batch
    loss2W = loss2W/b_size
    


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

clear__ = [G1.assign(G1 * resets__), G2_1.assign(G2_1 * resets__), G2_2.assign(G2_2 * resets__),  h.assign(h * resets__[:, 0, 0])]#, h.assign(h * resets__[:, 0])]

clear__ptb = [G1.assign(G1 * 0), G2_1.assign(G2_1 * 0), G2_2.assign(G2_2 * 0)]#, h.assign(h * 0)]



debug_grad = tf.gradients(loss, h)[0]


with tf.control_dependencies([D_diag, loss2W_, loss2wout, debug_grad]):#, G1_new, G2_new]):
    updates.append(loss2W_avg.assign_add(   loss2W_ ))
    updates.append(loss2wout_avg.assign_add(loss2wout))

    updates.append(h.assign(h_next))

    updates.append(G1.assign(G1_opt))
    #updates.append(G2.assign(G2_opt))
    updates.append(G2_1.assign(G2_1_opt))
    updates.append(G2_2.assign(G2_2_opt))


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
    inp_val = np.load('ptb.valid.npy')
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
            #print(inp)
            inp[0] = 2
            #inp[T] = 2
            inp[T + 1:] = 3
            #inp[2*T + 1] = 2

            out = np.random.randint(2, size=(2 * T + 2))
            #print(out)
            out[:T+1] = 3
            out[T+1 : 2 * T+2] = inp[:T+1]
            #out[2*T] = 2
            #out[2*T:] = 3
            
            #print(inp[:T])
            #print(inp)
            #print(out)
            #print(aaa)
            data_i[i] = inp
            data_o[i] = out
            idx[i] = 0
            cur_idx = idx[i]
            resets[i] = 0
        cp_inp[i] = data_i[i][cur_idx]
        cp_out[i] = data_o[i][cur_idx]
    
    return cp_inp, cp_out, resets
    

#check = tf.add_check_numerics_ops()
##########################

avg_loss = 0.1
avg_log_loss = -np.log2(0.5)
prev_clear = 0
avg_clear = 1000
norm = tf.norm(G1) / b_size

cur_time = time.time()
cur_T = 1
gpu_options = tf.GPUOptions(allow_growth=True)

best_val = 99

test_value_sum = 0
test_error_sum = 0

correct_index_average = 0
test_error_avg = 0
noise_estimate_running_average = 0
noise_estimate_sum = 0

expected_improvement_average = 0
##to save the session!
saver = tf.train.Saver()

max_mean_count_avg = 0

test_error_sum = 0
test_error_diag_sum = 0
correct_index_sum = 0
test_error_full_sum = 0
true_noise_sum = 0
true_noise_partial_sum = 0

noise_estimate_partial_sum = 0
noise_estimate_sum = 0

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
   
    sess.run(tf.global_variables_initializer())
    
    steps = 0
    c_ = 0
    while steps < 2000000:
        steps += 1

        if experiment == "ptb" and (steps % 10000) == 1:
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
            
            if best_val > cur_loss_val:
                best_val = cur_loss_val
                saver.save(sess, "./model_fixed_35_1.ckpt")

        if experiment == "ptb":
            idx = idx % data_s        
            feed_dict = {x_: inp_ptb[idx, 0],
                        target_: out_ptb[idx]} 
            #if steps % 200 == 0:
            #    sess.run(clear__ptb)
            #resets = np.random.choice([0, 1], size=b_size, p=[1/40., 39/40.])
            resets = np.clip(idx % reset_every_n, 0, 1) 
            sess.run(clear__, feed_dict={resets__: resets[:, None, None, None]})
        elif experiment == "cp":
            cp_inp, cp_out, resets = cp_data()
            feed_dict = {resets__: resets[:, None, None, None]}
            #if steps % 100 == 0:
            #    sess.run(clear__ptb)
            sess.run(clear__, feed_dict=feed_dict)
            feed_dict = {x_: cp_inp, target_: cp_out[:, None]}
            
            
        idx += 1       

        cur_loss, y__, _, n_, test_error, test_diag_err, corr_batch_avg, true_noise = sess.run([loss, y_soft, updates, norm, test_approx_error, test_diag_error, correct_batch_average, noise_step_true_value_batch_average], feed_dict=feed_dict)

        test_error_sum = test_error_sum + test_error

        #we store a running average of the correct_batch_index averaged over batch elements
        #the smaller this is the stronger our approach is at approximating the gradient

        test_error_diag_sum = test_error_diag_sum + test_diag_err
        
        correct_index_sum = correct_index_sum + corr_batch_avg

        true_noise_partial_sum = true_noise_partial_sum + true_noise
        
        noise_estimate_partial_sum = noise_estimate_partial_sum + test_error*test_error

        
        if steps % avg_steps == avg_steps-1:
            sess.run(train, {lr: cur_lr})
            sess.run([clear])
            pass

        avg_log_loss = avg_log_loss * 0.999 + 0.001 * cur_loss * np.log2(np.e)

        if n_ > 50: 
            #sess.run(clear__ptb)
            avg_clear = 0.99 * avg_clear + 0.01 * (steps - prev_clear)
            prev_clear = steps
            print("CLEARED")


        if steps % 1000 == 999:
            test_error_avg = test_error_sum/1000
            test_error_full_sum = test_error_full_sum + test_error_sum
            test_error_diag_avg = test_error_diag_sum/1000
            correct_index_avg = correct_index_sum/1000
            true_noise_partial_avg = true_noise_partial_sum/1000
            true_noise_sum = true_noise_sum + true_noise_partial_sum
            noise_estimate_sum = noise_estimate_sum + noise_estimate_partial_sum
            noise_estimate_partial_avg = noise_estimate_partial_sum/1000

            true_noise_partial_sum = 0
            noise_estimate_partial_sum = 0
            test_error_sum = 0
            test_error_diag_sum = 0
            correct_index_sum = 0
            print(steps, avg_clear, T_, avg_log_loss, n_, test_error_avg, test_error_full_sum, correct_index_avg)
            
            #if steps == 2000000 - 1:
            #    save_path = saver.save(sess, "/tmp/KF-RTRL-session-new-curr.ckpt")
            #    print("Model saved in path: %s" % save_path)
            print((time.time() - cur_time), test_error_diag_avg, true_noise_partial_avg, true_noise_sum, noise_estimate_partial_avg, noise_estimate_sum)

            cur_time = time.time()
            sys.stdout.flush()

                                                         
        #sess.run([clear__])
            

