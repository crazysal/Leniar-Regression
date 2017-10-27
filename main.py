# coding: utf-8
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
print("Name : Saleem AHmed \n Ubit : sahmed9 \n Person Number: 50247637")

syn_input_data = np.loadtxt('./input.csv', delimiter=',');
syn_trn_length = int(len(syn_input_data) *0.7);
syn_tst_length = int(syn_trn_length + (len(syn_input_data) *0.15));
syn_ip_train, syn_ip_test, syn_ip_valid = syn_input_data[:syn_trn_length, :] , syn_input_data[syn_trn_length:syn_tst_length, :] , syn_input_data[syn_tst_length:, :] 

l_forLambda = 0.1 ; #Regularization Lambda
k_forK_means = 100 ; # Number of Clusters

letor_input_data = np.loadtxt('./Querylevelnorm_X.csv', delimiter=',');
letor_trn_length = int(len(letor_input_data) *0.7);
letor_tst_length = int(letor_trn_length + (len(letor_input_data) *0.15));
letor_ip_train, letor_ip_test, letor_ip_valid = letor_input_data[:letor_trn_length, :] , letor_input_data[letor_trn_length:letor_tst_length, :] , letor_input_data[letor_tst_length:, :] 

# print(len(letor_ip_train),'\n', letor_ip_train[0]); 
# print(len(letor_ip_test),'\n', letor_ip_test[0]); 
# print(len(letor_ip_valid), '\n',letor_ip_valid[0]); 

syn_output_data = np.loadtxt('./output.csv', delimiter=',',dtype=None).reshape([-1, 1]);
syn_op_train, syn_op_test, syn_op_valid = syn_output_data[:syn_trn_length, :] , syn_output_data[syn_trn_length:syn_tst_length, :] , syn_output_data[syn_tst_length:, :] 

letor_output_data = np.loadtxt('./Querylevelnorm_t.csv', delimiter=',',dtype=None).reshape([-1, 1]);
letor_op_train, letor_op_test, letor_op_valid = letor_output_data[:letor_trn_length, :] , letor_output_data[letor_trn_length:letor_tst_length, :] , letor_output_data[letor_tst_length:, :] 
# print(len(letor_op_train),'\n', letor_op_train[0]); 
# print(len(letor_op_test),'\n', letor_op_test[0]); 
# print(len(letor_op_valid), '\n',letor_op_valid[0]);

### performing k-means clustering to spatiially divide inpute data and find centres matrix 

### Input - number_of_clusters - scalar int 
### Input - data_to_cluster - RXC 2d matrix : input data
### Output - Kmeans object having centroid and lableled clusters information

def computing_KMeans_clusters(number_of_clusters, data_to_cluster) : 
    return KMeans(n_clusters= number_of_clusters, random_state=0).fit(data_to_cluster)

### Finding Means ie center of each cluster -
### Input - clustered_obj - object - output of kmeans : computing_centers_from_clusters
### Output - array of centroid -  k_forK_means X C 2d matrix 
def get_data_centers(clustered_obj):
    return clustered_obj.cluster_centers_

### Performing kmeans on training input - Synthetic 
kmeans_t_syn = computing_KMeans_clusters(k_forK_means, syn_ip_train)

### Center value arrays for training input Synthetic
train_input_cntr_syn = get_data_centers(kmeans_t_syn)
# print(np.shape(train_input_cntr_syn))
# print(train_input_cntr_syn)

### Finding spread 
### Input - data - RXC 2d matrix : input data to find covariance over
### Output - RXC 2d matrix : cov matrix
def get_spreads(clusters):
    return list(np.diag(np.diag(np.cov(clusters[i].T))) for i in range(k_forK_means))   #np.cov(data.T)

### Creating a dictionary with key - cluster label and value - cluster inpputs set
def get_cluster_dictionary(data_set, cluster_obj): 
    return {i: data_set[np.where(cluster_obj.labels_ == i)] for i in range(k_forK_means)}

### Generating Synthetic Training Cluster Dictionary
syn_ip_train_clusters_dict = get_cluster_dictionary(syn_ip_train, kmeans_t_syn)

### Caculating Spreads for Synthetic Training Data
train_input_cov_syn = get_spreads(syn_ip_train_clusters_dict)
train_input_cov_syn[0]

### Using Junchu's function to find out design matrix - 
### Input - X - Data set: RXC 
### Input - centers - Stacked Means:  k_forK_ X C 
### Input - spread - covariance matrix for entire X:  R X C 
def compute_design_matrix(X, centers, spreads):
# use broadcast
    basis_func_outputs = np.exp(np.sum(np.matmul(X - centers, spreads) * (X - centers),axis=2) / (-2)).T
# insert ones to the 1st col
    return np.insert(basis_func_outputs, 0, 1, axis=1)

### Finding design matrix for training input set Synthetic Training
phi_syn_t = compute_design_matrix(syn_ip_train, train_input_cntr_syn[:, None] , train_input_cov_syn)

### Junchu's Function to find closed form solution 
### Input - L2_Lambda - lambda - scalar 
### Input - design_matrix - phi calculated before - R X (k_forK_means +1 ) 2d matrix
### Input - output_data - RX1 2d matrix - Target values from data set given
def closed_form_sol(L2_lambda, design_matrix, output_data):
    return np.linalg.solve( L2_lambda * np.identity(design_matrix.shape[1]) + np.matmul(design_matrix.T, design_matrix), np.matmul(design_matrix.T, output_data)).flatten()

### Calculating Wml - weight for maximum likelihood from the closed form sol Synthetic 
wml_syn = np.array(closed_form_sol(l_forLambda, phi_syn_t, syn_op_train)).reshape(1,-1)

### Function To Return WXPHI - Predicted Values as generated from my leniar regression equation
def get_predicted_value(design_matrix, wml):
    return np.matmul(design_matrix, wml.T)

### Getting predicted Values for Synthetic Training Input
prediction_set_syn = get_predicted_value (phi_syn_t, wml_syn)
### Getting Learning Error for Synthetic Data : 
print("RMS Error of Synthetic Training  SET Closed Form :  ", mean_squared_error(syn_op_train, prediction_set_syn)) 

### Validation Set Operations for Synthetic Data
kmeans_v_syn = computing_KMeans_clusters(k_forK_means, syn_ip_valid)
kmeans_v_cluster_dict_syn = get_cluster_dictionary(syn_ip_valid,kmeans_v_syn)
valid_input_cov_syn = get_spreads(kmeans_v_cluster_dict_syn)
valid_input_cntr_syn = get_data_centers(kmeans_v_syn)
phi_syn_v = compute_design_matrix(syn_ip_valid, valid_input_cntr_syn[:, None] , valid_input_cov_syn)

### Predicted Values for Synthetic Validation Set 
prediction_set_v_syn = get_predicted_value (phi_syn_v, wml_syn)
print("RMS Error of Synthetic Validation SET  Closed Form  : ",mean_squared_error(syn_op_valid, prediction_set_v_syn))

### Operations for Synthetic Testing Set Data
kmeans_te_syn = computing_KMeans_clusters(k_forK_means, syn_ip_test)
kmeans_v_cluster_dict_syn_te = get_cluster_dictionary(syn_ip_test, kmeans_te_syn)
input_cov_syn_te = get_spreads(kmeans_v_cluster_dict_syn_te)
input_cntr_syn_te = get_data_centers(kmeans_te_syn)
phi_syn_te = compute_design_matrix(syn_ip_test, input_cntr_syn_te[:, None] , input_cov_syn_te)

### Predicted Values for Synthetic Testing Set 
prediction_set_te_syn = get_predicted_value (phi_syn_te, wml_syn)
print("RMS Error of Synthetic Testing SET  Closed Form  : ",np.sqrt(mean_squared_error(syn_op_test, prediction_set_te_syn)))

### Output Closed Form Solution Synthetic Data Training Closed Form
print ("CLOSED FORM Solution For Synthetic Training Set \n", closed_form_sol (l_forLambda, phi_syn_t, syn_op_train))

### Performing kmeans on training input - Letor 
kmeans_t_letor = computing_KMeans_clusters(k_forK_means, letor_ip_train)
train_input_cntr_syn = get_data_centers(kmeans_t_letor)
# print(np.shape(train_input_cntr_syn))
# print(train_input_cntr_syn) 

### Center value arrays for training input Letor
train_input_cntr_letor = get_data_centers(kmeans_t_letor)
# print(np.shape(train_input_cntr_letor))
# print(train_input_cntr_letor)

### Generating Letor Training Cluster Dictionary
letor_ip_train_clusters_dict = get_cluster_dictionary(letor_ip_train, kmeans_t_letor)
letor_ip_train_clusters_dict

### Caculating Spreads for Synthetic Training Data
train_input_cov_letor = get_spreads(letor_ip_train_clusters_dict)
train_input_cov_letor[0]

### Finding design matrix for training input set Letor Training
phi_letor_t = compute_design_matrix(letor_ip_train, train_input_cntr_letor[:, None] , train_input_cov_letor)

### Calculating Wml - weight for maximum likelihood from the closed form sol for Letor 
wml_letor = np.array(closed_form_sol(l_forLambda, phi_letor_t, letor_op_train)).reshape(1,-1)

### Getting predicted Values for Letor Training Input
prediction_set_t_letor = get_predicted_value (phi_letor_t, wml_letor)
#Getting Learning Error for Letor Data : 
print("RMS Error of LETOR Training  SET  Closed Form  :  ", np.sqrt(mean_squared_error(letor_op_train, prediction_set_t_letor))); 

### Validation Set Operations for Letor Data
kmeans_v_letor = computing_KMeans_clusters(k_forK_means, letor_ip_valid)
kmeans_v_cluster_dict_letor = get_cluster_dictionary(letor_ip_valid,kmeans_v_letor)
valid_input_cov_letor = get_spreads(kmeans_v_cluster_dict_letor)
valid_input_cntr_letor = get_data_centers(kmeans_v_letor)
phi_letor_v = compute_design_matrix(letor_ip_valid, valid_input_cntr_letor[:, None] , valid_input_cov_letor)

### Predicted Values for Letor Validation Set 
prediction_set_v_letor = get_predicted_value (phi_letor_v, wml_letor)
print("RMS Error of LETOR Validation SET  Closed Form  :  ",np.sqrt(mean_squared_error(letor_op_valid, prediction_set_v_letor))  )

### Operations for Testing Set  Letor Data
kmeans_te_letor = computing_KMeans_clusters(k_forK_means, letor_ip_test)
kmeans_te_cluster_dict_letor = get_cluster_dictionary(letor_ip_test, kmeans_te_letor)
input_cov_letor_te = get_spreads(kmeans_te_cluster_dict_letor)
input_cntr_letor_te = get_data_centers(kmeans_te_letor)
phi_letor_te = compute_design_matrix(letor_ip_test, input_cntr_letor_te[:, None] , input_cov_letor_te)

### Predicted Values for Letor Testing Set 
prediction_set_te_letor = get_predicted_value (phi_letor_te, wml_letor)
print("RMS Error of LETOR Testing SET  Closed Form  :  ", np.sqrt(mean_squared_error(letor_op_test, prediction_set_te_letor)))  

### Output Closed Form Solution Letor Data Training Closed Form
print ("CLOSED FORM Solution For Letor Training Set \n", closed_form_sol (l_forLambda, phi_letor_t, letor_op_train))

###  Junchu's function to compute Stochastic Gradient Descent
def SGD_sol(learning_rate, minibatch_size, num_epochs, L2_lambda, design_matrix_t, design_matrix_v, output_data_t, output_data_v):
    N, _ = design_matrix_t.shape
    weights = np.zeros([1, k_forK_means+1])
    er_valid = -999
    j = 0 
    p = 10
    for epoch in range(num_epochs):
        er_valid_prev = er_valid
        for i in range(1):
            lower_bound = i * minibatch_size
            upper_bound = min((i+1)*minibatch_size, N)
            Phi = design_matrix_t[lower_bound : upper_bound, :]
            t = output_data_t[lower_bound : upper_bound, :]
            E_D = np.matmul((np.matmul(Phi, weights.T)-t).T,Phi)  
            E = (E_D + L2_lambda * weights) / minibatch_size
            weights_prev = weights
            weights = weights - learning_rate * E
        er_valid = compute_prediction_valid_error(weights_prev.flatten(), design_matrix_v, output_data_v)
        if(er_valid > er_valid_prev):
            j+=1
        else: 
            j=0
        if(j>=p):
            return weights_prev.flatten()
    print ("SGD DONE ! \n Norm E : ", np.linalg.norm(E))
    return weights.flatten()

def compute_prediction_valid_error(weights, phi, op):
    temp_wts = np.array(weights).reshape(1,-1)
    predicted_values = get_predicted_value(phi, temp_wts)
    return np.sqrt(mean_squared_error(op, predicted_values))

### Calculating Stochastic Gradient Descent for synthetic training set
learning_rate = 0.01
minibatch_size =  len(syn_op_train)
num_epochs = 1000
L2_lambda = 0.1 
sgd_weights_syn_t = SGD_sol(learning_rate, minibatch_size, num_epochs, L2_lambda, phi_syn_t, phi_syn_v, syn_op_train, syn_op_valid)

### Fetch predicted value for Synthetic Training Set 
sgd_weights_syn_t = np.array(sgd_weights_syn_t).reshape(1,-1)

### Calculate Error
sgd_predicted_val_syn = get_predicted_value(phi_syn_t, sgd_weights_syn_t )
print("SGD Solution of Synthetic Training Set : \n", sgd_weights_syn_t)
print("SGD RMS Error of Synthetic Training SET : ", np.sqrt(mean_squared_error(syn_op_train, sgd_predicted_val_syn)))

### Calculating Stochastic Gradient Descent for synthetic testing set
learning_rate = 0.001
minibatch_size =  len(syn_op_test)
num_epochs = 10000
L2_lambda = 0.1 
sgd_weights_syn_te = SGD_sol(learning_rate, minibatch_size, num_epochs, L2_lambda, phi_syn_te, phi_syn_v, syn_op_test, syn_op_valid )

### Fetch predicted value for Synthetic TEsting Set 
sgd_weights_syn_te = np.array(sgd_weights_syn_te).reshape(1,-1)

### Calculate Error
sgd_predicted_val_syn = get_predicted_value(phi_syn_te, sgd_weights_syn_te )
print("SGD Solution of Synthetic Testing Set : \n", sgd_weights_syn_te)
print("SGD RMS Error of Synthetic Testing SET : ", np.sqrt(mean_squared_error(syn_op_test, sgd_predicted_val_syn)))

### Calculating Stochastic Gradient Descent for Letor training set
learning_rate = 0.001
minibatch_size =  len(letor_op_train)
num_epochs = 10000
L2_lambda = 0.1 
sgd_weights_letor_t = SGD_sol(learning_rate, minibatch_size, num_epochs, L2_lambda, phi_letor_t,phi_letor_v, letor_op_train, letor_op_valid)

### Fetch predicted value for Letor Trainig Set 
sgd_weights_letor_t = np.array(sgd_weights_letor_t).reshape(1,-1)

### Calculate Error
sgd_predicted_t_letor = get_predicted_value(phi_letor_t, sgd_weights_letor_t )
print("SGD Solution of Letor Training Set : \n", sgd_weights_letor_t)
print("SGD RMS Error of LETOR Training SET : ", np.sqrt(mean_squared_error(letor_op_train, sgd_predicted_t_letor)))

### Calculating Stochastic Gradient Descent for Letor Testing set
learning_rate = 0.001
minibatch_size =  len(letor_op_test)
num_epochs = 10000
L2_lambda = 0.1 
sgd_weights_letor_te = SGD_sol(learning_rate, minibatch_size, num_epochs, L2_lambda, phi_letor_te,phi_letor_v,  letor_op_test,letor_op_valid)

### Fetch predicted value for Letor Validation Set 
sgd_weights_letor_te = np.array(sgd_weights_letor_te).reshape(1,-1)
sgd_predicted_te_letor = get_predicted_value(phi_letor_te, sgd_weights_letor_te )
print("SGD Solution of Letor Testing Set : \n", sgd_weights_letor_te)
print("SGD RMS Error of LETOR TESTING SET : ", np.sqrt(mean_squared_error(letor_op_test, sgd_predicted_te_letor)))
