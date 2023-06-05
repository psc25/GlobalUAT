import numpy as np
import tensorflow as tf
import iisignature as iisig
import matplotlib.pyplot as plt
import time

pltname = "learn_avgint"

M = 50000
K = 100
d = 1
T = 1.0
dt = T/K

dX = np.random.normal(size = [M, K-1, 1])*np.sqrt(dt)
X = np.zeros([M, K, 1])
X[:, 1:, :] = np.cumsum(dX, axis = 1)
tt = np.linspace(0, T, K)
time_var = np.tile(np.reshape(tt, (1, -1, 1)), [M, 1, 1])

Y = np.zeros_like(X)
for k in range(1, K):
    Y[:, k] = np.fmax(np.sum(X[:, :k], axis = 1)*dt/tt[k], -0.3)

ep = 2000
eval_every = 100
val_split = 0.2
Mtrain = int((1-val_split)*M)
lr = 1e-5
batch_size = 500
nr_batch = int(Mtrain/batch_size)
r = 2.0

ind_train = np.arange(Mtrain)
ind_test = np.arange(Mtrain, M)
ind_plot = np.random.choice(ind_test, 3, replace = False)
col = ['b', 'g', 'r']

fig = plt.figure()
for i in range(len(ind_plot)):
    plt.plot(tt, X[ind_plot[i]], c = col[i], ls = ":", alpha = 0.6)
    plt.plot(tt, Y[ind_plot[i]], c = col[i], ls = "-", alpha = 0.6)
    
plt.show()

# Train FNN
b_FNN = time.time()
loss_FNN = np.nan*np.ones([ep, 2])
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():         
        init = tf.random_normal_initializer(mean = 0.0, stddev = 0.001)      
        
        input1 = tf.placeholder(shape = (None, K, d), dtype = tf.float32)
        inp_time = tf.placeholder(shape = (None, K, 1), dtype = tf.float32)
        
        N = 30
        N1 = 20
        W = tf.Variable(initial_value = init(shape = [1, 1, N]), name = "W", dtype = tf.float32)
        A = tf.Variable(initial_value = init(shape = [1, 1, N]), name = "A", dtype = tf.float32)
        b = tf.Variable(initial_value = init(shape = [1, 1, N]), name = "b", dtype = tf.float32)
        
        V = tf.Variable(initial_value = init([N, d, N1], dtype = tf.float32), name = "V", dtype = tf.float32)
        U = tf.Variable(initial_value = init([1, 1, N, d, N1], dtype = tf.float32), name = "U", dtype = tf.float32)
        c = tf.Variable(initial_value = init([1, 1, N, d, N1], dtype = tf.float32), name = "c", dtype = tf.float32)
        
        W1_hidden = tf.tensordot(inp_time[:, :, 0], V, axes = 0) + c
        W1_output = tf.reduce_sum(U*tf.nn.relu(W1_hidden), -1)
        W1_intgnd = W1_output*tf.expand_dims(input1, -2)
        inp_int = tf.cumsum(W1_intgnd*dt, axis = 1)
        
        hidden1 = tf.reduce_sum(inp_int, -1) + A*inp_time + b
        y_pred = tf.reduce_sum(W*tf.nn.relu(hidden1), -1, keepdims = True)
        y_pred = y_pred - y_pred[:, 0:1]
        
        y_true = tf.placeholder(shape = (None, K, 1), dtype = tf.float32)
        loss = tf.reduce_mean(tf.pow(tf.abs(y_pred - y_true), r))
        
        optimizer = tf.train.AdamOptimizer(learning_rate = lr)
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars)
        
        sess.run(tf.global_variables_initializer())        
                
        print("\nTraining steps:")
        for i in range(ep):
            begin = time.time()
            ind_rand = np.random.permutation(Mtrain)
            loss_batch = np.zeros([nr_batch, 1])
            for l in range(nr_batch):
                ind_batch = ind_rand[l:l+batch_size]
                feed_dict = {input1: X[ind_batch], inp_time: time_var[ind_batch], y_true: Y[ind_batch]}
                _, loss_batch[l] = sess.run([train_op, loss], feed_dict)
                
            loss_FNN[i, 0] = np.mean(loss_batch)
            end = time.time()
            print("Step {}, Time {}s, Loss {:g}".format(i+1, round(end-begin, 1), loss_FNN[i, 0]))
            
            if val_split > 0 and (i+1) % eval_every == 0:
                print("\nEvaluation on test data:")
                feed_dict = {input1: X[ind_test], inp_time: time_var[ind_test], y_true: Y[ind_test]}
                loss_FNN[i, 1] = sess.run(loss, feed_dict)
                end = time.time()
                print("Step {}, Time {}s, Loss {:g}".format(i+1, round(end-begin, 1), loss_FNN[i, 1]))
                print("")
                
    feed_dict = {input1: X[ind_plot], inp_time: time_var[ind_plot]}
    Y_FNN = sess.run(y_pred, feed_dict)
    
e_FNN = time.time()

# Learn Signature
b_Sig = time.time()
N = 7
N1 = np.sum(np.power(2, np.arange(N+1)))-1
Sig = np.zeros([M, K, N1])
tt1 = np.reshape(tt, (-1, 1))
for m in range(M):
    print("Compute Signature m = " + str(m+1) + "/" + str(M))
    time_ext = np.concatenate((tt1, X[m]), axis = -1)
    Sig[m, 1:] = iisig.sig(time_ext, N, 2)
    
loss_Sig = np.nan*np.ones([ep, 2])
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():         
        init = tf.random_normal_initializer(mean = 0.0, stddev = 0.001)      
        
        input1 = tf.placeholder(shape = (None, K, N1), dtype = tf.float32)
        L = tf.Variable(initial_value = init(shape = [1, 1, N1]), name = "L", dtype = tf.float32)
        y_pred = tf.reduce_sum(L*input1, axis = -1, keepdims = True)
        
        y_true = tf.placeholder(shape = (None, K, 1), dtype = tf.float32)
        loss = tf.reduce_mean(tf.pow(tf.abs(y_pred - y_true), r))
        
        optimizer = tf.train.AdamOptimizer(learning_rate = lr)
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars)
        
        sess.run(tf.global_variables_initializer())        
        
        print("\nTraining steps:")
        for i in range(ep):
            begin = time.time()
            ind_rand = np.random.permutation(Mtrain)
            loss_batch = np.zeros([nr_batch, 1])
            for l in range(nr_batch):
                ind_batch = ind_rand[l:l+batch_size]
                feed_dict = {input1: Sig[ind_batch], y_true: Y[ind_batch]}
                _, loss_batch[l] = sess.run([train_op, loss], feed_dict)
                
            loss_Sig[i, 0] = np.mean(loss_batch)
            end = time.time()
            print("Step {}, Time {}s, Loss {:g}".format(i+1, round(end-begin, 1), loss_Sig[i, 0]))
            
            if val_split > 0 and (i+1) % eval_every == 0 and i > 0:
                print("\nEvaluation on test data:")
                feed_dict = {input1: Sig[ind_test], y_true: Y[ind_test]}
                loss_Sig[i, 1] = sess.run(loss, feed_dict)
                end = time.time()
                print("Step {}, Time {}s, Loss {:g}".format(i+1, round(end-begin, 1), loss_Sig[i, 1]))
                print("")
    
    feed_dict = {input1: Sig[ind_plot]}
    Y_Sig = sess.run(y_pred, feed_dict)
    
e_Sig = time.time()

# Plot Results
fig = plt.figure()
for i in range(len(ind_plot)):
    plt.plot(tt, X[ind_plot[i]], c = col[i], ls = ":", alpha = 0.6)
    plt.plot(tt, Y[ind_plot[i]], c = col[i], ls = "-", alpha = 0.6)
    plt.plot(tt, Y_FNN[i], c = col[i], ls = "--", alpha = 0.6)
    plt.plot(tt, Y_Sig[i], c = col[i], ls = "-.", alpha = 0.6)
    
plt.plot(tt, np.nan*np.ones(K), c = 'k', ls = ":", label = r'$x(t)$')
plt.plot(tt, np.nan*np.ones(K), c = 'k', ls = "-", label = r'$f_2(t,x)$')
plt.plot(tt, np.nan*np.ones(K), c = 'k', ls = "--", label = 'FNN')
plt.plot(tt, np.nan*np.ones(K), c = 'k', ls = "-.", label = 'Sig')
plt.legend(loc = "upper left", ncol = 4)
plt.xlabel("Time")
plt.savefig(pltname + "_result_" + str(int(r)) + ".png", bbox_inches = 'tight', dpi = 400)
plt.close(fig)

# Plot Loss
fig = plt.figure()
res2 = np.nan*np.ones(ep)
res2[np.arange(eval_every-1, ep, eval_every)] = 1.0
plt.plot(range(ep), loss_FNN[:, 0], c = "silver", ls = "-", label = "FNN Train")
plt.plot(range(ep), loss_FNN[:, 1], c = "silver", ls = "None", markeredgecolor = "k", markeredgewidth = 0.8, marker = "o", label = "FNN Test")
plt.plot(range(ep), loss_Sig[:, 0], c = "dimgray", ls = "-", label = "Sig Train")
plt.plot(range(ep), loss_Sig[:, 1], c = "dimgray", ls = "None", markeredgecolor = "k", markeredgewidth = 0.8, marker = "o", label = "Sig Test")
plt.legend(loc = "upper right", ncol = 2)
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.savefig(pltname + "_training_" + str(int(r)) + ".png", bbox_inches = 'tight', dpi = 400) 
plt.close(fig)

# Print Times
print("Time for FNN: " + str(np.round(e_FNN-b_FNN, 2)))
print("Time for Sig: " + str(np.round(e_Sig-b_Sig, 2)))