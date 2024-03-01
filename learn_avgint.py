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

alpha = 0.4
p = 2.1
beta = 0.01
gamma = 3.0

dX = np.random.normal(size = [M, K-1, 1]).astype(np.float32)*np.sqrt(dt)
X = np.zeros([M, K, 1], dtype = np.float32)
X[:, 1:, :] = np.cumsum(dX, axis = 1)
tt = np.linspace(0, T, K, dtype = np.float32)
time_var = np.tile(np.reshape(tt, (1, -1, 1)), [M, 1, 1])

Y = np.zeros_like(X)
for k in range(1, K):
    Y[:, k] = np.fmax(np.sum(X[:, :k], axis = 1)*dt/tt[k], -0.3)
    
# Compute Weight
X1 = np.transpose(X, [0, 2, 1])
time_var1 = np.transpose(time_var, [0, 2, 1])
hoelder = np.nanmax(np.abs(X-X1)/np.power(np.abs(time_var-time_var1), alpha), axis = (1, 2))
wght_FNN = np.exp(beta*np.power(X[:, 0, 0] + hoelder, gamma))

ep = 4000
eval_every = 200
val_split = 0.2
Mtrain = int((1-val_split)*M)
lr = 1e-5
batch_size = 500

ind_train = np.arange(Mtrain)
ind_test = np.arange(Mtrain, M)
ind_plot = np.random.choice(ind_test, 3, replace = False)
col = ['b', 'g', 'r']

fig = plt.figure()
for i in range(len(ind_plot)):
    plt.plot(tt, X[ind_plot[i]], c = col[i], ls = ":", alpha = 0.6)
    plt.plot(tt, Y[ind_plot[i]], c = col[i], ls = "-", alpha = 0.6)
    
plt.show()

# Learn FNN
N = 40
N1 = 30
init = tf.random_normal_initializer(mean = 0.0, stddev = 0.001)
optimizer = tf.optimizers.Adam(learning_rate = lr)

W = tf.Variable(initial_value = init(shape = [1, 1, N]))
A = tf.Variable(initial_value = init(shape = [1, 1, N]))
b = tf.Variable(initial_value = init(shape = [1, 1, N]))
V = tf.Variable(initial_value = init([N, d, N1]))
U = tf.Variable(initial_value = init([1, 1, N, d, N1]))
c = tf.Variable(initial_value = init([1, 1, N, d, N1]))
tr_vars_FNN = [W, A, b, V, U, c]

def FNN(inp_time, inp_x):
    W1_hidden = tf.tensordot(inp_time[:, :, 0], V, axes = 0) + c
    W1_output = tf.reduce_sum(U*tf.nn.relu(W1_hidden), -1)
    W1_intgnd = W1_output*tf.expand_dims(inp_x, -2)
    inp_int = tf.cumsum(W1_intgnd*dt, axis = 1)
    hidden1 = tf.reduce_sum(inp_int, -1) + A*inp_time + b
    y_pred = tf.reduce_sum(W*tf.nn.relu(hidden1), -1, keepdims = True)    
    return y_pred - y_pred[:, 0:1]

b_FNN = time.time()
loss_FNN = np.nan*np.ones([ep, 2])
print("\nTraining steps:")
for i in range(ep):
    begin = time.time()
    nr_batch = int(Mtrain/batch_size)
    ind_rand = np.random.permutation(Mtrain)
    loss_batch = np.zeros([nr_batch, 1])
    for l in range(nr_batch):
        with tf.GradientTape() as tape:
            y_pred = FNN(time_var[ind_rand[l:l+batch_size]], X[ind_rand[l:l+batch_size]])
            y_true = Y[ind_rand[l:l+batch_size]]
            wght_b = wght_FNN[ind_rand[l:l+batch_size]]
            loss = tf.reduce_mean(tf.square((y_pred - y_true)/wght_b))
            
        grad = tape.gradient(loss, tr_vars_FNN)
        optimizer.apply_gradients(zip(grad, tr_vars_FNN))
        loss_batch[l] = loss.numpy()
        
    loss_FNN[i, 0] = np.sqrt(np.mean(loss_batch))
    end = time.time()
    print("Step {}, Time {}s, Loss {:g}".format(i+1, round(end-begin, 1), loss_FNN[i, 0]))
    
    if val_split > 0 and (i+1) % eval_every == 0:
        begin = time.time()
        print("\nEvaluation on test data:")
        nr_batch = int((M-Mtrain)/batch_size)
        ind_rand = np.random.permutation(np.arange(Mtrain, M))
        loss_batch = np.zeros([nr_batch, 1])
        for l in range(nr_batch):
            y_pred = FNN(time_var[ind_rand[l:l+batch_size]], X[ind_rand[l:l+batch_size]])
            y_true = Y[ind_rand[l:l+batch_size]]
            wght_b = wght_FNN[ind_rand[l:l+batch_size]]
            loss = tf.reduce_mean(tf.square((y_pred - y_true)/wght_b))
            loss_batch[l] = loss.numpy()
            
        loss_FNN[i, 1] = np.sqrt(np.mean(loss_batch))
        end = time.time()
        print("Step {}, Time {}s, Loss {:g}".format(i+1, round(end-begin, 1), loss_FNN[i, 1]))
        print("")
        
Y_FNN = FNN(time_var[ind_plot], X[ind_plot]).numpy()
e_FNN = time.time()

# Learn Signature
b_Sig = time.time()
N = 7
N1 = np.sum(np.power(2, np.arange(N+1)))-1
Xhat = np.zeros([M, K, 2])
Sig = np.zeros([M, K, N1])
tt1 = np.reshape(tt, (-1, 1))
for m in range(M):
    print("Compute Signature m = " + str(m+1) + "/" + str(M))
    Xhat[m] = np.concatenate((tt1, X[m]), axis = -1)
    Sig[m, 1:] = iisig.sig(Xhat[m], N, 2)
    
# Compute Weight
Xhat0 = np.expand_dims(Xhat, axis = 1)
Xhat1 = np.transpose(Xhat0, [0, 2, 1, 3])
time_var1 = np.transpose(time_var, [0, 2, 1])
hoelder = np.nanmax(np.linalg.norm(Xhat0-Xhat1, axis = -1)/np.power(np.abs(time_var-time_var1), alpha), axis = (1, 2))
pvar = np.power(np.sum(np.power(np.linalg.norm(Xhat0[:, 0, 1:]-Xhat0[:, 0, :-1], axis = -1), p), axis = 1), 1/p)
wght_Sig = np.exp(beta*np.power(X[:, 0, 0] + hoelder + pvar, gamma))

L = tf.Variable(initial_value = init(shape = [1, 1, N1]))
tr_vars_Sig = [L]

def LinearFct(input_Sig):
    y_pred = tf.reduce_sum(L*input_Sig, axis = -1, keepdims = True)
    return y_pred
    
b_Sig = time.time()
loss_Sig = np.nan*np.ones([ep, 2])
print("\nTraining steps:")
for i in range(ep):
    begin = time.time()
    nr_batch = int(Mtrain/batch_size)
    ind_rand = np.random.permutation(Mtrain)
    loss_batch = np.zeros([nr_batch, 1])
    for l in range(nr_batch):
        with tf.GradientTape() as tape:
            y_pred = LinearFct(Sig[ind_rand[l:l+batch_size]])
            y_true = Y[ind_rand[l:l+batch_size]]
            wght_b = wght_Sig[ind_rand[l:l+batch_size]]
            loss = tf.reduce_mean(tf.square((y_pred - y_true)/wght_b))
            
        grad = tape.gradient(loss, tr_vars_Sig)
        optimizer.apply_gradients(zip(grad, tr_vars_Sig))
        loss_batch[l] = loss.numpy()
        
    loss_Sig[i, 0] = np.sqrt(np.mean(loss_batch))
    end = time.time()
    print("Step {}, Time {}s, Loss {:g}".format(i+1, round(end-begin, 1), loss_Sig[i, 0]))
    
    if val_split > 0 and (i+1) % eval_every == 0:
        begin = time.time()
        print("\nEvaluation on test data:")
        nr_batch = int((M-Mtrain)/batch_size)
        ind_rand = np.random.permutation(np.arange(Mtrain, M))
        loss_batch = np.zeros([nr_batch, 1])
        for l in range(nr_batch):
            y_pred = LinearFct(Sig[ind_rand[l:l+batch_size]])
            y_true = Y[ind_rand[l:l+batch_size]]
            wght_b = wght_Sig[ind_rand[l:l+batch_size]]
            loss = tf.reduce_mean(tf.square((y_pred - y_true)/wght_b))
            loss_batch[l] = loss.numpy()
            
        loss_Sig[i, 1] = np.sqrt(np.mean(loss_batch))
        end = time.time()
        print("Step {}, Time {}s, Loss {:g}".format(i+1, round(end-begin, 1), loss_Sig[i, 1]))
        print("")
    
Y_Sig = LinearFct(Sig[ind_plot]).numpy()
e_Sig = time.time()

# Plot Results
fig = plt.figure()
for i in range(len(ind_plot)):
    plt.plot(tt, X[ind_plot[i]], c = col[i], ls = ":", alpha = 0.6)
    plt.plot(tt, Y[ind_plot[i]], c = col[i], ls = "-", alpha = 0.6)
    plt.plot(tt, Y_FNN[i], c = col[i], ls = "--", alpha = 0.6)
    plt.plot(tt, Y_Sig[i], c = col[i], ls = "-.", alpha = 0.6)
    
plt.plot(tt, np.nan*np.ones(K), c = 'k', ls = ":", label = r'$x(t)$')
plt.plot(tt, np.nan*np.ones(K), c = 'k', ls = "-", label = r'$f_1(t,x)$')
plt.plot(tt, np.nan*np.ones(K), c = 'k', ls = "--", label = 'FNN')
plt.plot(tt, np.nan*np.ones(K), c = 'k', ls = "-.", label = 'Sig')
plt.legend(loc = "upper left", ncol = 4)
plt.xlabel("Time")
plt.savefig(pltname + "_result.png", bbox_inches = 'tight', dpi = 400)
plt.close(fig)

# Plot Loss
fig = plt.figure()
res2 = np.nan*np.ones(ep)
res2[np.arange(eval_every-1, ep, eval_every)] = 1.0
plt.plot(range(ep), loss_FNN[:, 0], c = "silver", ls = "-", label = "FNN Train")
plt.plot(range(ep), loss_FNN[:, 1], ls = "None", marker = "o", markerfacecolor = "None", markeredgecolor = "k", label = "FNN Test")
plt.plot(range(ep), loss_Sig[:, 0], c = "dimgray", ls = "-", label = "Sig Train")
plt.plot(range(ep), loss_Sig[:, 1], ls = "None", marker = "x", markerfacecolor = "None", markeredgecolor = "k", label = "Sig Test")
plt.legend(loc = "upper right", ncol = 2)
plt.xlabel("Epochs")
plt.ylabel("Weighted MSE")
plt.savefig(pltname + "_training.png", bbox_inches = 'tight', dpi = 400) 
plt.close(fig)

# Print Times
print("Time for FNN: " + str(np.round(e_FNN-b_FNN, 2)))
print("Time for Sig: " + str(np.round(e_Sig-b_Sig, 2)))