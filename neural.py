
import numpy as  np
import sklearn.model_selection as sk
import matplotlib.pyplot as plt

X = np.loadtxt('data.txt', delimiter=',')
Y = np.loadtxt('label.txt' ,delimiter=',')
X = np.hstack((np.ones((5000,1)),X))
n,in_dim = np.shape(X)
out_n,out_dim = np.shape(Y)
lamda   = 0.01

def make_batches(x,y):
    m,n = np.shape(y) 
    batches = []
    perm = np.random.permutation(m)
    new_x = x[perm]
    new_y = y[perm]
    for k in range(0,int(m/25)):
        mini_batch_x = new_x[k*25:(k+1)*25,]
        mini_batch_y = new_y[k*25:(k+1)*25,]
        batch = (mini_batch_x, mini_batch_y)
        batches.append(batch)
    if m%25 !=0:
        k = int(m/25)
        mini_batch_x = new_x[k*25:,]
        mini_batch_y = new_y[k*25:,]
        batch = (mini_batch_x, mini_batch_y)
        batches.append(batch)
    return batches

def neural_netwok(x,y,num_hidden,epocs):
    train_err_list = []
    valid_err_list = []
    for trial in range(0,5):
        #w = np.random.randn(in_dim, 500)/np.sqrt(in_dim)
        #v = np.random.randn(500,out_dim)/np.sqrt(500)
        w = np.random.randn(in_dim,500)/np.sqrt(in_dim)
        v = np.random.randn(500,out_dim)/np.sqrt(500)
        b1 = np.zeros((1, 500))
        b2 = np.zeros((1,out_dim))
        train_err = []
        valid_err = []
        for i in range(0, epocs):
            batchSet = make_batches(x,y)
            x,valid_x,y,valid_y = sk.train_test_split(train_x,train_y,train_size=0.8, random_state=1)
            for batch in batchSet:
                mini_x,mini_y = batch
                #----------------------Forward propagation---------------------------------------------------------
                z1 = mini_x.dot(w) + b1
                a1 = np.tanh(z1)
                z2 = a1.dot(v) + b2
                ok = np.exp(z2)
                o = ok / np.sum(ok,axis=1,keepdims=True) #softmax applied
                # --------------------Backpropagation---------------------------------------------------------------
                delta1 = o - mini_y
                dv = (a1.T).dot(delta1)
                db2 = np.sum(delta1,axis=0,keepdims=True)
                delta2 = delta1.dot(v.T) * (1 - np.power(a1, 2))
                dw = np.dot(mini_x.T, delta2)
                db1 = np.sum(delta2, axis=0)
                w -= 0.01*dw
                v -= 0.01*dv
                b1-= 0.01*db1
                b2-= 0.01*db2
            t_err = error(w,v,b1,b2,x,y)
            v_err = error(w,v,b1,b2,valid_x,valid_y)
            train_err.append(t_err)
            valid_err.append(v_err)
        train_err_list.append(train_err)
        valid_err_list.append(valid_err)
    train_err_matrix = np.matrix(train_err_list)
    valid_err_matrix = np.matrix(valid_err_list)
    return w,v,b1,b2,train_err_matrix,valid_err_matrix

def error(w,v,b1,b2,x,y):
    z1 = x.dot(w) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(v) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    ok = np.log(probs)
    err = np.multiply(y,ok)
    return np.sum(-err)/len(x)

train_x,test_x,train_y,test_y = sk.train_test_split(X,Y, train_size=0.8, random_state=1)

#-------------------------------------------training our neural network--------------------------------------------------

w,v,b1,b2,train_err_matrix,valid_err_matrix = neural_netwok(train_x,train_y,500,100)

#------------------------------------------calculating error varinace of train/validate set------------------------------
train_err = []
valid_err = []
train_variance = []
train_mean = []
validate_mean = []
validate_variance = []
epoc = []

for i in range(0,100):
    train_err.append(np.average(train_err_matrix[:,i])) #estimating avg. error for train-set over all 5 trials
    valid_err.append(np.average(valid_err_matrix[:,i])) #estimating avg. error for validate-set over all 5 trials
    
    train_mean.append(np.mean(train_err_matrix[:,i])) #estimating mean for train-set error for mean-var plot
    train_variance.append(np.var(train_err_matrix[:,i])*100) #estimating variance for train-set error for mean-var plot
    
    validate_mean.append(np.mean(valid_err_matrix[:,i])) #estimating mean for train-set error for mean-var plot
    validate_variance.append(np.var(valid_err_matrix[:,i])*100) #estimating variance for train-set error for mean-var plot
    
    epoc.append(i+1) 


#--------------------------------------plotting the training & validation error v/s epocs---------------------------------------#    
for i in range(0,5):
    fig, ax = plt.subplots()
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.tick_params(axis='x',which='minor',bottom='off')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.plot(epoc,np.transpose(train_err_matrix[i]),label = r'Training error')
    ax.plot(epoc,np.transpose(valid_err_matrix[i]),label = r'Validation error')
    ax.set_xlabel(r'epocs',fontsize=15)
    ax.set_ylabel("error",fontsize=15)
    plt.ylim((-0.0001,0.6))
    plt.xlim((0,105))
    ax.set_title('error for varying epocs\ntrial_'+str(i),fontweight= 'bold',fontsize=15)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig("epc_vs_error_"+str(i+1)+".png", bbox_inches='tight')
#--------------------------------------plotting average of all 5 trials error v/s epocs--------------------------------#
fig, ax = plt.subplots()
ax.yaxis.grid(True)
ax.set_axisbelow(True)
ax.minorticks_on()
ax.tick_params(axis='x',which='minor',bottom='off')
ax.yaxis.grid(True)
ax.set_axisbelow(True)
ax.plot(epoc,train_err,label = r'Training error')
ax.plot(epoc,valid_err,label = r'Validation error')
ax.set_xlabel(r'epocs',fontsize=15)
ax.set_ylabel("error",fontsize=15)
plt.ylim((-0.0001,0.6))
plt.xlim((0,105))
ax.set_title('avg. of all 5 trials error_vs_epocs',fontweight= 'bold',fontsize=15)
ax.legend(loc="best")
plt.tight_layout()
plt.savefig("epc_vs_error.png", bbox_inches='tight')

#------------------------------------plotting mean error and variance of train set-------------------------------------------------------------------#
    
fig, ax = plt.subplots()
ax.yaxis.grid(True)
ax.set_axisbelow(True)
ax.minorticks_on()
ax.tick_params(axis='x',which='minor',bottom='off')
ax.yaxis.grid(True)
ax.set_axisbelow(True)
ax.plot(epoc,train_mean,label = r'Training error')
ax.errorbar(epoc,train_mean,train_variance,linestyle='None',marker='x')
ax.set_xlabel(r'epocs',fontsize=15)
ax.set_ylabel("error",fontsize=15)
plt.ylim((0.0,0.4))
plt.xlim((0,101))
ax.set_title('mean-variance of training set',fontweight= 'bold',fontsize=15)
ax.legend(loc="best")
plt.tight_layout()
plt.savefig("Train_mean_variance.png", bbox_inches='tight')

#------------------------------------plotting mean error and variance of validation set-----------------------------------------
fig, ax = plt.subplots()
ax.yaxis.grid(True)
ax.set_axisbelow(True)
ax.minorticks_on()
ax.tick_params(axis='x',which='minor',bottom='off')
ax.yaxis.grid(True)
ax.set_axisbelow(True)
ax.plot(epoc,validate_mean,label = r'Validation error')
ax.errorbar(epoc,validate_mean,validate_variance,linestyle='None',marker='x')
ax.set_xlabel(r'epocs',fontsize=15)
ax.set_ylabel("error",fontsize=15)
plt.ylim((0.28,0.35))
plt.xlim((0,101))
ax.set_title('mean-variance of validation set',fontweight= 'bold',fontsize=15)
ax.legend(loc="best")
plt.tight_layout()
plt.savefig("Validation_mean_variance.png", bbox_inches='tight')
    





