from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import matplotlib.pyplot as plt
#100 exemples, 4 features, 3 classes

class SoftmaxClassifier(BaseEstimator, ClassifierMixin):  
    """A softmax classifier"""

    def __init__(self, lr = 0.1, alpha = 100, n_epochs = 1000, eps = 1.0e-5,threshold = 1.0e-10 , regularization = True, early_stopping = True):
       
        """
            self.lr : the learning rate for weights update during gradient descent
            self.alpha: the regularization coefficient 
            self.n_epochs: the number of iterations
            self.eps: the threshold to keep probabilities in range [self.eps;1.-self.eps]
            self.regularization: Enables the regularization, help to prevent overfitting
            self.threshold: Used for early stopping, if the difference between losses during 
                            two consecutive epochs is lower than self.threshold, then we stop the algorithm
            self.early_stopping: enables early stopping to prevent overfitting
        """

        self.lr = lr 
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.eps = eps
        self.regularization = regularization
        self.threshold = threshold
        self.early_stopping = early_stopping
        self.nb_classes = 3

    """
        Public methods, can be called by the user
        To create a custom estimator in sklearn, we need to define the following methods:
        * fit
        * predict
        * predict_proba
        * fit_predict        
        * score
    """


    """
        In:
        X : the set of examples of shape nb_example * self.nb_features
        y: the target classes of shape nb_example *  1

        Do:
        Initialize model parameters: self.theta_
        Create X_bias i.e. add a column of 1. to X , for the bias term
        For each epoch
            compute the probabilities
            compute the loss
            compute the gradient
            update the weights
            store the loss
        Test for early stopping

        Out:
        self, in sklearn the fit method returns the object itself


    """

    def fit(self, X, y=None):
    
        prev_loss = np.inf
        self.losses_ = []

        self.nb_feature = X.shape[1]
        self.nb_classes = len(np.unique(y))
        m =X.shape[0]  
        bias = np.ones((m,1))
        X_bias = np.concatenate((bias,X),1)
        self.theta_  = np.random.rand(X_bias.shape[1],self.nb_classes)
#        print(self.theta_)
        for epoch in range( self.n_epochs):
#            logits = np.dot(X_bias,self.theta_)
#            probabilities = self._softmax(logits)
            probabilities = self.predict_proba(X,y)
            loss = self._cost_function(probabilities,y) 
            self.theta_ = self.theta_ - self.lr * self._get_gradient(X_bias,y,probabilities) 
            self.losses_.append(loss)
            self.early_stopping = abs(self.losses_[epoch] - prev_loss) <= self.threshold
            if self.early_stopping:
                break
            prev_loss = loss
        plt.plot(self.losses_)
        return self

    

   
    

    """
        In: 
        X without bias

        Do:
        Add bias term to X
        Compute the logits for X
        Compute the probabilities using softmax

        Out:
        Predicted probabilities
    """

    def predict_proba(self, X, y=None):
        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        m = X.shape[0]  
        bias = np.ones((m,1))
        X_bias = np.concatenate((bias,X),1)
        Z = np.dot(X_bias,self.theta_)
        y = self._softmax(Z)
        return y


        """
        In: 
        X without bias

        Do:
        Add bias term to X
        Compute the logits for X
        Compute the probabilities using softmax
        Predict the classes

        Out:
        Predicted classes
    """

    
    def predict(self, X, y=None):
        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        y = self.predict_proba(X,y)
        return np.argmax(y,axis = 1)
    

    

    def fit_predict(self, X, y=None):
            
        self.fit(X, y)
        return self.predict(X,y)


    """
        In : 
        X set of examples (without bias term)
        y the true labels

        Do:
            predict probabilities for X
            Compute the log loss without the regularization term

        Out:
        log loss between prediction and true labels

    """    

    def score(self, X, y=None):
        
        self.regularization = False
        y_chap = self.predict_proba(X)
        " %%% retablir sur true ? ICI OU DANS LE FIT"
        #self.regularization = True 
        " true labels = colonne avant one hot?"
        return self._cost_function(y_chap,y)
    

    """
        Private methods, their names begin with an underscore
    """

    """
        In :
        y without one hot encoding
        probabilities computed with softmax %%%computed with predict proba

        Do:
        One-hot encode y
        Ensure that probabilities are not equal to either 0. or 1. using self.eps
        Compute log_loss
        If self.regularization, compute l2 regularization term
        Ensure that probabilities are not equal to either 0. or 1. using self.eps

        Out:
       cost:real number
    """
    #https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
    def _cost_function(self,probabilities, y ): 
        yohe = self._one_hot(y)

        probabilities_adapted = np.maximum(self.eps, np.minimum(1-self.eps, probabilities))
        probabilities_adapted = np.log(probabilities_adapted)
        ce_cost = (-1) * np.mean(np.sum(yohe * probabilities_adapted, axis = 1))
        L2_cost = self.alpha * np.sum(np.square(self.theta_[1:self.nb_feature +1,:]))
        if self.regularization:
            return ce_cost + L2_cost
        else:
            return ce_cost 
        


    """
        In :
        Target y: nb_examples * 1

        Do:
        One hot-encode y
        [1,1,2,3,1] --> [[1,0,0],
                         [1,0,0],
                         [0,1,0],
                         [0,0,1],
                         [1,0,0]]
        Out:
        y one-hot encoded
    """

    
    def _one_hot(self,y):
        temp = np.unique(y)
        oneHot = np.zeros((len(y), self.nb_classes))
        j=0
        for i in y:
            index = list(temp).index(i)
            oneHot[j][index]=1
            j += 1
        return oneHot
    
    def _one_hotNOUS(self,y):
                
        size = np.shape(y)
        yohe = np.zeros((size[0],self.nb_classes),dtype = int)  
        for i in range (0,size[0]):
            yohe[i, y[i]] = 1
        return yohe
    
                
    """
        In :
        Logits: (nb_examples) * self.nb_classes

        Do:
        Compute softmax on logits

        Out:
        Probabilities
    """
    
    def _softmax(self,z):
#        z = np.subtract(z.T, np.max(z,axis = 1)).T
        A = np.exp(z)
        sum_vector = np.sum(A, 1) #N vecteur ligne et verifier que somme vecvteur vaut bien 1
        p_chap = A/sum_vector[:,None] #chaque ligne i de la matrice exp est divise par le terme i du vecteur sum
        return p_chap
    

    """
        In:
        X with bias
        y without one hot encoding
        probabilities resulting of the softmax step (matrix m x k)

        Do:
        One-hot encode y
        Compute gradients
        If self.regularization add l2 regularization term

        Out:
        Gradient

    """

    def _get_gradient(self,X,y, probas):

        yohe = self._one_hot(y)
        m = y.shape[0] 
        if self.regularization:
            theta_bis = self.theta_[1:]
            zero = np.zeros((1,self.nb_classes))
            bis = np.concatenate((zero,theta_bis),0)

#            print(theta_bis)
            return (1/m) * np.dot(X.T , (probas - yohe)) + (1/m)* 2 * self.alpha * bis
#            A = (1/m) * np.dot(X.T , (probas - yohe))
#            A[1:-1,:] = A[1:-1,:] + (1/m)* 2 * self.alpha * (self.theta_[1:-1,:])
#            return A
        else:
            return (1/m) * np.dot(X.T ,(probas - yohe))
        

    

def main():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    
    # load dataset
    data,target =load_iris().data,load_iris().target
    
    # split data in train/test sets
    X_train, X_test, y_train, y_test = train_test_split( data, target, test_size=0.33, random_state=42)

    cl = SoftmaxClassifier()

   # train on X_train and not on X_test to avoid overfitting
    train_p = cl.fit_predict(X_train,y_train)
    test_p = cl.predict(X_test)
    
#    A = [1,0,2,2,1,2,1]
#    size = np.shape(A)
#    print(size)
#    print(size[0])
#    print(cl._one_hot(A))
    from sklearn.metrics import precision_recall_fscore_support

    #display precision, recall and f1-score on train/test set
    print("train : "+ str(precision_recall_fscore_support(y_train, train_p,average = "macro")))
    print("test : "+ str(precision_recall_fscore_support(y_test, test_p,average = "macro")))
    
if __name__ == '__main__':
    main()
    