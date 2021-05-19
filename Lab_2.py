import numpy as np
import string
from collections import Counter


np.random.seed(1)

# Problem 1

def count_frequency(documents):

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    lower_case_doc = []
    for s in documents:
      lower_case_doc.append(s.lower())

    no_punc_doc = []
    no_punc_doc = [''.join(c for c in s if c not in string.punctuation) for s in lower_case_doc]

    words_doc = []
    for i in no_punc_doc:
      word = i.split(" ")
      words_doc.append(word)

    all_words = []
    for i in words_doc:
      all_words.extend(i)

    frequency = Counter()
    for word in all_words:
      frequency[word] +=1
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return frequency

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
def prior_prob(y_train):
    unique, counts = np.unique(y_train, return_counts=True)
    prior = dict(zip(unique, counts))
    for i in prior:
        prior[i] = float(prior[i]/y_train.size)

    return prior


def conditional_prob(X_train, y_train):
    alpha = 1.0
    N = 20000

    lower_case_doc = []
    for s in X_train:
      lower_case_doc.append(s.lower())

    no_punc_doc = []
    no_punc_doc = [''.join(c for c in s if c not in string.punctuation) for s in lower_case_doc]

    words_doc = []
    for i in no_punc_doc:
      word = i.split(" ")
      words_doc.append(word)

    countArray = []
    for i in words_doc:
      lengths = (len(i))
      countArray.append(lengths)

    numberSpamWords = 0
    numberNonSpamWord = 0

    for i in range(len(y_train)):
      if y_train[i] == 1:
        numberSpamWords+= countArray[i]
        #print(countArray[i])
      else:
        numberNonSpamWord+= countArray[i]

    #print(x_train[1].size)

    SpamMessageArray = []
    NonSpamMessageArray = []
    for i in range(len(y_train)):
      if y_train[i] == 1:
        SpamMessageArray.append(X_train[i])
        #print(countArray[i])
      else:
        NonSpamMessageArray.append(X_train[i])


    denomSpam = numberSpamWords + (N*alpha)
    denomNonSpam = numberNonSpamWord + (N*alpha)

    CounterSpam = count_frequency(SpamMessageArray)
    CS = dict(CounterSpam)
    CS2 = {}
    for k, v in CS.items():
      CS2[k] = (v + alpha)/denomSpam

    CounterNonSpam = count_frequency(NonSpamMessageArray)
    CNS = dict(CounterNonSpam)
    CNS2 = {}
    for k, v in CNS.items():
      CNS2[k] = (v + alpha)/denomNonSpam

    #print(C)
    cond_prob = dict( {0: CNS2, 1: CS2})


    return cond_prob

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


def predict_label(X_test, prior_prob, cond_prob):

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    a = 1
    N = 20000
    predict = np.array([])
    test_prob = np.zeros((X_test.shape[0], len(prior_prob)))

    for label in prior_prob:
        for idx, example in enumerate(X_test):
            sum_probs = 0
            words_ex = example.lower().translate(str.maketrans('', '', string.punctuation)).split()
        #print('freqs', count_frequency(X_test))
            for word in words_ex:
                if word not in cond_prob[label].keys():
            # Used if we added a dummy word
            #word = ''
            #Instead of the dummy word, the prob for a new word is defined as a/(N*a). This value gives the same output as the unt test case
                    sum_probs = sum_probs + np.log(a / (N*a))
                else:
                    sum_probs = sum_probs + np.log(cond_prob[label][word])
          #sum_probs = sum_probs + np.log(cond_prob[label][word])
            g = np.log(prior_prob[label]) + sum_probs
            test_prob[idx][label] = g

    m = np.amax(test_prob, axis=1)
    predict = np.argmax(test_prob, axis=1)
    #test_prob = np.exp(test_prob-m)/np.sum(np.exp(test_prob-m), axis=1)
    # We have to calculate transpose of m and summation so we can subtract it
    test_prob = np.subtract(test_prob, m[:,None])
    test_prob = np.exp(test_prob)
    summation = np.sum(test_prob, axis=1)
    test_prob = test_prob/summation[:, None]


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return predict, test_prob

def compute_test_prob(word_count, prior_cat, cond_cat):

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    sum_probs = 0
    a = 1
    N = 20000
    for word in word_count:
        if word not in cond_cat:
            sum_probs = sum_probs + (word_count[word]*np.log(a / (N*a)))
        else:
            sum_probs = sum_probs + (word_count[word]*np.log(cond_cat[word]))
    prob = np.log(prior_cat) + sum_probs
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return prob

def compute_metrics(y_pred, y_true):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    acc = accuracy_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred)
    cm = confusion_matrix(y_true,y_pred)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return acc, cm, f1


# Problem 2

def featureNormalization(X):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    X_mean = []
    rowN = X.shape[0]
    colN = X.shape[1]
    Y = X.T
    X_normalized = np.zeros((rowN,colN))
    #print(X_normalized.shape)
    X_std = []
    for i in Y:
        meanA = np.mean(i)
        X_mean.append(meanA)

    for i in Y:
        SD = np.std(i)
        X_std.append(SD)

    for i in range(X.shape[0]):
      for j in range(X.shape[1]):
        X_normalized[i][j] = (X[i][j] - X_mean[j])/X_std[j]


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return X_normalized, X_mean, X_std


def applyNormalization(X, X_mean, X_std):

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    rowN = X.shape[0]
    colN = X.shape[1]
    X_normalized = np.zeros((rowN,colN))

    for i in range(X.shape[0]):
      for j in range(X.shape[1]):
        X_normalized[i][j] = (X[i][j] - X_mean[j])/X_std[j]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return X_normalized

def computeMSE(X, y, theta):

  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  error = 0
  num_samples = X.shape[0]
  #print(num_samples)
  sum = 0
  theta = theta.T
  for i in range(num_samples):
    dotProductResult = (np.dot(theta,X[i]) - y[i])**2
    print(dotProductResult)
    sum += dotProductResult
  error = ((0.5)*(1/num_samples)) * sum
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return error[0]

def computeGradient(X, y, theta):

  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  gradient = []
  Xtranspose_theta = np.dot(X,theta)

  firstPart = Xtranspose_theta*X
  sum = np.sum(firstPart, axis = 0)
  secondPart = np.dot(X.T,y)

  gradient2 = (sum-secondPart)/X.shape[0]
  gradient = np.reshape(gradient2,(X.shape[1],1))

  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

  return gradient

def gradientDescent(X, y, theta, alpha, num_iters):

  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  Loss_record = []
  gradient = []
  for i in range(num_iters):
      theta = theta - alpha*computeGradient(X,y,theta)
      result = computeMSE(X, y, theta)
      Loss_record.append(result)
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return theta, Loss_record

def closeForm(X, y):

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  theta = []
  XT = X.T
  XTX=XT@X
  pinv=np.linalg.pinv(XTX)

  theta1 = (pinv@XT)@y
  theta = np.reshape(theta1,(-1,1))
# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return theta
