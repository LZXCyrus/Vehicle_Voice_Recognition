from utils import *
from hmmlearn import hmm

def train_model_hmm(train_dir):
    """
    :param train_dir: derivative order, default order is 2
    :return: hmm_models
    """
    hmm_models = []
    # iterate through the training directory
    for digit in os.listdir(train_dir):
        # get the directory of digit
        digit_dir = os.path.join(train_dir, digit)
        # get the digit label
        label = digit_dir[digit_dir.rfind('/') + 1:]
        # start training
        X = np.array([])
        train_files = [x for x in os.listdir(digit_dir) if x.endswith('.wav')]
        for file_name in train_files:
            file_path = os.path.join(digit_dir, file_name)
            # get mfcc feature and ignore the warning
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                features_mfcc = mfcc(file_path)
            # append mfcc to X
            if len(X) == 0:
                X = features_mfcc
            else:
                X = np.append(X, features_mfcc, axis=0)
        #557
        # get the hmm model
      
        model = hmm.GMMHMM(n_components=4, covariance_type='diag', n_iter=1800)
        # fit hmm model
        np.seterr(all='ignore')
        model.fit(X)
        #print(model.decode(X))
        hmm_models.append((model, label))
    return hmm_models

from sklearn.metrics import roc_auc_score
def predict_hmm(hmm_models, test_files):
    """
    :param hmm_models: trained hmm models
    :param test_files: test files
    """
    count = 0
    pred_true = 0
    y_pred = []
    y = []
    for test_file in test_files:
        # get mfcc feature and ignore the warning
        #print(test_file)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            features_mfcc = mfcc(test_file)
        # calculate the score and get the maximum score
        max_score = -float('inf')
        predicted_label = ""
        # features_mfcc,h0 = rnn(torch.tensor(features_mfcc))
        # features_mfcc = features_mfcc.detach().numpy()
        for item in hmm_models:
            model, label = item
            score = model.score(features_mfcc)
            if score > max_score:
                max_score = score
                predicted_label = label

        # determine if the prediction is correct
        count += 1
        y_pred.append(int(os.path.splitext(test_file)[0][31]))
        y.append(int(predicted_label[-1]))
        if os.path.splitext(test_file)[0][31] == predicted_label[-1]:
            pred_true += 1
    yp=np.zeros([120,3])
    yy=np.zeros([120,3])
    for i in range(120):
        m=y_pred[i]
        n=y[i]
        yp[i,m]=1
        yy[i,n] = 1
    auc_score1 = roc_auc_score(yp,yy,multi_class='ovo')
    print("---------- HMM (GaussianHMM) ----------")
    print("Train num: 160, Test num: %d, Predict true num: %d"%(count, pred_true))
    print("Accuracy: %.2f"%(pred_true / count))
    print("auc:",auc_score1)


if __name__ == '__main__':
    # train the model
    hmm_models = train_model_hmm("./train_set")

    # append all test records and start digit recognition
    test_files = []
    for root, dirs, files in os.walk("./test_set"):
        for file in files:
            # Make sure the suffix is correct and avoid the influence of hidden files
            if os.path.splitext(file)[1] == '.wav':
                test_files.append(os.path.join(root, file))
    predict_hmm(hmm_models, test_files)
