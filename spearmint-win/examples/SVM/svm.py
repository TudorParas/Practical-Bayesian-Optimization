from data import get_data
import time
from pystruct.models import LatentGraphCRF
from pystruct.learners import OneSlackSSVM, LatentSSVM


def latent_svm(C=1.0, epsilon=.01):
    # if C == 0 or epsilon == 0:
    #     return 1
    print "Running Latent SVM with C={0} and epsilon={1}".format(C, epsilon)
    print "Generating data"
    X_train, X_test, y_train, y_test = get_data(test_size=0.3)
    print "Creating Latent Graph CRF"
    latent_pbl = LatentGraphCRF(n_states_per_label=4, n_features=228, inference_method='unary')
    print "Creating One Slack SSVM"
    base_ssvm = OneSlackSSVM(latent_pbl, C=C, tol=epsilon, inactive_threshold=1e-3, verbose=0)
    print "Creating Latent SSVM"
    latent_svm = LatentSSVM(base_ssvm=base_ssvm, latent_iter=2)
    print "Fitting Data"
    latent_svm.fit(X_train, y_train)
    print "Retrieving accuracy"
    acc = latent_svm.score(X_test, y_test)

    print "Successfully completed function. Returning error rate"
    return (1 - acc)


def main(job_id, params):
    C = params['C'][0]
    epsilon = params['EPSILON'][0]

    res = latent_svm(C, epsilon)
    print "Latent Sturctured SVM on Molecular Biology (Promoter Gene Sequences) dataset:", str(job_id)
    print "\tf(%.2f, %0.2f) = %f" % (C, epsilon, res)
    print time.gmtime()[3:6]
    return res

# print(latent_svm(C=1000000.0, epsilon=0.0001))