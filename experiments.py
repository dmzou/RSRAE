"""RSRAE"""

import os
import time

import numpy as np
from sklearn.preprocessing import normalize as nmlz
from sklearn.metrics import roc_auc_score, average_precision_score

import pickle
import argparse

import tensorflow.compat.v1 as tf
from keras.datasets import fashion_mnist

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.logging.set_verbosity(tf.logging.FATAL)

from RSRAE.model import CAE

#%% Parser

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", 
                    help="choose the gpu to use", 
                    default="0")
parser.add_argument("-t", "--true",
                    help="specify normal data (caltech101/fashion/20news/reuters)",
                    default="fashion")
parser.add_argument("-l", "--loss",
                    help="specify loss norm type",
                    default="L21")
parser.add_argument("-a", "--lambda1", type=float,
                    help="specify regularization coefficient",
                    default=0.0025)
parser.add_argument("-r", "--num2run", type=int,
                    help="specify number of runs for each setting",
                    default=1)
parser.add_argument("-q", "--enforce_proj",
                    help="whether enforce projection",
                    default="1")
parser.add_argument("-z", "--all_alt",
                    help="whether all alternation",
                    default="1")
parser.add_argument("-m", "--dim_latent", type=int,
                    help="dimension of latent layer",
                    default=10)
parser.add_argument("-n", "--renormalize",
                    default="1")
parser.add_argument("-b", "--batchnormalization",
                    default="1")
args = parser.parse_args()

"""Set GPU for use."""
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

"""Open log file."""
filename = "./res/" + args.true + "_" + args.loss + "_" + str(args.lambda1) + "_" + str(args.enforce_proj) + "_" + str(args.all_alt) + ".txt"
dataname = "./res/" + args.true + "_" + args.loss + "_" + str(args.lambda1) + "_" + str(args.enforce_proj) + "_" + str(args.all_alt) + ".data"

"""Data usage."""

if args.true == "fashion":
    _ , (X_test_origin, y_test_origin) = fashion_mnist.load_data()
elif args.true == "caltech101":
    with open("../../data/caltech101.data", 'rb') as f:
        data = pickle.load(f)
    X_test_origin = data["X"]
    y_test_origin = data["y"]
elif args.true == "tinyimagenet":
    with open("../data/tinyimagenet.data", 'rb') as f:
        data = pickle.load(f)
    X_test_origin = data["X"]
    y_test_origin = data["y"]
elif args.true == "tinyimagenetvar":
    with open("../data/tinyimagenetvar.data", 'rb') as f:
        data = pickle.load(f)
    X_test_origin = data["Xvar"]
    y_test_origin = data["y"]
elif args.true == "20news":
    with open("../data/20news.data", 'rb') as f:
        data = pickle.load(f)
    X_test_origin = data["X"]
    y_test_origin = data["y"]
elif args.true == "reuters":
    with open("../data/reuters.data", 'rb') as f:
        data = pickle.load(f)
    X_test_origin = data["X"]
    y_test_origin = data["y"]
else:
    raise Exception("Dataset not recognized!")
   
if args.enforce_proj == "1":
    if_enforce_proj = True
    if_rsr = True
else:
    if_enforce_proj = False
    if_rsr = False
if args.all_alt == "1":
    if_all_alt = True
else:
    if_all_alt = False
if args.renormalize == "1":
    renormalize = True
else:
    renormalize = False
if args.batchnormalization == "1":
    bn = True
else:
    bn = False

#%% Main test

if __name__ == "__main__":

    to_save_auc = {}
    to_save_ap = {}
    to_save_time = {}
    to_save_std_auc = {}
    to_save_std_ap = {}
    to_save_std_time = {}
    
    num_experiments = args.num2run
   
    if args.true in ("fashion", "tinyimagenet", "tinyimagenetvar"):
        inliers_set = list(range(10))
    elif args.true == "caltech101":
        inliers_set = list(range(11))
    elif args.true == "20news":
        inliers_set = list(range(20))
    elif args.true == "reuters":
        inliers_set = list(range(5))

    for cvalue in (0.1, 0.3, 0.5, 0.7, 0.9):
        
        to_save_auc[cvalue] = {}
        to_save_ap[cvalue] = {}
        to_save_time[cvalue] = {}
        to_save_std_auc[cvalue] = {}
        to_save_std_ap[cvalue] = {}
        to_save_std_time[cvalue] = {}
        
        
        for inlier_class in inliers_set:
            
            with open(filename, 'a') as f_log:
                f_log.write("Inlier digit: "+str(inlier_class)+"; c: "+str(cvalue) + "\n")

            print("Inlier digit: "+str(inlier_class)+"; c: "+str(cvalue))
            
            if args.true == "fashion":
                num_pure = 1000
            elif args.true == "caltech101":
                num_pure = 100
            elif args.true in ("tinyimagenet", "tinyimagenetvar"):
                num_pure = 500
            elif args.true in ("20news", "reuters"):
                num_pure = 360 
            num_anomaly = int( num_pure * cvalue )
           
            y_test = (np.array(y_test_origin) == inlier_class).astype(int)
                
            X_test_normal = X_test_origin[y_test==1]
            X_test_anomaly = X_test_origin[y_test==0][0:num_anomaly]
            X_test = np.concatenate((X_test_normal, X_test_anomaly))
                
            y_test_normal = y_test[y_test==1]
            y_test_anomaly = y_test[y_test==0][0:num_anomaly]
            y_test = [False] * len(X_test_normal) + [True] * num_anomaly
            
            if args.true == "fashion":
                X_test = np.reshape(X_test, (-1, 28*28*1)) / 255. * 2 - 1
                X_test = nmlz(X_test)
                X_test = np.reshape(X_test, (-1,28,28,1))
                input_shape = (28,28,1)
            elif args.true == "caltech101":
                X_test = np.reshape(X_test, (-1, 32*32*3)) / 255. * 2 - 1
                X_test = nmlz(X_test)
                X_test = np.reshape(X_test, (-1,32,32,3))
                input_shape = (32,32,3)
            elif args.true == "tinyimagenet":
                X_test = np.reshape(X_test, (-1, 32*32*3)) * 2 - 1
                X_test = nmlz(X_test)
                X_test = np.reshape(X_test, (-1,32,32,3))
                input_shape = (32,32,3)
            elif args.true == "tinyimagenetvar":
                X_test = np.reshape(X_test, (-1, 256)) / 4. * 2 - 1
                X_test = nmlz(X_test)
                input_shape = (256,)
            elif args.true == "20news":
                input_shape = (10000,)
            elif args.true == "reuters":
                input_shape = (26147,)

            if np.min(X_test) >= 0.:
                activation = tf.nn.relu
            elif np.abs(np.max(X_test)) <= 1.:
                activation = tf.nn.tanh
            else:
                activation = tf.nn.leaky_relu

                
            aucs = []
            aps = []
            time_elapses = []


            for idx_exp in range(num_experiments):

                print(f"Experiment No. {idx_exp+1}")
            
                cae = CAE(input_shape=input_shape, hidden_layer_sizes=(32,64,128), intrinsic_size=args.dim_latent,
                          activation=activation,
                          norm_type='L21', loss_norm_type=args.loss,
                          if_rsr=if_rsr, enforce_proj=if_enforce_proj, all_alt=if_all_alt,
                          learning_rate=0.00025, 
                          epoch_size=200, batch_show=20, 
                          lambda1=args.lambda1,
                          normalize=renormalize,
                          bn=bn,
                          random_seed=None)
                
                tStart = time.time()
                cae.fit(X_test, X_test)
                    
                features = cae.get_output(X_test)
                flat_output = np.reshape(features, (np.shape(X_test)[0], -1))
                flat_input = np.reshape(X_test, (np.shape(X_test)[0], -1))
                
                cosine_similarity = np.sum(flat_output * flat_input, -1) / (np.linalg.norm(flat_output, axis=-1) + 0.000001) / (np.linalg.norm(flat_input, axis=-1) + 0.000001)

                tEnd = time.time()
                tDiff = tEnd - tStart
                with open(filename, 'a') as f_log:
                    f_log.write("Time elapsed: " + str(tDiff) + "\n")

                auc = roc_auc_score(y_test, -cosine_similarity)
                ap = average_precision_score(y_test, -cosine_similarity)
                                    
                print("auc = ", auc)
                print("ap = ", ap)
                print("time elapse = ", tDiff)
                aucs.append(auc)
                aps.append(ap)
                time_elapses.append(tDiff)

            std_auc = np.std(aucs)
            std_ap = np.std(aps)
            std_time = np.std(time_elapses)

            to_save_auc[cvalue][inlier_class] = aucs
            to_save_ap[cvalue][inlier_class] = aps
            to_save_time[cvalue][inlier_class] = time_elapses
            to_save_std_auc[cvalue][inlier_class] = std_auc
            to_save_std_ap[cvalue][inlier_class] = std_ap
            to_save_std_time[cvalue][inlier_class] = std_time


            with open(filename, 'a') as f_log:
                f_log.write("Mean AUC score: "+str(np.mean(aucs))+"\n")
                f_log.write("Mean AP score: "+str(np.mean(aps))+"\n")
                f_log.write("Mean Time: "+str(np.mean(time_elapses))+"\n")
                f_log.write("Std AUC score: "+str(np.std(aucs))+"\n")
                f_log.write("Std AP score: "+str(np.std(aps))+"\n")
                f_log.write("Std Time: "+str(np.std(time_elapses))+"\n")
                f_log.write("\n")
        
        with open(filename, 'a') as f_log:
            f_log.write("Mean over class AUC: "+str(np.mean( list(to_save_auc[cvalue].values()) ))+"\n")
            f_log.write("Mean over class AP: "+str(np.mean( list(to_save_ap[cvalue].values()) ))+"\n")
            f_log.write("Mean over class Time: "+str(np.mean( list(to_save_time[cvalue].values()) ))+"\n")
            f_log.write("Std over class AUC: "+str(np.mean( list(to_save_std_auc[cvalue].values()) )) + "\n")
            f_log.write("Std over class AP: "+str(np.mean( list(to_save_std_ap[cvalue].values()) )) + "\n")
            f_log.write("Std over class Time: "+str(np.mean( list(to_save_std_time[cvalue].values()) )) + "\n")
            f_log.write("-----------------------------------------\n")


    to_save = (to_save_auc, to_save_ap, to_save_time, to_save_std_auc, to_save_std_ap, to_save_std_time)
    with open(dataname, "wb") as savedefile:
        pickle.dump(to_save, savedefile)
