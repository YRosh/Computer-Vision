import os
import numpy as np
import random

def feature_extract(path, labels):
    fl = open(labels, 'r')
    labels = fl.readline().split(',')
    labels[-1] = labels[-1][0]
    fl.close()
    
    descriptors = []
    feats = []
    for des_file in sorted(os.listdir(path), key=lambda x: int(x.split('_')[0])):
        fp = open(path+'/'+des_file, 'r')
        feat = []
        for line in fp:
            des = line.split(',')
            des = [int(i) for i in des]
            descriptors.append(des[4:])
            feat.append(des[4:])
        feats.append(feat)
        fp.close()
    return (descriptors, feats, labels)

def kmeans(descriptors, k, max_iter=100):
    clusters = {}
    
    centroids = random.sample(descriptors, k=k)
    centroids = np.array(centroids)
    
    for itr in range(max_iter):
        for i in range(k):
            clusters[str(i)] = []
        for des in descriptors:
            des = np.array(des)
            distances = np.linalg.norm(centroids - des, axis=-1)
            ind = np.argmin(distances)
            clusters[str(ind)].append(des)
        for key, values in clusters.items():
            cluster = np.array(values)
            centroids[int(key)] = np.mean(cluster, axis=0)
        print("\rInteration: {}".format(itr+1), end="\r")
    return centroids

def get_histograms(words, features):
    hists = []
    for img in features:
        histogram = np.zeros(len(words))
        for feat in img:
            distances = np.linalg.norm(words - feat, axis=-1)
            ind = np.argmin(distances)
            histogram[ind] += 1
        hists.append(histogram)
    return hists

def knn(x_train, y_train, x_test, y_test, k):
    count = 0
    cls_based = np.zeros((8,8))
    for i, test_img in enumerate(x_test, 0):
        distances = np.linalg.norm(x_train - test_img, axis=-1)
        distances = np.argsort(distances)[:k]
        classes = [y_train[ind] for ind in distances]
        yhat = max(set(classes), key=classes.count)
        if yhat == y_test[i]:
            count += 1
        cls_based[int(yhat)-1][int(y_test[i])-1] += 1
    print("Accuracy : {}, for k={}.\nConfusion Matrix\n{}".format(round(count/len(y_test), 5), k, cls_based))
        
if __name__ == '__main__':
    train_features_path = r'E:\sem-6\CV\Assignment-3\HW3_data\train_sift_features'
    test_features_path = r'E:\sem-6\CV\Assignment-3\HW3_data\test_sift_features'
    train_labels = r'E:\sem-6\CV\Assignment-3\HW3_data\train_labels.csv'
    test_labels = r'E:\sem-6\CV\Assignment-3\HW3_data\test_labels.csv'
    
    output = feature_extract(train_features_path, train_labels)
    print("Train features Extracted.")
    descriptors_list = output[0]
    train_features = output[1]
    train_labels = output[2]
    
    output = feature_extract(test_features_path, test_labels)
    test_features = output[1]
    test_labels = output[2]
    print("Test features extracted.\n\nKMeans begins...")

    visual_words = kmeans(descriptors_list, 64)
    
    print("KMeans done. {} visual words extracted.\n\nGetting histograms.".format(len(visual_words)))
    
    train_hists = get_histograms(visual_words, train_features)
    print("Training histograms.")
    
    test_hists = get_histograms(visual_words, test_features)
    print("Testing histograms.\n\nTesting images")
    
    knn(train_hists, train_labels, test_hists, test_labels, 61)