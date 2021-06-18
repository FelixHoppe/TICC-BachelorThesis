import numpy as np
from scipy import stats

def computeF1Score_delete(num_cluster,matching_algo,actual_clusters,threshold_algo,save_matrix = False):
    """
    computes the F1 scores and returns a list of values
    """
    F1_score = np.zeros(num_cluster)
    for cluster in range(num_cluster):
        matched_cluster = matching_algo[cluster]
        true_matrix = actual_clusters[cluster]
        estimated_matrix = threshold_algo[matched_cluster]
        if save_matrix: np.savetxt("estimated_matrix_cluster=" + str(cluster)+".csv",estimated_matrix,delimiter = ",", fmt = "%1.4f")
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(num_stacked*n):
            for j in range(num_stacked*n):
                if estimated_matrix[i,j] == 1 and true_matrix[i,j] != 0:
                    TP += 1.0
                elif estimated_matrix[i,j] == 0 and true_matrix[i,j] == 0:
                    TN += 1.0
                elif estimated_matrix[i,j] == 1 and true_matrix[i,j] == 0:
                    FP += 1.0
                else:
                    FN += 1.0
        precision = (TP)/(TP + FP)
        recall = TP/(TP + FN)
        f1 = (2*precision*recall)/(precision + recall)
        F1_score[cluster] = f1
    return F1_score

def compute_confusion_matrix(num_clusters,clustered_points_algo, sorted_indices_algo):
    """
    computes a confusion matrix and returns it
    """
    seg_len = 200
    true_confusion_matrix = np.zeros([num_clusters,num_clusters])
    for point in range(len(clustered_points_algo)):
        cluster = int(clustered_points_algo[point])


        ##CASE G: ABBACCCA
        # num = (int(sorted_indices_algo[point]/seg_len) )
        # if num in [0,3,7]:
        #   true_confusion_matrix[0,cluster] += 1
        # elif num in[1,2]:
        #   true_confusion_matrix[1,cluster] += 1
        # else:
        #   true_confusion_matrix[2,cluster] += 1

        ##CASE F: ABCBA
        # num = (int(sorted_indices_algo[point]/seg_len))
        # num = min(num, 4-num)
        # true_confusion_matrix[num,cluster] += 1

        #CASE E : ABCABC
        num = (int(sorted_indices_algo[point]/seg_len) %num_clusters)
        true_confusion_matrix[num,cluster] += 1

        ##CASE D : ABABABAB
        # num = (int(sorted_indices_algo[point]/seg_len) %2)
        # true_confusion_matrix[num,cluster] += 1

        ##CASE C: 
        # num = (sorted_indices_algo[point]/seg_len)
        # if num < 15:
        #   true_confusion_matrix[0,cluster] += 1
        # elif num < 20:
        #   true_confusion_matrix[1,cluster] += 1
        # else:
        #   true_confusion_matrix[0,cluster] += 1

        ##CASE B : 
        # if num > 4:
        #   num = 9 - num
        # true_confusion_matrix[num,cluster] += 1

        ##CASE A : ABA
        # if sorted_indices_algo[point] < seg_len:
        #   true_confusion_matrix[0,cluster] += 1

        # elif sorted_indices_algo[point] <3*seg_len:
        #   true_confusion_matrix[1,cluster] += 1
        # else:
        #   true_confusion_matrix[0,cluster] += 1

    return true_confusion_matrix

def computeF1_macro(confusion_matrix,matching, num_clusters):
    """
    computes the macro F1 score
    confusion matrix : requres permutation
    matching according to which matrix must be permuted
    """
    ##Permute the matrix columns
    permuted_confusion_matrix = np.zeros([num_clusters,num_clusters])
    for cluster in range(num_clusters):
        matched_cluster = matching[cluster]
        permuted_confusion_matrix[:,cluster] = confusion_matrix[:,matched_cluster]
    ##Compute the F1 score for every cluster
    F1_score = 0
    for cluster in range(num_clusters):
        TP = permuted_confusion_matrix[cluster,cluster]
        FP = np.sum(permuted_confusion_matrix[:,cluster]) - TP
        FN = np.sum(permuted_confusion_matrix[cluster,:]) - TP
        precision = TP/(TP + FP)
        recall = TP/(TP + FN)
        f1 = stats.hmean([precision,recall])
        F1_score += f1
    F1_score /= num_clusters
    return F1_score