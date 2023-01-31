Folder 1
Train
[ 6927  6929  6940 ... 35884 35885 35886]
Test
[   0    1    2 ... 7543 7562 7570]

Classification Report
              precision    recall  f1-score   support

           0       0.40      0.45      0.42       991
           1       0.64      0.29      0.40       110
           2       0.41      0.40      0.40      1024
           3       0.75      0.64      0.69      1798
           4       0.38      0.42      0.40      1216
           5       0.75      0.61      0.67       800
           6       0.43      0.52      0.47      1239

    accuracy                           0.51      7178
   macro avg       0.54      0.47      0.49      7178
weighted avg       0.53      0.51      0.52      7178

Confusion Matrix
[[ 443    7  124   71  163   25  158]
 [  23   32   14    3   23    1   14]
 [ 153    0  408   50  203   72  138]
 [ 135    5  104 1142  166   37  209]
 [ 195    2  143  102  512   11  251]
 [  42    1  104   43   49  485   76]
 [ 123    3  102  120  233   19  639]]

Test accuracy on folder 1
51.00306272506714

Folder 2
Train
[    0     1     2 ... 35884 35885 35886]
Test
[ 6927  6929  6940 ... 14998 15004 15008]

Classification Report
              precision    recall  f1-score   support

           0       0.70      0.74      0.72       991
           1       0.80      0.70      0.75       110
           2       0.70      0.66      0.68      1024
           3       0.87      0.89      0.88      1798
           4       0.72      0.69      0.70      1215
           5       0.85      0.84      0.85       800
           6       0.73      0.74      0.73      1240

    accuracy                           0.77      7178
   macro avg       0.77      0.75      0.76      7178
weighted avg       0.77      0.77      0.77      7178

Confusion Matrix
[[ 732    3   53   56   61   13   73]
 [   8   77    4    3    7    4    7]
 [  86    5  676   33  101   47   76]
 [  41    1   38 1604   43   15   56]
 [  92    6   93   61  834   13  116]
 [  17    1   40   34   13  675   20]
 [  68    3   62   62   97   26  922]]

Test accuracy on folder 2
76.9016444683075

Folder 3
Train
[    0     1     2 ... 35884 35885 35886]
Test
[14067 14107 14110 ... 22129 22131 22132]

Classification Report
              precision    recall  f1-score   support

           0       0.97      0.96      0.97       991
           1       1.00      0.95      0.98       109
           2       0.95      0.96      0.96      1024
           3       0.98      0.99      0.99      1798
           4       0.97      0.94      0.96      1215
           5       0.98      0.98      0.98       800
           6       0.96      0.98      0.97      1240

    accuracy                           0.97      7177
   macro avg       0.97      0.97      0.97      7177
weighted avg       0.97      0.97      0.97      7177

Confusion Matrix
[[ 954    0   11    4   10    1   11]
 [   3  104    0    1    1    0    0]
 [   9    0  980    8   10    8    9]
 [   3    0    4 1783    2    2    4]
 [  13    0   22   10 1145    2   23]
 [   0    0    6    7    2  782    3]
 [   4    0    4    5   10    1 1216]]

Test accuracy on folder 3
97.03218340873718

Folder 4
Train
[    0     1     2 ... 35884 35885 35886]
Test
[20325 20335 20384 ... 28916 28921 28922]

Classification Report
              precision    recall  f1-score   support

           0       0.99      0.98      0.99       990
           1       0.99      0.99      0.99       109
           2       0.98      0.98      0.98      1025
           3       1.00      1.00      1.00      1797
           4       0.99      0.99      0.99      1215
           5       0.99      0.99      0.99       801
           6       0.99      1.00      0.99      1240

    accuracy                           0.99      7177
   macro avg       0.99      0.99      0.99      7177
weighted avg       0.99      0.99      0.99      7177

Confusion Matrix
[[ 975    0    2    1    7    0    5]
 [   1  108    0    0    0    0    0]
 [   2    1 1003    1    7    7    4]
 [   0    0    0 1795    0    2    0]
 [   3    0    8    0 1201    0    3]
 [   2    0    3    1    0  795    0]
 [   2    0    3    1    0    0 1234]]

Test accuracy on folder 4
99.08039569854736

Folder 5
Train
[    0     1     2 ... 28916 28921 28922]
Test
[28541 28543 28547 ... 35884 35885 35886]

Classification Report
              precision    recall  f1-score   support

           0       0.99      0.99      0.99       990
           1       1.00      1.00      1.00       109
           2       0.98      0.99      0.98      1024
           3       1.00      1.00      1.00      1798
           4       1.00      0.99      0.99      1216
           5       0.99      0.99      0.99       801
           6       1.00      1.00      1.00      1239

    accuracy                           0.99      7177
   macro avg       0.99      0.99      0.99      7177
weighted avg       0.99      0.99      0.99      7177

Confusion Matrix
[[ 984    0    3    0    3    0    0]
 [   0  109    0    0    0    0    0]
 [   7    0 1009    0    2    4    2]
 [   0    0    1 1794    0    2    1]
 [   3    0    7    0 1205    0    1]
 [   0    0    5    2    0  794    0]
 [   0    0    0    0    1    0 1238]]

Test accuracy on folder 5
99.38693046569824

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 48, 48, 64)        1664      
                                                                 
 activation (Activation)     (None, 48, 48, 64)        0         
                                                                 
 conv2d_1 (Conv2D)           (None, 44, 44, 64)        102464    
                                                                 
 activation_1 (Activation)   (None, 44, 44, 64)        0         
                                                                 
 max_pooling2d (MaxPooling2D  (None, 14, 14, 64)       0         
 )                                                               
                                                                 
 conv2d_2 (Conv2D)           (None, 14, 14, 64)        36928     
                                                                 
 activation_2 (Activation)   (None, 14, 14, 64)        0         
                                                                 
 conv2d_3 (Conv2D)           (None, 12, 12, 64)        36928     
                                                                 
 activation_3 (Activation)   (None, 12, 12, 64)        0         
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 6, 6, 64)         0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 2304)              0         
                                                                 
 dense (Dense)               (None, 512)               1180160   
                                                                 
 activation_4 (Activation)   (None, 512)               0         
                                                                 
 dropout (Dropout)           (None, 512)               0         
                                                                 
 dense_1 (Dense)             (None, 7)                 3591      
                                                                 
 activation_5 (Activation)   (None, 7)                 0         
                                                                 
=================================================================
Total params: 1,361,735
Trainable params: 1,361,735
Non-trainable params: 0
_________________________________________________________________

SUMMARY

Mean accuracy
84.68084335327148

F1-Score
84.7997163498362

Precision
85.09460530102749

Recall
84.68084423272029

Confusion Matrix
[[ 817    2   38   26   48    7   49]
 [   7   86    3    1    6    1    4]
 [  51    1  815   18   64   27   45]
 [  35    1   29 1623   42   11   54]
 [  61    1   54   34  979    5   78]
 [  12    0   31   17   12  706   19]
 [  39    1   34   37   68    9 1049]]
