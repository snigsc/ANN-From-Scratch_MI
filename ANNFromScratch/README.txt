README FILE

ASSIGNMENT 3

TEAM NAME: PESU-MI_0045_0190_1143

Zip File Name: PESU-MI_0045_0190_1143.zip
Contents:
	data
		Clean_LBW_Dataset.csv
	src
		LBW_Dataset.csv
		preprocessing.py
		model.py
	README.txt
	Accuracy (screenshot showing our accuracies)

------

* Implementation:

The structure of our neural network is based on the fundamental math that goes behind it. We referred to 'Machine Learning' by Tom M Mitchell to get an understanding of how feedForward, error calculation and backPropagation work and coded the functions accordingly. We started off with filling missing values according to the what the attributes indicate; for instance, we filled missing HB values with the mean of HB values of women belonging to the same delivery phase, because some basic research concluded that HB levels vary according to the delivery phase a pregnant woman is in (similarly, reasons for how and why we've filled missing values of other attributes are specified in our code in comments). We have 9 nodes in the input layer, 22 in our 1 hidden layer (16 gave us the best accuracy), and 2 in the output layer. Finally, our accuracy is exactly 85%.


* Hyperparameters:

We have used 1 hidden layer with 22 nodes. The input layer has 9 nodes and the output layer has 2. The activation function we used is tanH. Since its values lie between -1 and 1, the mean for the hidden layer comes out to be 0 or very close to it, thus helping to center the data by bringing the mean close to 0. This makes learning for the next layer much easier. Our learning rate, eta, is 0.013. The number of epochs that gave us the highest accuracy is 1500. 


* Key Features:

- Rather than filling all missing values with the mean directly, we used different measures - mean/mode/median - based on what each attribute practically signifies. This ensured that binary attributes remained binary, and all filled in values were as practically sensible as possible.
- The key feature of our code is its adherence to the root mathematics behind neural networks and the simplicity of it. The reason for this is that after trying many other ways, what made our code crisp and clear was following the math from scratch. It's very easy to navigate and understand.


* Implementation Beyond the Basics:

While our pre-processing techniques were not the usual & standard kind, our neural network is quite straightforward. We ran our model with a balanced split on our dataset for making the train and test portions, which would've made our code unordinary; unfortunately this brought down our accuracy to 75%. We went ahead with an 80/20 split instead.


* Steps to run our file:

- While in directory 'src', first run preprocessing.py (this creates 'Clean_LBW_Dataset.csv' in the src folder, which is also already there in the 'data' directory as specified to us).
- Run model.py
Finally, model.py prints the confusion matrix, accuracy, precision, recall & FI score for our neural network.

**** N O T E :**** 

Despite using the seed function, our accuracies seem to slightly change each time with the testing accuracy almost always remaining 85 or above. It's our humble request to run model.py at least thrice! The accuracy in the attached screenshot in our zip file is 90% for test set and 86.8% for train set.

------