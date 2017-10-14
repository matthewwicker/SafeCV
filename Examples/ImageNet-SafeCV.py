
# coding: utf-8

# In[2]:


import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import os
import cv2
import numpy as np
import keras 
from keras import applications
from SafeCV import *



model = keras.applications.vgg16.VGG16()
imagenet_class = "920"
image = cv2.imread("TEST_IMAGE.JPEG")
image = cv2.resize(image, (224,224))

params_for_run = MCTS_Parameters(image, int(imagenet_class), model)
params_for_run.verbose = True
params_for_run.simulations_cutoff = 35
params_for_run.backtracking_constant = 50

best_image, sev, prob, statistics = MCTS(params_for_run)


# In[4]:


import matplotlib.pyplot as plt
print("BEST ADVERSARIAL EXAMPLE:")
plt.imshow(best_image)
plt.show()
prob = model.predict(best_image.reshape(1,224,224,3))
new_class = np.argmax(prob[0])
new_prob = prob[0][np.argmax(prob)]
print("True class: %s; Predicted as: %s with confidence: %s; After %s manipulations"%(imagenet_class, new_class, new_prob, sev ))
plt.clf()
print("MCTS Run analysis:")
a, = plt.plot(statistics[0], label="Min Severity Found")
b, = plt.plot(statistics[1], label="Severity per Iteration")
c, = plt.plot(statistics[2], label="Rolling Average Severity")
plt.legend(handles=[a,b,c], loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("Single Run MCTS Statisitcs")
plt.xlabel("Iteration")
plt.ylabel("L_0 Severity")
plt.show()

