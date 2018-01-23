import cv2
import numpy as np
from pomegranate import *
import numpy as np
import copy
from copy import deepcopy
import DFMCS
from DFMCS import DFMCS
from DFMCS import DFMCS_Parameters
import math

def RUN_UCB(keypoint_distribution, plays_per_node, TOTAL_PLAYS):
    retval = []
    for i in range(len(keypoint_distribution)):
        retval.append(keypoint_distribution[i] + math.sqrt(log(plays_per_node[i])/TOTAL_PLAYS))
    retval = np.asarray(retval)
    return retval/sum(retval)


class TreeNode(object):
    visited = False
    num_visits = 1
    visits_per_node = []
    lst = []
    dst = []
    lvl = []
    id_num = 0
    def __init__(self, lst, dst, lvl, id_num, params):
        if(id_num == -1):
            self.kp_list = None
            self.kp_dist = None
            self.level = -1
            self.id = -1 
        """ Creates an node object with a di"""
        self.kp_list = lst
        self.kp_dist = dst
        self.level = lvl
        self.id = id_num
        self.visits_per_node = np.ones(len(lst))
        self.params = params
        
    def selection(self):
        """ Returns a selection from the list of keypoints"""
        val = np.random.choice(range(len(self.kp_list)), p=self.kp_dist)
        self.visits_per_node[val]+=1
        return val
    
    def exploration(self):
        """ Returns a keypoint based on the UCB"""
        ucb = RUN_UCB(self.kp_dist, self.visits_per_node, self.num_visits)
        return np.random.choice(range(len(self.kp_list)), p=ucb)
    
    def visit_helper(self, k):
        """ Returns a tuple x,y that coresponds
        to the coords which we will manipulate"""
        mu_x, mu_y, sigma = int(round(k.pt[0])), int(round(k.pt[1])),  k.size
        # Remember, it may be wise to expand simga - greater varience = less honed attack
        sigma += self.params.SIGMA_CONSTANT
        d_x = NormalDistribution(mu_x, sigma)
        d_y = NormalDistribution(mu_y, sigma)
        x = d_x.sample()
        y = d_y.sample()
        if(self.params.small_image):
            x/=self.params.inflation_constant
            y/=self.params.inflation_constant
        if(x >= self.params.X_SHAPE):
            x = self.params.X_SHAPE-1
        elif(x < 0):
            x = 0
        if(y >= self.params.Y_SHAPE):
            y = self.params.Y_SHAPE-1
        elif(y < 0):
            y = 0
        return int(x), int(y)
    
    def visit(self, im, vc, manip_list):
        for i in range(vc):
            attempts = 0
            while(True):
                if(attempts == 5):
                    break
                x, y = self.visit_helper(self.kp_list[self.id])
                try:
                    if(((x,y) not in manip_list) and (list(self.params.MANIP(im[y][x], 3)) != list(im[y][x]))):
                        manip_list.append((x,y))
                        im[y][x] = self.params.MANIP(im[y][x], 3)
                        attempts = 0
                        break
                    else:
                        attempts +=1
                except:
                    if(((x,y) not in manip_list) and ((self.params.MANIP(im[y][x], 3)) != (im[y][x]))):
                        manip_list.append((x,y))
                        im[y][x] = self.params.MANIP(im[y][x], 3)
                        attempts = 0
                        break
                    else:
                        attempts +=1
        return im, manip_list
            
    def visit_random(self, im, vc):
        val = np.random.choice(range(len(self.kp_list)), p=self.kp_dist)
        for i in range(vc):
            x, y = self.visit_helper(self.kp_list[val])
            im[y][x] = MANIP(im[y][x], 3)
        return im

    def backprop(self, index, reward, severity):
        """ Updates the distribution based upon the
        reward passed in"""
        #severity /=10
        self.kp_dist[index] += (float(reward)/severity)
        if(self.kp_dist[index] < 0):
            self.kp_dist[index] = 0
        self.kp_dist = self.kp_dist/sum(self.kp_dist)
        

def white_manipulation(val, dim):
    return [255, 255, 255]


class MCTS_Parameters(object):
    def __init__(self, image, true_class, model, predshape = (1,224, 224, 3)):
        self.model = model
        self.ORIGINAL_IMAGE = copy.deepcopy(image)
        self.TRUE_CLASS = true_class
        self.MANIP = white_manipulation
        self.VISIT_CONSTANT = 100
        self.SIGMA_CONSTANT = 15
        self.X_SHAPE = 224
        self.Y_SHAPE = 224
        self.predshape = predshape
        self.kp, self.des, self.r = [],[],[]
        self.verbose = False
        self.small_image = False
        self.inflation_constant = 15
        self.simulations_cutoff = 10

        def preproc(im):
            im_pred = im.reshape(self.predshape)
            im_pred = im_pred.astype('float')
            return im_pred

        self.preprocess = preproc

        def predi(im):
            im_pred = self.preprocess(im)
            prob = self.model.predict(im_pred, batch_size=1, verbose=0)
            pred = np.argmax(np.asarray(prob))
            return pred, prob

        self.predict = predi
        pred, prob = self.predict(image)
        self.PROBABILITY = max(max(prob))
        self.backtracking_constant  = 10

def SIFT_Filtered(image, parameters, threshold=0.00):
    # We need to expand the image to get good keypoints
    if(parameters.small_image):
        xs = parameters.X_SHAPE * parameters.inflation_constant;
        ys = parameters.Y_SHAPE * parameters.inflation_constant;
        image = cv2.resize(image, (xs,ys))
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image,None)
    #FILTER RESPONSES:
    responses = []
    for x in kp:
        responses.append(x.response)
    responses.sort()
    ret = []
    index_tracker = 0
    for x in kp:
        if(x.response >= threshold):
            ret.append((x, des[index_tracker], x.response))            
        index_tracker = index_tracker + 1
    retval = sorted(ret, key=lambda tup: tup[2]) 
    return zip(*retval)

        
def MCTS(params):
    params.kp, params.des, params.r = SIFT_Filtered(params.ORIGINAL_IMAGE, params)
    params.r = np.asarray(params.r)
    params.r = params.r/sum(params.r)
    root = TreeNode(params.kp, params.r, 0, 0, params)
    levels = [[root]]
    current_level = 0
    node = root
    manipulation = []
    visited = [root]
    severities_over_time = []
    IMAGE = copy.deepcopy(params.ORIGINAL_IMAGE)
    MISCLASSIFIED = False
    raw_searches = params.simulations_cutoff
    count_searches = 0
    min_severity = -1
    best_image = None
    severities_over_time = []
    raw_severities =  []
    avg_severities =  []
    count_prior_saturation = 0
    while(True):
        #print(count_searches)
        if(count_searches == raw_searches):
            break
        count_searches += 1
        explored = False
        nxt = node.exploration()
        # First, we need to make sure that the layer we are going to initialize even exists
        try:
            test = levels[current_level+1]
        except:
            # If the layer does not exist, then we create the layers
            levels.append([TreeNode(params.kp, params.r, -1, -1, params) for i in range(len(params.kp))])
            if(params.verbose == True):
                print("Exploring new keypoints on a new layer: %s on node: %s"%(current_level+1, nxt))
        
            #Initialize the new node in the tree
            levels[current_level+1][nxt] = TreeNode(params.kp, params.r, current_level+1, nxt, params)
            IMAGE, manipulation = levels[current_level+1][nxt].visit(IMAGE, params.VISIT_CONSTANT, manipulation)
            visited.append(levels[current_level+1][nxt])
        
            #Visit the new node
            pred, prob = params.predict(IMAGE)
            NEW_PROBABILITY = prob[0][pred]
            if(pred != int(params.TRUE_CLASS)):
                MISCLASSIFIED = True
                break
            
            #Generate a DFS Adversarial example
            prior_severity = len(manipulation)
            if(prior_severity > min_severity):
                count_prior_saturation +=1
        
            if(MISCLASSIFIED == True):
                print("Satisfied before simulation")
                adv = copy.deepcopy(IMAGE)
                softmax = copy.deepcopy(prob)
                severity = prior_severity
            else:
                dparams = DFMCS_Parameters(params, IMAGE)
                adv, softmax, severity, kpd = DFMCS(dparams, cutoff=min_severity)
                if(severity != -1):
                    severity += prior_severity
                elif(severity == -1 and count_searches == 1):
                    break
            if((severity < min_severity or min_severity == -1) and severity != -1):
                severities_over_time.append(severity)
                min_severity = severity
                best_image = copy.deepcopy(adv)
            else:
                severities_over_time.append(min_severity)
                
            if(severity != -1):
                raw_severities.append(severity)
            else:
                raw_severities.append(min_severity)
                    
            avg_severities.append(np.average(raw_severities[-10:]))
            
            if(params.verbose == True):
                print("Back propogating and restarting search. Current Severity: %s"%(severity))
                print("Best severity: %s"%(min_severity))
                print("=================================================================\n")
             #Backprop
            for i in range(len(visited)):
                if(i == (len(visited)-1)):
                    break
                else:
                    visited[i].backprop(visited[i+1].id, params.PROBABILITY - NEW_PROBABILITY, current_level+1)
                
            IMAGE =  copy.deepcopy(params.ORIGINAL_IMAGE)   
            current_level = 0
            visited = [root]
            manipulations = []
            node = root
            explored = True
        if(not explored):
            if(  not (levels[current_level+1][nxt].id == -1)):
                pass
            else:
                if(params.verbose == True):
                    print("Exploring new keypoints on an existing layer: %s on node: %s"%(current_level+1, nxt))
            
                #Initialize the new node in the tree
                levels[current_level+1][nxt] = TreeNode(params.kp, params.r, current_level+1, nxt, params)
                IMAGE, manipulation = levels[current_level+1][nxt].visit(IMAGE, params.VISIT_CONSTANT, manipulation)
                visited.append(levels[current_level+1][nxt])
            
               
                pred, prob = params.predict(IMAGE)
                NEW_PROBABILITY = prob[0][pred]
                if(pred != int(params.TRUE_CLASS)):
                    MISCLASSIFIED = True
                    break
            
                #Generate a DFS Adversarial example
                prior_severity = (current_level+1)*params.VISIT_CONSTANT
                if(prior_severity > min_severity):
                    count_prior_saturation +=1
            
                if(MISCLASSIFIED == True):
                    adv = copy.deepcopy(IMAGE)
                    softmax = copy.deepcopy(prob)
                    severity = prior_severity
                else:
                    dparams = DFMCS_Parameters(params, IMAGE)
                    adv, softmax, severity, kpd = DFMCS(dparams, cutoff=min_severity)
                    if(severity != -1):
                        severity += prior_severity
            
                if((severity < min_severity or min_severity == -1) and severity != -1):
                    severities_over_time.append(severity)
                    min_severity = severity
                    best_image = copy.deepcopy(adv)
                else:
                    severities_over_time.append(min_severity)
                
                if(severity != -1):
                    raw_severities.append(severity)
                else:
                    raw_severities.append(min_severity)
                    
                avg_severities.append(np.average(raw_severities[-10:]))
                
                if(params.verbose == True):
                    print("Back propogating and restarting search. Current Severity: %s"%(severity))
                    print("Best severity: %s"%(min_severity))
                    print("=================================================================\n")
            
            
                for i in range(len(visited)):
                    if(i == (len(visited)-1)):
                        break
                    else:
                        visited[i].backprop(visited[i+1].id, params.PROBABILITY - NEW_PROBABILITY, current_level+1)
                    
                IMAGE =  copy.deepcopy(params.ORIGINAL_IMAGE)
                current_level = 0
                visited = [root]
                manipulations = []
                node = root
                explored = True 
        if(not explored):
            if(params.verbose == True):
                print("manipulating and continuing the search: %s"%(nxt))
            # Visit this node
            count_searches -= 1
            IMAGE, manipulation = levels[current_level+1][nxt].visit(IMAGE, params.VISIT_CONSTANT, manipulation)
        
            # Predict image class
            pred, prob = params.predict(IMAGE)
            NEW_PROBABILITY = prob[0][pred]
            if(pred != int(params.TRUE_CLASS)):
                MISCLASSIFIED = True
                break
                
        
            visited.append(node)
            node = levels[current_level+1][nxt]
            current_level = current_level + 1
        if(count_prior_saturation >= float(len(params.kp))):
            break
    if best_image is not None:
        pred, prob = params.predict(IMAGE)
        NEW_PROBABILITY = prob[0][pred]
    
    else:
        pred, prob = params.predict(IMAGE)
        NEW_PROBABILITY = prob[0][pred]
        best_image = params.ORIGINAL_IMAGE
        min_severity = 0

    stats = (severities_over_time, raw_severities, avg_severities)
    
    return best_image, min_severity, prob, stats
   
    
    
    
