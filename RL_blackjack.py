# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 12:47:28 2021

@author: Andrea
"""




from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


ACTION_HIT = 0
ACTION_STAND = 1
actions = [ACTION_HIT, ACTION_STAND]


policyPlayer = np.zeros(22)

for i in range(12,20):
    policyPlayer[i] = ACTION_HIT
    
policyPlayer[20] = ACTION_STAND
policyPlayer[21] = ACTION_STAND

def targetPolicyPlayer(usableAcePlayer, playerSum, dealerCard):
    return policyPlayer[playerSum]


def behaviorPolicyPlayer(usableAcePlayer, playerSum, dealerCard):
    if np.random.binomial(1, 0.5)== 1:
        return ACTION_STAND
    return ACTION_HIT

policyDealer = np.zeros(22)
for i in range(12,17):
     policyDealer[i] = ACTION_HIT
for i in range(17,22):
     policyDealer[i] = ACTION_STAND
     
def getCard():
    card = np.random.randint(1,14)
    card = min(card, 10)
    return card

def play(policyPlayerFn, initialState = None, initialAction = None):
    playerSum = 0
    playerTrajectory = []
    usableAcePlayer = False
    dealerCard1 = 0
    dealerCard2 = 0
    usableAceDealer = False
    if initialState is None:
        numOfAce = 0
        while playerSum <12:
            card = getCard()
            if card == 1:
                numOfAce += 1
                card = 11
                usableAcePlayer = True
            playerSum += card
            
        if playerSum >21:
            playerSum-= 10
            if numOfAce == 1:
                usableAcePlayer = False
            dealerCard1 = getCard()
            dealerCard2 = getCard()
        else:
            usableAcePlayer = initialState[0]
            playerSum = initialState[1]
            dealerCard1 = initialState[2]
            dealerCard = getCard()
        state = [usableAcePlayer, playerSum, dealerCard]
        dealerSum = 0
        if dealerCard1 == 1 and dealerCard2 != 1:
            dealerSum += 11 + dealerCard2
            usableAceDealer = True
        elif dealerCard1 != 1 and dealerCard2 ==1:
            dealerSum += dealerCard1 +11
            usableAceDealer = True
        elif dealerCard1 == 1 and dealerCard2 ==1:
            dealerSum += 1 +11
            usableAceDealer = True
        else:
            dealerSum += dealerCard1 + dealerCard2
        while True:
            if initialAction is not None:
                action = initialAction
                initialAction = None
            else:
                action = policyPlayerFn(usableAcePlayer, playerSum, dealerCard1)
            playerTrajectory.append([action, (usableAcePlayer, playerSum, dealerCard1)])
            if action == ACTION_STAND:
                break
            playerSum += getCard()
            if playerSum > 21:
                if usableAcePlayer == True:
                    playerSum -= 10
                    usableAcePlayer = False
                else:
                    return state, -1, playerTrajectory
        while True:
            action = policyDealer[dealerSum]
            if action == ACTION_STAND:
                break
            dealerSum += getCard()
            if dealerSum > 21:
                if usableAceDealer == True:
                    dealerSum -= 10
                    usableAceDealer = False
                else: 
                    return state, 1, playerTrajectory
        if playerSum > dealerSum:
            return state, 1, playerTrajectory
        elif playerSum == dealerSum:
            return state, 0, playerTrajectory
        else: 
            return state, -1, playerTrajectory
        
    
            
def monteCarloOnPolicy(nEpisodes):
    statesUsableAce = np.zeros((10,10))
    statesUsableAceCount = np.ones((10,10))
    statesNoUsableAce = np.zeros((10,10))
    statesNoUsableAceCount = np.ones((10,10))
    for i in range(0, nEpisodes):
        state, reward, _ = play(targetPolicyPlayer)
        state[1] -= 12
        state[2] -= 1
        if state[0]:
            statesUsableAceCount[state[1], state[2]] += 1
            statesUsableAce[state[1], state[2]] += reward
        else: 
            statesNoUsableAceCount[state[1], state[2]] += 1
            statesUsableAce[state[1], state[2]] += reward
    return statesUsableAce / statesUsableAceCount, statesNoUsableAce / statesNoUsableAceCount


def monteCarloES(nEpisodes):
    stateActionValues = np.zeros((10,10,2,2))
    stateActionPairCount = np.ones((10,10,2,2))
    def behaviorPolicy(usableAce, playerSum, dealerCard):
        usableAce = int(usableAce)
        playerSum -= 12
        dealerCard -= 1
        return np.argmax(stateActionValues[playerSum, dealerCard, usableAce, :] / stateActionPairCount[playerSum, dealerCard, usableAce, :])
    for episode in range(nEpisodes):
        if episode % 1000 == 0:
            print('episdoe:', episode)
        initialState = [bool(np.random.choice([0,1])), 
                        np.random.choice(range(12,22)), 
                        np.random.choice(range(1,11))]
        initialAction = np.random.choice(actions)
        _, reward, trajectory = play(behaviorPolicy, initialState, initialAction)
        for action, (usableAce, playerSum, dealerCard) in trajectory:
            usableAce = int(usableAce)
            playerSum -= 12
            dealerCard -= 1
            stateActionValues[playerSum, dealerCard, usableAce, action] += reward
            stateActionPairCount[playerSum, dealerCard, usableAce, action] += 1
    return stateActionValues / stateActionPairCount

figureIndex = 0

def prettyPrint(data, tile, zlabel = 'reward'):
    global figureIndex
    fig = plt.figure(figureIndex)
    figureIndex += 1
    fig.suptile(tile)
    ax = fig.add_subplot(111, projection = '3d')
    x_axis = []
    y_axis = []
    z_axis = []
    for i in range(12,22):
        for j in range(1,11):
            x_axis.apppend(i)
            y_axis.append(j)
            z_axis.append(data[i-22, j-1])
    ax.scatter(x_axis, y_axis, z_axis, c = 'red')
    ax.set_xlabel('player sum')
    ax.set_ylabel('dealer showing')
    ax.set_z_label(zlabel)
    
def onPolicy():
    statesUsableAce1, statesNoUsableAce1 = monteCarloOnPolicy(10000)
    statesUsableAce2, statesNoUsableAce2 = monteCarloOnPolicy(500000)
    prettyPrint(statesUsableAce1, "usable Ace & 1000 episodes")
    prettyPrint(statesNoUsableAce1, "no usable ace & 10000 episodes")
    prettyPrint(statesUsableAce2, "usable ace & 500000 episodes")
    prettyPrint(statesNoUsableAce2, "no usable Ace & 500000 episodes")
    plt.show()
    
def MC_ES_optimalPolicy():
    stateActionValues = monteCarloES(500000)
    stateValueUsableAce = np.zeros((10,10))
    stateValueNoUsableAce = np.zeros((10,10))
    #get optimal policy
    actionUsableAce = np.zeros((10,10), dtype = 'int')
    actionNoUsableAce = np.zeros((10,10), dtype = 'int')
    for i in range(10):
        for j in range(10):
            stateValueNoUsableAce[i,j] = np.max(stateActionValues[i,j,0,:])
            stateValueUsableAce[i,j] = np.max(stateActionValues[i,j, 1,:])
            actionNoUsableAce[i,j] = np.argmax(stateActionValues[i,j, 0, :])
            actionUsableAce[i,j] = np.argmax(stateActionValues[i,j,1,:])
    prettyPrint(stateValueUsableAce, 'optimal state value with usblae ace')
    prettyPrint(stateValueNoUsableAce, 'optimal state value with no usable ace')
    prettyPrint(actionUsableAce, 'optimal policy with usable ace', "action (0 hit, 1 stick)")
    prettyPrint(actionNoUsableAce, "optimal policy with no usable ace", "action (0 hit, 1 stick)")    
    plt.show()
    
#onPolicy()

#MC_ES_optimalPolicy()
    
onPolicy()
    

            
            
    
 
                
            
            
            
            
            
                
                
       