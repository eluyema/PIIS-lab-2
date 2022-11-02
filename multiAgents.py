# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from email.mime import base
from turtle import position
from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        inf = 99999

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        pacmanPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newCapsulas = successorGameState.getCapsules()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        scaredGhostList = list()
        dangerousGhostList = list()
        for i, ghostPos in enumerate(successorGameState.getGhostPositions()):
            if newScaredTimes[i] < 3:
                dangerousGhostList.append(ghostPos)
            else:
                scaredGhostList.append(ghostPos)

        minFoodLen = inf
        minGhostLen = inf
        minCapsulasLen = inf
        for foodPos in newFood:
            minFoodLen = min(minFoodLen, manhattanDistance(pacmanPos, foodPos))
        for ghostPos in dangerousGhostList:
            minGhostLen = min(minGhostLen,manhattanDistance(pacmanPos, ghostPos))
        for capsulasPos in newCapsulas:
            minCapsulasLen = min(minCapsulasLen,manhattanDistance(pacmanPos, capsulasPos))

        foodScore = 1/(minFoodLen)

        scaredGhostPoints = 0
        if len(scaredGhostList)> 0:
            scaredGhostPoints = 2
        ghostsScore = 0
        if minGhostLen < 3:
            ghostsScore = inf
        stopScore = 0
        if action == Directions.STOP:
            stopScore = 5
        print(foodScore - ghostsScore + scaredGhostPoints - stopScore,
            foodScore,ghostsScore,scaredGhostPoints,stopScore)
        return foodScore - ghostsScore + scaredGhostPoints - stopScore

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the finalScore of the state.
    The finalScore is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    inf = 99999

    def max(self, gameState, agentIndex, depth):
            finalScore, finalAction = -self.inf, Directions.STOP
            for action in gameState.getLegalActions(agentIndex):
                nextGameState = gameState.generateSuccessor(agentIndex, action)
                nextScore, _ = self.minimax(nextGameState, agentIndex +1, depth)
                if finalScore < nextScore:
                    finalAction = action
                    finalScore = nextScore
            return finalScore, finalAction
    def mini(self, gameState, agentIndex, depth):
            finalScore, finalAction = self.inf, Directions.STOP
            for action in gameState.getLegalActions(agentIndex):
                nextGameState = gameState.generateSuccessor(agentIndex, action)
                nextScore, _ = self.minimax(nextGameState, agentIndex + 1, depth)
                if finalScore > nextScore:
                    finalAction = action
                    finalScore = nextScore
            return finalScore, finalAction

    def minimax(self, gameState, agentIndex, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP

        if agentIndex == gameState.getNumAgents() - 1:
            depth = depth - 1

        if agentIndex == gameState.getNumAgents():
            agentIndex = 0
                
        if agentIndex == 0:
            return self.max(gameState, agentIndex, depth)
        else:
            return self.mini(gameState, agentIndex, depth)

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        Here are some method calls that might be useful when implementing minimax.
        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1
        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action
        gameState.getNumAgents():
        Returns the total number of agents in the game
        gameState.isWin():
        Returns whether or not the game state is a winning state
        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        return self.minimax(gameState, 0, self.depth)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    inf = 99999
    def mini(self, gameState, agentIndex, depth, alpha, beta):
            finalScore, finalAction = self.inf, Directions.STOP
            for action in gameState.getLegalActions(agentIndex):
                newGameState = gameState.generateSuccessor(agentIndex, action)
                nextScore = self.alphaBeta(newGameState, agentIndex + 1, depth, alpha, beta)[0]
                if finalScore > nextScore:
                    finalScore = nextScore
                    finalAction = action
                if finalScore < alpha:
                    break
                beta = min(finalScore, beta)
            return (finalScore, finalAction)
    def max(self, gameState, agentIndex, depth, alpha, beta):
            finalScore, finalAction = -self.inf, Directions.STOP
            for action in gameState.getLegalActions(agentIndex):
                newGameState = gameState.generateSuccessor(agentIndex, action)
                nextScore = self.alphaBeta(newGameState, agentIndex + 1, depth, alpha, beta)[0]
                if finalScore < nextScore:
                    finalScore = nextScore
                    finalAction = action
                if finalScore > beta:
                    break
                alpha = max(finalScore, alpha)
            return (finalScore, finalAction)



    def alphaBeta(self, gameState, agentIndex, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return (self.evaluationFunction(gameState), Directions.STOP)

        if agentIndex == gameState.getNumAgents():
            agentIndex = 0

        if agentIndex == gameState.getNumAgents()-1:
            depth = depth - 1

        if agentIndex == 0:
            return self.max(gameState, agentIndex, depth, alpha, beta)
        else:
            return self.mini(gameState, agentIndex, depth, alpha, beta)

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphaBeta(gameState,0,self.depth,-self.inf,self.inf)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    inf = 99999

    def mini(self, gameState, agentIndex, depth):
            finalScore, finalAction = 0, Directions.STOP
            availableActions = gameState.getLegalActions(agentIndex)
            for action in availableActions:
                nextGameState = gameState.generateSuccessor(agentIndex, action)
                nextScore, _ = self.expectimax(nextGameState, agentIndex +1, depth)
                finalScore = finalScore + nextScore
            return finalScore/len(availableActions), finalAction
    def max(self, gameState, agentIndex, depth):
            finalScore, finalAction = -self.inf, Directions.STOP
            for action in gameState.getLegalActions(agentIndex):
                nextGameState = gameState.generateSuccessor(agentIndex, action)
                nextScore, _ = self.expectimax(nextGameState, agentIndex + 1, depth)
                if finalScore < nextScore:
                    finalAction = action
                    finalScore = nextScore
            return finalScore, finalAction
            
    def expectimax(self, gameState, agentIndex, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP

        if agentIndex == gameState.getNumAgents():
            agentIndex = 0

        if agentIndex == gameState.getNumAgents()-1:
            depth = depth - 1
                
        if agentIndex == 0:
            return self.max(gameState, agentIndex, depth)
        else:
            return self.mini(gameState, agentIndex, depth)

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(gameState, 0, self.depth)[1]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
