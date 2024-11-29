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


from util import manhattanDistance
from game import Directions
import random, util

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
        Chooses among the best actions based on the evaluation function.
        Returns a direction (NORTH, SOUTH, WEST, EAST, STOP).
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Compute scores for all legal moves
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        
        # Find the best score and corresponding actions
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        
        # Pick randomly among the best actions
        chosenIndex = random.choice(bestIndices)
        
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Optimized evaluation function.
        """
        # Step 1: Generate the successor state
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Step 2: Calculate individual feature scores
        foodScore = self.computeFoodScore(newPos, newFood)
        ghostScore = self.computeGhostScore(newPos, newGhostStates, newScaredTimes)
        stopPenalty = self.computeStopPenalty(action)
        foodCountPenalty = self.computeFoodCountPenalty(newFood)

        # Step 3: Combine feature scores into a single evaluation score
        totalScore = (
            successorGameState.getScore() +  # Base game score
            foodScore +                     # Food-related score
            ghostScore +                    # Ghost-related score
            stopPenalty +                   # Penalty for stopping
            foodCountPenalty                # Penalty for remaining food
        )

        return totalScore

    # Helper Methods for Evaluation Function

    def computeFoodScore(self, pacmanPos, foodGrid):
        """
        Compute a score for food based on the distance to the closest food.
        """
        foodList = foodGrid.asList()
        if not foodList:  # No food left
            return 0
        closestFoodDistance = min(manhattanDistance(pacmanPos, food) for food in foodList)
        return 10.0 / closestFoodDistance  # Inverse proportional reward

    def computeGhostScore(self, pacmanPos, ghostStates, scaredTimes):
        """
        Compute a score for ghost behavior, considering proximity and scared states.
        """
        ghostDistances = [manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in ghostStates]
        score = 0

        for ghostDistance, scaredTime in zip(ghostDistances, scaredTimes):
            if scaredTime > 0:  # Reward chasing scared ghosts
                score += 200 / max(ghostDistance, 1)
            else:  # Penalize proximity to active ghosts
                score -= 500 if ghostDistance <= 1 else 50 / max(ghostDistance, 1)

        return score

    def computeStopPenalty(self, action):
        """
        Penalize stopping to encourage movement.
        """
        return -50 if action == Directions.STOP else 0

    def computeFoodCountPenalty(self, foodGrid):
        """
        Penalize remaining food count to encourage faster clearing.
        """
        return -10 * len(foodGrid.asList()) 
    
    
def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

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
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
