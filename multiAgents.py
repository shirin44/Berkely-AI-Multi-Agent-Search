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
    Your minimax agent (question 2).
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Steps:
        1. Get all legal actions for Pacman (agent 0).
        2. Compute minimax scores for each action by calling the minimax function.
        3. Identify the action(s) with the maximum minimax score.
        4. Return one of the best actions (randomly chosen if there's a tie).
        """
        # Step 1: Get legal moves for Pacman (agent 0)
        legalMoves = gameState.getLegalActions(0)

        # Step 2: Compute minimax scores for each action
        scores = [self.minimax(1, 0, gameState.generateSuccessor(0, action)) for action in legalMoves]

        # Step 3: Find the action(s) with the maximum score
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]

        # Step 4: Choose randomly among the best actions
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def minimax(self, agentIndex, depth, gameState: GameState):
        """
        The core minimax recursive function.

        Steps:
        1. Check for terminal conditions (win, lose, or depth limit reached).
        2. If it's Pacman's turn, call maxValue to maximize the score.
        3. If it's a ghost's turn, call minValue to minimize the score.
        """
        # Step 1: Check for terminal conditions
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # Step 2: If it's Pacman's turn, maximize the score
        if agentIndex == 0:
            return self.maxValue(agentIndex, depth, gameState)

        # Step 3: If it's a ghost's turn, minimize the score
        return self.minValue(agentIndex, depth, gameState)

    def maxValue(self, agentIndex, depth, gameState: GameState):
        """
        Calculates the maximum value for Pacman's turn.

        Steps:
        1. Get all legal actions for Pacman.
        2. If there are no legal actions, return the state's evaluation score.
        3. For each action, compute the minimax value of the successor state.
        4. Return the maximum value among all successor states.
        """
        # Step 1: Get legal actions for Pacman
        legalActions = gameState.getLegalActions(agentIndex)

        # Step 2: Check for terminal conditions (no legal actions)
        if not legalActions:
            return self.evaluationFunction(gameState)

        # Step 3: Compute minimax values for all successor states
        scores = [self.minimax(1, depth, gameState.generateSuccessor(agentIndex, action)) for action in legalActions]

        # Step 4: Return the maximum score
        return max(scores)

    def minValue(self, agentIndex, depth, gameState: GameState):
        """
        Calculates the minimum value for the ghosts' turn.

        Steps:
        1. Get all legal actions for the current ghost.
        2. If there are no legal actions, return the state's evaluation score.
        3. Determine the next agent and depth (handle ghost and Pacman turns).
        4. For each action, compute the minimax value of the successor state.
        5. Return the minimum value among all successor states.
        """
        # Step 1: Get legal actions for the ghost
        legalActions = gameState.getLegalActions(agentIndex)

        # Step 2: Check for terminal conditions (no legal actions)
        if not legalActions:
            return self.evaluationFunction(gameState)

        # Step 3: Determine the next agent and depth
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        nextDepth = depth + 1 if nextAgent == 0 else depth

        # Step 4: Compute minimax values for all successor states
        scores = [self.minimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in legalActions]

        # Step 5: Return the minimum score
        return min(scores)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Alpha-Beta Pruning agent.
    """

    def getAction(self, gameState: GameState):
        """
        Returns the best action for Pacman using alpha-beta pruning.
        """
        # Step 1: Initialize alpha and beta to negative and positive infinity, respectively.
        alpha = float('-inf')
        beta = float('inf')
        bestAction = None

        # Step 2: Iterate through all legal actions for Pacman (agent 0).
        for action in gameState.getLegalActions(0):
            # Step 3: Compute the value of the successor state using the minValue function.
            value = self.minValue(1, 0, gameState.generateSuccessor(0, action), alpha, beta)
            
            # Step 4: Update alpha and bestAction if the computed value is greater than alpha.
            if value > alpha:
                alpha = value
                bestAction = action

        # Step 5: Return the action that maximizes Pacman's utility.
        return bestAction

    def maxValue(self, agentIndex, depth, gameState, alpha, beta):
        """
        Handles the maximizing player (Pacman).
        """
        # Step 1: Check terminal conditions (win, lose, or depth reached).
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # Step 2: Initialize the value to negative infinity.
        value = float('-inf')

        # Step 3: Iterate through all legal actions for the maximizing player (Pacman).
        for action in gameState.getLegalActions(agentIndex):
            # Step 4: Compute the minimum value of the successor state.
            value = max(value, self.minValue(1, depth, gameState.generateSuccessor(agentIndex, action), alpha, beta))
            
            # Step 5: Prune branches if the value exceeds beta (no need to explore further).
            if value > beta:
                return value
            
            # Step 6: Update alpha to the maximum of its current value and the computed value.
            alpha = max(alpha, value)

        # Step 7: Return the computed value for Pacman.
        return value

    def minValue(self, agentIndex, depth, gameState, alpha, beta):
        """
        Handles the minimizing player (ghosts).
        """
        # Step 1: Check terminal conditions (win, lose).
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # Step 2: Initialize the value to positive infinity.
        value = float('inf')

        # Step 3: Determine the next agent and depth.
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        nextDepth = depth + 1 if nextAgent == 0 else depth

        # Step 4: Iterate through all legal actions for the minimizing player (ghosts).
        for action in gameState.getLegalActions(agentIndex):
            # Step 5: Compute the value for the next agent.
            if nextAgent == 0:  # If the next agent is Pacman (maximizing player)
                value = min(value, self.maxValue(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action), alpha, beta))
            else:  # If the next agent is another ghost (minimizing player)
                value = min(value, self.minValue(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action), alpha, beta))
            
            # Step 6: Prune branches if the value is less than alpha (no need to explore further).
            if value < alpha:
                return value
            
            # Step 7: Update beta to the minimum of its current value and the computed value.
            beta = min(beta, value)

        # Step 8: Return the computed value for the ghost player.
        return value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An agent that uses the expectimax algorithm (question 4).
    """

    def getAction(self, gameState: GameState):
        """
        Returns the best action for Pacman using the expectimax algorithm.
        All ghosts are modeled to act randomly.
        """

        def expectimax(state, agentIndex, depth):
            """
            Recursive helper function to calculate expectimax values.
            """

            # Step 1: Check terminal conditions
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # Step 2: Pacman's turn (maximizing player)
            if agentIndex == 0:
                return maxValue(state, depth)

            # Step 3: Ghosts' turn (expectation player)
            else:
                return expectValue(state, agentIndex, depth)

        def maxValue(state, depth):
            """
            Computes the maximum value for Pacman.
            """
            v = float("-inf")  # Initialize with worst case for maximizer
            for action in state.getLegalActions(0):  # Iterate over Pacman's legal actions
                successor = state.generateSuccessor(0, action)
                v = max(v, expectimax(successor, 1, depth))  # Call expectimax for next agent
            return v

        def expectValue(state, agentIndex, depth):
            """
            Computes the expected value for ghosts, assuming random behavior.
            """
            v = 0  # Initialize value for expectation
            actions = state.getLegalActions(agentIndex)
            probability = 1.0 / len(actions) if actions else 0  # Equal probability for each action

            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                # If it's the last ghost, go back to Pacman (increase depth)
                if agentIndex == state.getNumAgents() - 1:
                    v += probability * expectimax(successor, 0, depth + 1)
                else:  # Otherwise, go to the next ghost
                    v += probability * expectimax(successor, agentIndex + 1, depth)

            return v

        # Main logic: choose the best action for Pacman
        bestAction = None
        bestValue = float("-inf")

        for action in gameState.getLegalActions(0):  # Iterate over all Pacman's legal actions
            successor = gameState.generateSuccessor(0, action)
            actionValue = expectimax(successor, 1, 0)  # Start expectimax with depth=0 and agentIndex=1 (first ghost)

            if actionValue > bestValue:  # Update the best action and value
                bestValue = actionValue
                bestAction = action

        return bestAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    A more advanced evaluation function for Pacman (question 5).

    Features:
    - Distance to the closest food.
    - Distance to ghosts (with penalties for proximity to active ghosts).
    - Number of remaining food items.
    - Scared ghost behavior (reward for eating scared ghosts).
    - Game score.
    """

    # Step 1: Extract useful information from the game state
    pacmanPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    currentScore = currentGameState.getScore()

    # Step 2: Initialize feature scores
    foodScore = 0
    ghostScore = 0
    capsuleScore = 0
    scaredGhostScore = 0

    # Step 3: Compute food-related score
    foodList = foodGrid.asList()
    if foodList:
        closestFoodDistance = min(manhattanDistance(pacmanPos, food) for food in foodList)
        foodScore = 10.0 / closestFoodDistance  # Inverse of distance to prioritize closer food

    # Step 4: Compute ghost-related score
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        ghostDistance = manhattanDistance(pacmanPos, ghostPos)
        if ghost.scaredTimer > 0:  # Scared ghost
            scaredGhostScore += 200 / max(ghostDistance, 1)  # Reward for chasing scared ghosts
        else:  # Active ghost
            if ghostDistance <= 1:
                ghostScore -= 500  # Severe penalty for being adjacent to a ghost
            else:
                ghostScore -= 50 / max(ghostDistance, 1)  # Penalize proximity to active ghosts

    # Step 5: Compute capsule-related score
    if capsules:
        closestCapsuleDistance = min(manhattanDistance(pacmanPos, capsule) for capsule in capsules)
        capsuleScore = 50.0 / closestCapsuleDistance  # Reward for being close to capsules

    # Step 6: Combine all scores into a single evaluation value
    totalScore = (
        currentScore +       # Base game score
        foodScore +          # Food proximity
        ghostScore +         # Ghost avoidance
        scaredGhostScore +   # Scared ghost chasing
        capsuleScore         # Capsule proximity
    )

    return totalScore


# Abbreviation
better = betterEvaluationFunction
