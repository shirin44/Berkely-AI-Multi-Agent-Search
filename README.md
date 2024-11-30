# **Multi-Agent Pacman Search**

### **Author**: Shirin Shujaa

This project is based on UC Berkeley’s Pacman AI framework and implements multi-agent search strategies to balance immediate rewards and long-term planning in adversarial and stochastic environments. It includes Reflex Agents, Minimax, Alpha-Beta Pruning, Expectimax, and enhanced evaluation functions to optimize Pacman’s gameplay.

---

## **Overview**

In this project, Pacman navigates a grid, avoiding ghosts and collecting food to maximize its score. The implemented agents use search strategies to handle adversarial and stochastic ghost behaviors effectively. Key features include:

- **Reflex Agent**: Makes immediate decisions based on state-action evaluations.  
- **Minimax Agent**: Uses adversarial search to predict and counteract ghost strategies.  
- **Alpha-Beta Pruning**: Optimizes Minimax by improving efficiency through pruning.  
- **Expectimax Agent**: Models ghosts' stochastic behaviors probabilistically.  
- **Better Evaluation Function**: Enhances state evaluations with feature engineering for strategic decision-making.  

---

## **Problem Statements**

### **Part 1: Reflex Agent and Minimax**

---

#### **Question 1: Reflex Agent**

- **Objective**:  
  The Reflex Agent selects actions based on immediate rewards without planning ahead. It balances food collection and ghost avoidance in layouts like `testClassic`.

- **Steps**:
  1. **Retrieve Legal Moves**:
     - Use `gameState.getLegalActions()` to get all possible actions for Pacman in the current state.
  2. **Evaluate Each Action**:
     - Compute a score for each action using the `evaluationFunction`, which includes factors like:
       - Distance to the nearest food.
       - Distance to ghosts, with penalties for proximity and rewards for chasing scared ghosts.
       - Penalty for stopping.
       - Penalty for remaining food count.
  3. **Select the Best Action**:
     - Find the action(s) with the maximum score. If there is a tie, randomly choose one of the best actions.
  4. **Return the Action**:
     - Return the chosen action for Pacman to execute.

**Code Snippet**:  
```python
def evaluationFunction(self, currentGameState, action):
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()

    # Food score
    foodList = newFood.asList()
    closestFoodDistance = min(manhattanDistance(newPos, food) for food in foodList)
    foodScore = 10.0 / closestFoodDistance if foodList else 0

    # Ghost score
    ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
    ghostScore = -500 if min(ghostDistances) <= 1 else -50 / min(ghostDistances)

    # Combine scores
    return successorGameState.getScore() + foodScore + ghostScore
```

---

#### **Question 2: Minimax Agent**

- **Objective**:  
  Design a Minimax Agent that uses adversarial search to maximize Pacman’s utility while minimizing the ghosts' utility.

- **Steps**:
  1. **Initialization**:
     - Use `gameState.getLegalActions(0)` to retrieve all legal actions for Pacman.
  2. **Recursive Evaluation**:
     - Compute the Minimax value for each action:
       - **Pacman’s Turn (`maxValue`)**:
         - Calculate the maximum value of all successor states.
       - **Ghosts’ Turn (`minValue`)**:
         - Calculate the minimum value of all successor states.
  3. **Handle Multiple Ghosts**:
     - Alternate between minimizing agents until all ghosts have taken their turns.
  4. **Base Case**:
     - Stop recursion if:
       - The game reaches a terminal state (win/lose).
       - The depth limit is reached.
  5. **Choose the Best Action**:
     - Select the action with the maximum Minimax value.
  6. **Return the Action**:
     - Return the selected action for Pacman to execute.

**Code Snippet**:  
```python
def minimax(self, agentIndex, depth, gameState):
    # Base case
    if gameState.isWin() or gameState.isLose() or depth == self.depth:
        return self.evaluationFunction(gameState)

    # Pacman's turn to maximize
    if agentIndex == 0:
        return self.maxValue(agentIndex, depth, gameState)

    # Ghosts' turn to minimize
    return self.minValue(agentIndex, depth, gameState)
```

---

### **Part 2: Advanced Multi-Agent Search**

---

#### **Question 3: Alpha-Beta Pruning**

- **Objective**:  
  Optimize the Minimax Agent by incorporating alpha-beta pruning to reduce the number of nodes explored.

- **Steps**:
  1. **Initialization**:
     - Start with `alpha = -∞` and `beta = +∞`.
  2. **Recursive Logic**:
     - **Pacman’s Turn (`maxValue`)**:
       - Compute the maximum value for all actions.
       - Update `alpha` with the maximum value so far.
       - Prune branches where `value > beta`.
     - **Ghosts’ Turn (`minValue`)**:
       - Compute the minimum value for all actions.
       - Update `beta` with the minimum value so far.
       - Prune branches where `value < alpha`.
  3. **Action Selection**:
     - Evaluate all legal actions for Pacman, applying pruning to minimize computations.
  4. **Return the Action**:
     - Return the action that maximizes Pacman’s utility.

**Code Snippet**:  
```python
def maxValue(self, agentIndex, depth, gameState, alpha, beta):
    value = float('-inf')
    for action in gameState.getLegalActions(agentIndex):
        value = max(value, self.minValue(1, depth, gameState.generateSuccessor(agentIndex, action), alpha, beta))
        if value > beta:  # Prune
            return value
        alpha = max(alpha, value)
    return value
```

---

#### **Question 4: Expectimax Agent**

- **Objective**:  
  Implement the Expectimax algorithm, assuming ghosts act randomly instead of optimally.

- **Steps**:
  1. **Initialization**:
     - Use the recursive `expectimax` function to evaluate the game tree.
  2. **Recursive Logic**:
     - **Pacman’s Turn (`maxValue`)**:
       - Compute the maximum value of all successor states.
     - **Ghosts’ Turn (`expectValue`)**:
       - Compute the expected value of all successor states, assuming uniform probabilities for ghost actions.
  3. **Base Case**:
     - Stop recursion if:
       - The game reaches a terminal state.
       - The depth limit is reached.
  4. **Action Selection**:
     - Choose the action with the highest expected value for Pacman.
  5. **Return the Action**:
     - Return the selected action for Pacman.

**Code Snippet**:  
```python
def expectimax(self, state, agentIndex, depth):
    if state.isWin() or state.isLose() or depth == self.depth:
        return self.evaluationFunction(state)

    if agentIndex == 0:  # Pacman's turn
        return max(self.expectimax(state.generateSuccessor(agentIndex, action), 1, depth)
                   for action in state.getLegalActions(agentIndex))
    else:  # Ghosts' turn
        actions = state.getLegalActions(agentIndex)
        probability = 1.0 / len(actions)
        return sum(probability * self.expectimax(state.generateSuccessor(agentIndex, action), (agentIndex + 1) % state.getNumAgents(), depth + (agentIndex + 1 == state.getNumAgents()))
                   for action in actions)
```

---

#### **Question 5: Better Evaluation Function**

- **Objective**:  
  Develop an advanced evaluation function to optimize Pacman’s decision-making.  

- **Steps**:
  1. **Extract Features**:
     - Use `currentGameState` to extract key information such as Pacman’s position, ghost states, food locations, capsules, and current score.
  2. **Compute Feature Scores**:
     - **Food Score**: Reward proximity to the nearest food.
     - **Ghost Score**: Penalize proximity to active ghosts and reward chasing scared ghosts.
     - **Capsule Score**: Reward proximity to capsules.
  3. **Combine Scores**:
     - Compute the total evaluation as a weighted sum of feature scores.
  4. **Return Evaluation**:
     - Return the computed score as the evaluation of the current state.

**Code Snippet**:  
```python
def betterEvaluationFunction(currentGameState):
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    # Food score
    closestFoodDistance = min(manhattanDistance(pacmanPos, food) for food in foodList) if foodList else 0
    foodScore = 10.0 / (closestFoodDistance + 1)

    # Ghost score
    ghostDistances = [manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in ghostStates]
    ghostScore = sum(-500 if dist <= 

1 else -50 / dist for dist in ghostDistances)

    # Capsule score
    capsuleDistance = min(manhattanDistance(pacmanPos, capsule) for capsule in capsules) if capsules else 0
    capsuleScore = 50.0 / (capsuleDistance + 1)

    return currentGameState.getScore() + foodScore + ghostScore + capsuleScore
```

---

## **How to Run**

### **Reflex Agent**
```bash
python autograder.py -q q1
python pacman.py -p ReflexAgent -l testClassic
```

### **Minimax Agent**
```bash
python autograder.py -q q2
python pacman.py -p MinimaxAgent -a depth=2
```

### **Alpha-Beta Pruning**
```bash
python autograder.py -q q3
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
```

### **Expectimax Agent**
```bash
python autograder.py -q q4
python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
```

### **Improved Evaluation Function**
```bash
python autograder.py -q q5
```

---

## **Conclusion**

This project highlights AI techniques like adversarial search, stochastic modeling, and heuristic optimization in dynamic environments. The Reflex Agent provides quick heuristic-based solutions, while advanced agents like Minimax and Expectimax add strategic depth. Alpha-Beta Pruning optimizes computational efficiency, and the improved evaluation function refines Pacman’s decision-making to maximize performance.
