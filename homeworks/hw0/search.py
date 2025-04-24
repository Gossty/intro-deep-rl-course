# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    if problem.isGoalState(problem.getStartState()):
        return []
    st = util.Stack()
    st.push((problem.getStartState(), []))

    visited = set()
    while not st.isEmpty():
        curr_name, curr_path = st.pop()
        if curr_name in visited:
            continue
        if problem.isGoalState(curr_name):
            return curr_path
        visited.add(curr_name)

        neighbors = problem.getSuccessors(curr_name)
        for neighbor_name, neighbor_action, neighbor_weight in neighbors:
            if neighbor_name in visited:
                continue
            neighbor_path = curr_path.copy()
            neighbor_path.append(neighbor_action)
            st.push((neighbor_name, neighbor_path))
    return []


def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    if problem.isGoalState(problem.getStartState()):
        return []
    q = util.Queue()
    q.push((problem.getStartState(), []))

    visited = set()
    while not q.isEmpty():
        curr_name, curr_path = q.pop()
        if curr_name in visited:
            continue
        if problem.isGoalState(curr_name):
            return curr_path
        visited.add(curr_name)

        neighbors = problem.getSuccessors(curr_name)
        for neighbor_name, neighbor_action, neighbor_weight in neighbors:
            if neighbor_name in visited:
                continue
            neighbor_path = curr_path.copy()
            neighbor_path.append(neighbor_action)
            q.push((neighbor_name, neighbor_path))
    return []

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    if problem.isGoalState(problem.getStartState()):
        return []
    pq = util.PriorityQueue()
    pq.push((problem.getStartState(), [], 0), 0)

    visited = set()

    while not pq.isEmpty():
        curr_name, curr_path, curr_weight = pq.pop()
        if curr_name in visited:
            continue
        if problem.isGoalState(curr_name):
            return curr_path
        visited.add(curr_name)

        neighbors = problem.getSuccessors(curr_name)
        for neighbor_name, neighbor_action, neighbor_weight in neighbors:
            if neighbor_name in visited:
                continue
            neighbor_path = curr_path.copy()
            neighbor_path.append(neighbor_action)
            action_weight = curr_weight + neighbor_weight
            pq.update((
                neighbor_name, neighbor_path, curr_weight + neighbor_weight),
                action_weight)
    return []


def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    if problem.isGoalState(problem.getStartState()):
        return []
    pq = util.PriorityQueue()
    start_score = 0 + heuristic(problem.getStartState(), problem)
    pq.push(problem.getStartState(), start_score)

    gscore = dict()
    came_from = dict()
    gscore[problem.getStartState()] = 0

    path_start = None
    while not pq.isEmpty():
        curr_name = pq.pop()
        if problem.isGoalState(curr_name):
            path_start = curr_name
            break
        neighbors = problem.getSuccessors(curr_name)
        for neighbor_name, neighbor_action, neighbor_weight in neighbors:
            tentative_gscore = gscore[curr_name] + neighbor_weight
            if neighbor_name not in gscore:
                gscore[neighbor_name] = float('inf')
            if tentative_gscore < gscore[neighbor_name]:
                gscore[neighbor_name] = tentative_gscore
                came_from[neighbor_name] = (curr_name, neighbor_action)
                pq.update(neighbor_name, gscore[neighbor_name] + heuristic(neighbor_name, problem))
    path = []
    curr = path_start
    end = problem.getStartState()
    while curr != end:
        curr, action = came_from[curr]
        path.append(action)
    return list(reversed(path))

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
 