package tools;

import java.util.*;
import java.util.logging.*;
import java.util.stream.Collectors;

import cartago.Artifact;
import cartago.OPERATION;
import cartago.OpFeedbackParam;

public class QLearner extends Artifact {

  private Lab lab; // the lab environment that will be learnt 
  private int stateCount; // the number of possible states in the lab environment
  private int actionCount; // the number of possible actions in the lab environment
  private Map<String, double[][]> qTables = new HashMap<>();


    private static final Logger LOGGER = Logger.getLogger(QLearner.class.getName());

  public void init(String environmentURL) {

    // the URL of the W3C Thing Description of the lab Thing
    this.lab = new Lab(environmentURL);

    this.stateCount = this.lab.getStateCount();
    LOGGER.info("Initialized with a state space of n="+ stateCount);

    this.actionCount = this.lab.getActionCount();
    LOGGER.info("Initialized with an action space of m="+ actionCount);

    qTables = new HashMap<>();
  }

/**
* Computes a Q matrix for the state space and action space of the lab, and against
* a goal description. For example, the goal description can be of the form [z1level, z2Level],
* where z1Level is the desired value of the light level in Zone 1 of the lab,
* and z2Level is the desired value of the light level in Zone 2 of the lab.
* For exercise 11, the possible goal descriptions are:
* [0,0], [0,1], [0,2], [0,3], 
* [1,0], [1,1], [1,2], [1,3], 
* [2,0], [2,1], [2,2], [2,3], 
* [3,0], [3,1], [3,2], [3,3].
*
*<p>
* HINT: Use the methods of {@link LearningEnvironment} (implemented in {@link Lab})
* to interact with the learning environment (here, the lab), e.g., to retrieve the
* applicable actions, perform an action at the lab during learning etc.
*</p>
* @param  goalDescription  the desired goal against the which the Q matrix is calculated (e.g., [2,3])
* @param  episodesObj the number of episodes used for calculating the Q matrix
* @param  alphaObj the learning rate with range [0,1].
* @param  gammaObj the discount factor [0,1]
* @param epsilonObj the exploration probability [0,1]
* @param rewardObj the reward assigned when reaching the goal state
**/

  @OPERATION
  public void calculateQ(Object[] goalDescription , Object episodesObj, Object alphaObj, Object gammaObj, Object epsilonObj, Object rewardObj) {

      // ensure that the right datatypes are used
      Integer episodes = Integer.valueOf(episodesObj.toString());
      Double alpha = Double.valueOf(alphaObj.toString());
      Double gamma = Double.valueOf(gammaObj.toString());
      Double epsilon = Double.valueOf(epsilonObj.toString());
      Integer reward = Integer.valueOf(rewardObj.toString());

      // Create a new Q-table for this goal description if it doesn't already exist
      Integer z1 = Integer.valueOf(goalDescription[0].toString());
      Integer z2 = Integer.valueOf(goalDescription[1].toString());
      String goalKey = z1.toString() + z2.toString();

      if (!this.qTables.containsKey(goalKey)) {
          this.qTables.put(goalKey, initializeQTable());
      }

      // Retrieve the Q-table for this goal description
      double[][] qTable = this.qTables.get(goalKey);

      // Perform Q-learning
      Random random = new Random();
      for (int episode = 0; episode < episodes; episode++) {
          int currentState = this.lab.readCurrentState();

          while (true) {
              List<Integer> possibleActions = this.lab.getApplicableActions(currentState);
              int action = random.nextDouble() < epsilon
                      ? possibleActions.get(random.nextInt(possibleActions.size()))
                      : getBestAction(currentState, qTable);

              // Perform the action and get the reward
              this.lab.performAction(action);
              int nextState = this.lab.readCurrentState();
              int rewardReceived = getReward(goalDescription, nextState);

              // Update the Q-table
              double oldValue = qTable[currentState][action];
              double newValue = (1 - alpha) * oldValue + alpha * (rewardReceived + gamma * getMaxValue(qTable[nextState]));
              qTable[currentState][action] = newValue;

              // If we've reached the goal state, break out of the loop
              if (rewardReceived == reward) {
                  break;
              }

              // Update the current state
              currentState = nextState;
          }
      }
  }

    private int getBestAction(int state, double[][] qTable) {
        List<Integer> possibleActions = this.lab.getApplicableActions(state);
        int bestAction = possibleActions.get(0);
        for (Integer action : possibleActions) {
            if (qTable[state][action] > qTable[state][bestAction]) {
                bestAction = action;
            }
        }
        return bestAction;
    }

    private double getMaxValue(double[] values) {
        double max = values[0];
        for (double value : values) {
            if (value > max) {
                max = value;
            }
        }
        return max;
    }

    private int getReward(Object[] goalDescription, int state) {
        Integer z1 = Integer.valueOf(goalDescription[0].toString());
        Integer z2 = Integer.valueOf(goalDescription[1].toString());
        Integer[] currentStateDescription = this.lab.getFullCurrentState().toArray(new Integer[0]);
        return (z1.equals(currentStateDescription[0]) && z2.equals(currentStateDescription[1])) ? 1 : 0;
    }


    /**
* Returns information about the next best action based on a provided state and the QTable for
* a goal description. The returned information can be used by agents to invoke an action 
* using a ThingArtifact.
*
* @param  goalDescription  the desired goal against the which the Q matrix is calculated (e.g., [2,3])
* @param  currentStateDescription the current state e.g. [2,2,true,false,true,true,2]
* @param  nextBestActionTag the (returned) semantic annotation of the next best action, e.g. "http://example.org/was#SetZ1Light"
* @param  nextBestActionPayloadTags the (returned) semantic annotations of the payload of the next best action, e.g. [Z1Light]
* @param nextBestActionPayload the (returned) payload of the next best action, e.g. true
**/
  @OPERATION
  public void getActionFromState(Object[] goalDescription, Object[] currentStateDescription,
      OpFeedbackParam<String> nextBestActionTag, OpFeedbackParam<Object[]> nextBestActionPayloadTags,
      OpFeedbackParam<Object[]> nextBestActionPayload) {

      // Retrieve the Q-table for the given goal description
      Integer z1 = Integer.valueOf(goalDescription[0].toString());
      Integer z2 = Integer.valueOf(goalDescription[1].toString());
      String goalKey = z1.toString() + z2.toString();
      double[][] qTable = this.qTables.get(goalKey);

      // Convert the currentStateDescription to an array of integers
      Integer[] currentState = Arrays.stream(currentStateDescription)
              .map(Object::toString)
              .map(Integer::valueOf)
              .toArray(Integer[]::new);

      // Get the next best action based on the current state and Q-table
      int action = getBestAction(currentState[0], qTable);

      // Set the next best action tag
      nextBestActionTag.set("http://example.org/was#SetZ1Light");

      // Set the next best action payload tags
      Object[] payloadTags = {"Z1Light"};
      nextBestActionPayloadTags.set(payloadTags);

      // Set the next best action payload
      Object[] payload = {action == 1};
      nextBestActionPayload.set(payload);
  }

    /**
    * Print the Q matrix
    *
    * @param qTable the Q matrix
    */
  void printQTable(double[][] qTable) {
    System.out.println("Q matrix");
    for (int i = 0; i < qTable.length; i++) {
      System.out.print("From state " + i + ":  ");
     for (int j = 0; j < qTable[i].length; j++) {
      System.out.printf("%6.2f ", (qTable[i][j]));
      }
      System.out.println();
    }
  }

  /**
  * Initialize a Q matrix
  *
  * @return the Q matrix
  */
 private double[][] initializeQTable() {
    double[][] qTable = new double[this.stateCount][this.actionCount];
    for (int i = 0; i < stateCount; i++){
      for(int j = 0; j < actionCount; j++){
        qTable[i][j] = 0.0;
      }
    }
    return qTable;
  }
}
