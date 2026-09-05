"""
Sutton and Barto from Scratch 1: Bandits and Dynamic Programming

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - create_bandit_testbed
def create_bandit_testbed(k, seed, mean=0.0, std=1.0):
    # TODO: Build a k-armed bandit testbed with seeded normal true values.
    rng = np.random.RandomState(seed)
    return rng.normal(mean, std, size=k)

# Step 2 - pull_arm
def pull_arm(true_values, action, rng):
    """Pull one arm and return reward = true value + unit-normal noise.

    Args:
        true_values (np.ndarray): Shape (k,) true mean reward of each arm.
        action (int): Index of the arm to pull.
        rng (np.random.Generator): Seeded random generator for the noise.

    Returns:
        float: Stochastic reward for this pull.
    """
    # TODO: Return the arm's true value plus unit-normal noise from rng
    noise = rng.normal(0.0, 1.0)
    reward = true_values[action] + noise
    return reward

# Step 3 - sample_average_update
def sample_average_update(q_values, action_counts, action, reward):
    # TODO: Update an action-value estimate incrementally from one new reward...
    q_values = q_values.copy()
    action_counts = action_counts.copy()
    action_counts[action] += 1
    q_values[action] += (1 / action_counts[action]) * (reward - q_values[action])
    return q_values, action_counts

# Step 4 - epsilon_greedy_action
def epsilon_greedy_action(q_values, epsilon, rng):
    # TODO: Choose an action epsilon-greedily from the current value estimates.
    k = len(q_values)

    if rng.random() < epsilon:
        return int(rng.integers(0, k))

    return int(np.argmax(q_values))

# Step 5 - run_bandit_episode
def run_bandit_episode(true_values, n_steps, epsilon, rng):
    """Run one bandit episode with epsilon-greedy selection and sample-average updates.

    Args:
        true_values (np.ndarray): Shape (k,) true mean reward of each arm.
        n_steps (int): Number of pulls in the episode.
        epsilon (float): Exploration probability for epsilon-greedy.
        rng (np.random.Generator): Seeded random generator.

    Returns:
        tuple: (rewards, actions) with shapes (n_steps,) and (n_steps,) of ints.
    """
    # TODO: Run one episode and return the rewards and actions sequences
    k = len(true_values)
    q_values = np.zeros(k)
    action_counts = np.zeros(k, dtype=int)

    rewards = np.zeros(n_steps)
    actions = np.zeros(n_steps, dtype=int)

    for t in range(n_steps):
        action = epsilon_greedy_action(q_values, epsilon, rng)
        reward = pull_arm(true_values, action, rng)
        q_values, action_counts = sample_average_update(q_values, action_counts, action, reward)
        rewards[t] = reward
        actions[t] = action

    return rewards, actions

# Step 6 - track_rewards_and_optimal_actions
def track_rewards_and_optimal_actions(true_values, n_steps, epsilon, rng):
    """Run one episode tracking rewards and optimal-arm choices.

    Args:
        true_values (np.ndarray): Shape (k,) true mean reward of each arm.
        n_steps (int): Number of pulls in the episode.
        epsilon (float): Exploration probability for epsilon-greedy.
        rng (np.random.Generator): Seeded random generator.

    Returns:
        tuple: (rewards, optimal_flags) each shape (n_steps,).
            optimal_flags entries are 0.0 or 1.0 floats.
    """
    # TODO: return per-step rewards and 0/1 optimal-arm flags
    rewards, actions = run_bandit_episode(true_values, n_steps, epsilon, rng)
    optimal_action = int(np.argmax(true_values))
    optimal_flags = np.zeros(n_steps, dtype=float)

    for i in range(len(actions)):
        if actions[i] == optimal_action:
            optimal_flags[i] = 1.0
    
    return rewards.astype(float), optimal_flags

# Step 7 - average_bandit_curves
def average_bandit_curves(k, n_runs, n_steps, epsilon, seed):
    # TODO: Average reward and optimal-action curves over many independent bandit runs.
    all_rewards = np.zeros((n_runs, n_steps), dtype=float)
    all_optimal = np.zeros((n_runs, n_steps), dtype=float)

    for i in range(n_runs):
        run_seed = seed + i

        true_values = create_bandit_testbed(k, run_seed)
        rng = np.random.default_rng(run_seed)

        rewards, optimal_flags = track_rewards_and_optimal_actions(
            true_values,
            n_steps,
            epsilon,
            rng
        )

        all_rewards[i] = rewards
        all_optimal[i] = optimal_flags

    mean_rewards = np.mean(all_rewards, axis=0)
    mean_optimal = np.mean(all_optimal, axis=0)

    return mean_rewards, mean_optimal

# Step 8 - apply_random_walk_drift
def apply_random_walk_drift(true_values, drift_std, rng):
    # TODO: Add an independent random-walk increment to every arm's true value.
    k = len(true_values)
    noise = rng.normal(0.0, drift_std, size=k)
    return true_values + noise

# Step 9 - constant_step_size_update
def constant_step_size_update(q_values, action, reward, alpha):
    # TODO: Apply the constant step-size update to the selected action...
    pred = q_values[action]
    q_values[action] += alpha * (reward - pred)
    return q_values

# Step 10 - optimistic_initialization
def optimistic_initialization(k, initial_value):
    # TODO: return a NumPy array of shape (k,) filled with initial_value
    ret = np.zeros(shape=(k, ))
    for i in range(k):
        ret[i] = initial_value
    
    return ret

# Step 11 - ucb_action_select
def ucb_action_select(q_values, action_counts, timestep, c):
    """Select an action by upper-confidence-bound scores.

    Args:
        q_values (np.ndarray): Action-value estimates, shape (k,).
        action_counts (np.ndarray): Visit counts per action, shape (k,).
        timestep (int): Current time step t (>= 1).
        c (float): Exploration constant.

    Returns:
        int: Index of the selected action.
    """
    # TODO: Choose action by UCB scores balancing value vs visit counts
    unvisited = np.where(action_counts == 0)[0]

    if len(unvisited) > 0:
        return int(unvisited[0])
    ucb_scores = q_values + c * np.sqrt(np.log(timestep) / action_counts)
    return np.argmax(ucb_scores)

# Step 12 - gradient_bandit_update
def gradient_bandit_update(preferences, action, reward, average_reward, alpha):
    # TODO: Update softmax action preferences with one gradient-bandit step.
    def softmax(x):
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()

    probs = softmax(preferences)
    preferences[action] += alpha * (reward - average_reward) * (1 - probs[action])
    for i in range(len(preferences)):
        if i == action:
            continue
        preferences[i] -= alpha * (reward - average_reward) * probs[i]
    
    return preferences

# Step 13 - bandit_parameter_study
def bandit_parameter_study(n_runs, n_steps, seed, settings):
    k = 10
    results = {}

    for setting in settings:
        method = setting["method"]
        param = setting["param"]
        nonstationary = setting.get("nonstationary", False)

        if nonstationary:
            label = f"{method}({param}),ns"
        else:
            label = f"{method}({param})"

        final_rewards = np.zeros(n_runs, dtype=float)

        for i in range(n_runs):
            run_seed = seed + i
            rng = np.random.default_rng(run_seed)

            # -------------------------
            # Create environment
            # -------------------------

            if nonstationary:
                true_values = np.zeros(k, dtype=float)
            else:
                true_values = create_bandit_testbed(
                    k,
                    run_seed
                )

            action_counts = np.zeros(k, dtype=int)

            # -------------------------
            # Initialize method
            # -------------------------

            if method == "optimistic":
                q_values = optimistic_initialization(
                    k,
                    param
                )
            else:
                q_values = np.zeros(k, dtype=float)

            if method == "gradient":
                preferences = np.zeros(k, dtype=float)
                average_reward = 0.0

            # -------------------------
            # Run episode
            # -------------------------

            for t in range(n_steps):

                # Nonstationary environment drifts
                if nonstationary:
                    true_values = apply_random_walk_drift(
                        true_values,
                        0.01,
                        rng
                    )

                # =========================
                # SELECT ACTION
                # =========================

                if method == "epsilon_greedy":

                    action = epsilon_greedy_action(
                        q_values,
                        param,
                        rng
                    )

                elif method == "constant_step":

                    action = epsilon_greedy_action(
                        q_values,
                        0.1,
                        rng
                    )

                elif method == "optimistic":

                    action = epsilon_greedy_action(
                        q_values,
                        0.0,
                        rng
                    )

                elif method == "ucb":

                    action = ucb_action_select(
                        q_values,
                        action_counts,
                        t + 1,
                        param
                    )

                elif method == "gradient":

                    shifted = preferences - np.max(preferences)
                    exp_preferences = np.exp(shifted)

                    probs = (
                        exp_preferences
                        / np.sum(exp_preferences)
                    )

                    action = int(
                        rng.choice(k, p=probs)
                    )

                else:
                    raise ValueError(
                        f"Unknown method: {method}"
                    )

                # =========================
                # GET REWARD
                # =========================

                reward = pull_arm(
                    true_values,
                    action,
                    rng
                )

                # =========================
                # UPDATE AGENT
                # =========================

                if method == "epsilon_greedy":

                    q_values, action_counts = (
                        sample_average_update(
                            q_values,
                            action_counts,
                            action,
                            reward
                        )
                    )

                elif method == "constant_step":

                    action_counts[action] += 1

                    q_values = constant_step_size_update(
                        q_values,
                        action,
                        reward,
                        param
                    )

                elif method == "optimistic":

                    action_counts[action] += 1

                    q_values = constant_step_size_update(
                        q_values,
                        action,
                        reward,
                        0.1
                    )

                elif method == "ucb":

                    q_values, action_counts = (
                        sample_average_update(
                            q_values,
                            action_counts,
                            action,
                            reward
                        )
                    )

                elif method == "gradient":

                    action_counts[action] += 1

                    # t starts at 0, so t + 1 is
                    # number of rewards seen.
                    average_reward += (
                        reward - average_reward
                    ) / (t + 1)

                    preferences = gradient_bandit_update(
                        preferences,
                        action,
                        reward,
                        average_reward,
                        param
                    )

                # We only care about reward at
                # the final timestep.
                if t == n_steps - 1:
                    final_rewards[i] = reward

        results[label] = np.mean(final_rewards)

    return results

# Step 14 - build_gridworld_mdp (not yet solved)
# TODO: implement

# Step 15 - iterative_policy_evaluation (not yet solved)
# TODO: implement

# Step 16 - greedy_policy_improvement (not yet solved)
# TODO: implement

# Step 17 - policy_iteration (not yet solved)
# TODO: implement

# Step 18 - value_iteration (not yet solved)
# TODO: implement

# Step 19 - build_gambler_mdp (not yet solved)
# TODO: implement

# Step 20 - gambler_value_iteration (not yet solved)
# TODO: implement

# Step 21 - extract_optimal_stakes (not yet solved)
# TODO: implement

