import pickle

# Test loading the model
print("Testing model loading...")
with open("q_agent.pkl", "rb") as f:
    agent = pickle.load(f)

print(f"Agent loaded successfully!")
print(f"Agent type: {type(agent)}")
print(f"Agent has Q attribute: {hasattr(agent, 'Q')}")

if hasattr(agent, 'Q'):
    print(f"Q type: {type(agent.Q)}")
    print(f"Number of actions in Q: {len(agent.Q)}")

    # Check first action
    first_action = list(agent.Q.keys())[0] if agent.Q else None
    if first_action:
        print(f"First action: {first_action}")
        print(f"Type of Q[action]: {type(agent.Q[first_action])}")
        print(f"Number of states for first action: {len(agent.Q[first_action])}")

        # Show sample states
        sample_states = list(agent.Q[first_action].keys())[:3]
        print(f"Sample states: {sample_states}")
        for state in sample_states:
            print(f"  Q[{first_action}][{state}] = {agent.Q[first_action][state]}")

# Test a simple board state
test_board = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]
test_state = ''.join(test_board[r][c] for r in range(3) for c in range(3))
print(f"\nTest state (empty board): '{test_state}'")

# Check if this state exists in Q-values
for action in agent.Q:
    if test_state in agent.Q[action]:
        print(f"  Action {action} has Q-value: {agent.Q[action][test_state]}")
        break
else:
    print("  Empty board state not found in Q-values (this is normal)")

print("\n✅ Model test complete!")
