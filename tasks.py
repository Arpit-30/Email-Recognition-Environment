from email_env import EmailEnv
from models import EmailAction

# 🎯 grader function (IMPORTANT)
def grader(predicted, actual):
    if predicted == actual:
        return 1.0
    elif predicted in ["important", "reply"] and actual in ["important", "reply"]:
        return 0.5
    return 0.0


# 🟢 EASY TASK
def easy_task():
    env = EmailEnv()
    obs = env.reset()

    total_score = 0
    steps = 0

    while True:
        # simple correct actions
        correct = env.emails[env.current_index]["label"]
        action = EmailAction(action=correct)

        obs, reward, done, _ = env.step(action)

        total_score += grader(action.action, correct)
        steps += 1

        if done:
            break

    return total_score / steps


# 🟡 MEDIUM TASK
def medium_task():
    env = EmailEnv()
    obs = env.reset()

    total_score = 0
    steps = 0

    while True:
        correct = env.emails[env.current_index]["label"]

        # introduce confusion
        if correct == "important":
            action = EmailAction(action="reply")
        else:
            action = EmailAction(action=correct)

        obs, reward, done, _ = env.step(action)

        total_score += grader(action.action, correct)
        steps += 1

        if done:
            break

    return total_score / steps


# 🔴 HARD TASK
def hard_task():
    env = EmailEnv()
    obs = env.reset()

    total_score = 0
    steps = 0

    while True:
        correct = env.emails[env.current_index]["label"]

        # mostly wrong decisions
        action = EmailAction(action="spam")

        obs, reward, done, _ = env.step(action)

        total_score += grader(action.action, correct)
        steps += 1

        if done:
            break

    return total_score / steps


if __name__ == "__main__":
    print("Easy:", easy_task())
    print("Medium:", medium_task())
    print("Hard:", hard_task())