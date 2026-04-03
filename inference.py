import os
from models import EmailAction
from email_env import EmailEnv

TASK_NAME = "email_triage"
BENCHMARK = "email_env"
MODEL_NAME = "baseline-model"

def log_start():
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)

def log_step(step, action, reward, done, error=None):
    error = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error}", flush=True)

def log_end(success, steps, rewards):
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def run():
    env = EmailEnv()
    obs = env.reset()

    log_start()

    rewards = []
    step_count = 0

    try:
        while True:
            step_count += 1

            # simple baseline logic
            text = obs.email_text.lower()

            if "free" in text or "offer" in text:
                action_str = "spam"
            elif "meeting" in text or "report" in text:
                action_str = "important"
            else:
                action_str = "reply"

            action = EmailAction(action=action_str)

            obs, reward_obj, done, _ = env.step(action)
            reward = reward_obj.score

            rewards.append(reward)

            log_step(step_count, action_str, reward, done)

            if done:
                break

        success = sum(rewards) > 0

    except Exception as e:
        success = False
        log_step(step_count, "error", 0.0, True, str(e))

    log_end(success, step_count, rewards)


if __name__ == "__main__":
    run()