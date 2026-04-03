from models import EmailObservation, EmailAction, EmailReward


class EmailEnv:
    def __init__(self):
        self.emails = [
            {
                "email_text": "Congratulations! You’ve been selected for a free iPhone. Click here to claim now.",
                "sender": "promo@deals-now.com",
                "subject": "Exclusive Offer Just for You",
                "label": "spam"
            },
            {
                "email_text": "Hi team, please find attached the quarterly sales report. Let’s discuss tomorrow.",
                "sender": "manager@company.com",
                "subject": "Quarterly Review Meeting",
                "label": "important"
            },
            {
                "email_text": "Hey, can you check this code and give feedback by today?",
                "sender": "dev@company.com",
                "subject": "Code Review Needed",
                "label": "reply"
            }
        ]
        self.current_index = 0

    def reset(self):
        self.current_index = 0
        email = self.emails[self.current_index]
        return EmailObservation(**email)

    def step(self, action: EmailAction):
        current_email = self.emails[self.current_index]
        correct_label = current_email["label"]

        # 🔥 Improved reward logic
        if action.action == correct_label:
            reward = 1.0
        elif action.action in ["important", "reply"] and correct_label in ["important", "reply"]:
            reward = 0.5  # partial correct
        else:
            reward = -1.0

        # move to next email
        self.current_index += 1
        done = self.current_index >= len(self.emails)

        if not done:
            next_email = self.emails[self.current_index]
            observation = EmailObservation(**next_email)
        else:
            observation = None

        return observation, EmailReward(score=reward), done, {}

    def state(self):
        return {
            "current_index": self.current_index,
            "total_emails": len(self.emails)
        }


# 🧪 test
if __name__ == "__main__":
    env = EmailEnv()

    obs = env.reset()
    print(obs)

    action = EmailAction(action="spam")
    print(env.step(action))