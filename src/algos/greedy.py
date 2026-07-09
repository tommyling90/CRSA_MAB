class Greedy:
    def __init__(self):
        self.blocked_by_agent = {
            "A": set(),
            "B": set(),
        }
        self.name = "greedy"

    def is_acceptable(self, agent, u):
        return agent.true_meaning[u] <= agent.tau

    def get_rank(self, agent, u):
        return agent.true_meaning[u]

    def choose_utterance(self, speaker, listener, game, U_space, turn, history):
        speaker_id = speaker.agent_id

        if len(history) > 0:
            prev_u = history[-1]["utterance"]

            if self.is_acceptable(speaker, prev_u):
                return prev_u
            else:
                self.blocked_by_agent[speaker_id].add(prev_u)


        candidates = []

        for u in U_space:
            if u in self.blocked_by_agent[speaker_id]:
                continue

            if self.is_acceptable(speaker, u):
                candidates.append(u)

        if len(candidates) == 0:
            raise RuntimeError(
                f"Greedy failed: agent {speaker_id} has no remaining acceptable proposals."
            )

        best_u = min(
            candidates,
            key=lambda u: (self.get_rank(speaker, u), u)
        )

        self.blocked_by_agent[speaker_id].add(best_u)

        return best_u