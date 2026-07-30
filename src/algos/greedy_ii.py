class GreedyII:
    name = "greedy_ii"

    def __init__(self):
        self.used_by_agent = {
            "A": set(),
            "B": set(),
        }

    @staticmethod
    def get_rank(agent, utterance):
        return int(agent.true_meaning[utterance])

    def is_acceptable(self, agent, utterance):
        return self.get_rank(agent, utterance) <= agent.tau

    def get_unproposed_acceptable_actions(self, agent, U_space):
        agent_id = agent.agent_id

        candidates = [
            utterance
            for utterance in U_space
            if utterance not in self.used_by_agent[agent_id]
            and self.is_acceptable(agent, utterance)
        ]

        return sorted(
            candidates,
            key=lambda utterance: (
                self.get_rank(agent, utterance),
                utterance,
            ),
        )

    def choose_utterance(
        self,
        speaker,
        listener,
        game,
        U_space,
        turn,
        history,
    ):
        del listener, game, turn  # Not required by this algorithm.

        speaker_id = speaker.agent_id

        if not history:
            candidates = self.get_unproposed_acceptable_actions(
                speaker,
                U_space,
            )

            proposal = candidates[0]
            self.used_by_agent[speaker_id].add(proposal)
            return proposal

        previous_proposal = history[-1]["utterance"]
        previous_rank = self.get_rank(speaker, previous_proposal)

        candidates = self.get_unproposed_acceptable_actions(
            speaker,
            U_space,
        )

        better_candidates = [
            utterance
            for utterance in candidates
            if self.get_rank(speaker, utterance) < previous_rank
        ]

        if (
            self.is_acceptable(speaker, previous_proposal)
            and not better_candidates
        ):
            return previous_proposal

        self.used_by_agent[speaker_id].add(previous_proposal)

        if better_candidates:
            proposal = better_candidates[0]
            self.used_by_agent[speaker_id].add(proposal)
            return proposal

        raise RuntimeError(
            f"Greedy II failed: agent {speaker_id} cannot accept "
            f"utterance {previous_proposal} with rank {previous_rank}, "
            "and has no acceptable unproposed actions remaining."
        )