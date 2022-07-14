import torch
from allennlp.modules.conditional_random_field import ConditionalRandomField
from allennlp.nn.util import logsumexp


class SmoothCRF(ConditionalRandomField):
    def forward(self, inputs: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor = None):
        """

        :param inputs: Shape [batch, token, tag]
        :param tags: Shape [batch, token] or [batch, token, tag]
        :param mask: Shape [batch, token]
        :return:
        """
        if mask is None:
            mask = tags.new_ones(tags.shape, dtype=torch.bool)
        mask = mask.to(dtype=torch.bool)
        if tags.dim() == 2:
            return super(SmoothCRF, self).forward(inputs, tags, mask)

        # smooth mode
        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._smooth_joint_likelihood(inputs, tags, mask)

        return torch.sum(log_numerator - log_denominator)

    def _smooth_joint_likelihood(
        self, logits: torch.Tensor, soft_tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, sequence_length, num_tags = logits.size()

        epsilon = 1e-30
        soft_tags = soft_tags.clone()
        soft_tags[soft_tags < epsilon] = epsilon

        # Transpose batch size and sequence dimensions
        mask = mask.transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()
        soft_tags = soft_tags.transpose(0, 1).contiguous()

        # Initial alpha is the (batch_size, num_tags) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.
        if self.include_start_end_transitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0] + soft_tags[0].log()
        else:
            alpha = logits[0] * soft_tags[0]

        # For each i we compute logits for the transitions from timestep i-1 to timestep i.
        # We do so in a (batch_size, num_tags, num_tags) tensor where the axes are
        # (instance, current_tag, next_tag)
        for i in range(1, sequence_length):
            # The emit scores are for time i ("next_tag") so we broadcast along the current_tag axis.
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            # Transition scores are (current_tag, next_tag) so we broadcast along the instance axis.
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            # Alpha is for the current_tag, so we broadcast along the next_tag axis.
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)

            # Add all the scores together and logexp over the current_tag axis.
            inner = broadcast_alpha + emit_scores + transition_scores + soft_tags[i].log().unsqueeze(1)

            # In valid positions (mask == True) we want to take the logsumexp over the current_tag dimension
            # of `inner`. Otherwise (mask == False) we want to retain the previous alpha.
            alpha = logsumexp(inner, 1) * mask[i].view(batch_size, 1) + alpha * (
                ~mask[i]
            ).view(batch_size, 1)

        # Every sequence needs to end with a transition to the stop_tag.
        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha

        # Finally we log_sum_exp along the num_tags dim, result is (batch_size,)
        return logsumexp(stops)
