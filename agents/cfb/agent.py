"""Module for specifiying the Conservative Forward-Backward Agent."""

import torch
from typing import Dict, Tuple

from agents.fb.agent import FB


class CFB(FB):
    """
    Conservative Forward-Backward Agent. Can be
    either Measure-Conservative or Value-Conservative.
    """

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        preprocessor_hidden_dimension: int,
        preprocessor_output_dimension: int,
        preprocessor_hidden_layers: int,
        preprocessor_activation: str,
        z_dimension: int,
        forward_hidden_dimension: int,
        forward_hidden_layers: int,
        forward_number_of_features: int,
        backward_hidden_dimension: int,
        backward_hidden_layers: int,
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        forward_activation: str,
        backward_activation: str,
        actor_activation: str,
        critic_learning_rate: float,
        actor_learning_rate: float,
        learning_rate_coefficient: float,
        orthonormalisation_coefficient: float,
        discount: float,
        batch_size: int,
        z_mix_ratio: float,
        gaussian_actor: bool,
        std_dev_clip: float,
        std_dev_schedule: str,
        tau: float,
        device: torch.device,
        total_action_samples: int,
        ood_action_weight: float,
        alpha: float,
        target_conservative_penalty: float,
        vcfb: bool,
        mcfb: bool,
        lagrange: bool = False,
    ):
        assert vcfb != mcfb
        self.vcfb = vcfb
        self.mcfb = mcfb
        if self.vcfb:
            name = "VCFB"
        elif self.mcfb:
            name = "MCFB"
        else:
            raise ValueError("Either vcfb or mcfb must be True")

        super().__init__(
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            preprocessor_output_dimension=preprocessor_output_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            preprocessor_activation=preprocessor_activation,
            z_dimension=z_dimension,
            forward_hidden_dimension=forward_hidden_dimension,
            forward_hidden_layers=forward_hidden_layers,
            forward_number_of_features=forward_number_of_features,
            backward_hidden_dimension=backward_hidden_dimension,
            backward_hidden_layers=backward_hidden_layers,
            actor_hidden_dimension=actor_hidden_dimension,
            actor_hidden_layers=actor_hidden_layers,
            forward_activation=forward_activation,
            backward_activation=backward_activation,
            actor_activation=actor_activation,
            critic_learning_rate=critic_learning_rate,
            actor_learning_rate=actor_learning_rate,
            learning_rate_coefficient=learning_rate_coefficient,
            orthonormalisation_coefficient=orthonormalisation_coefficient,
            discount=discount,
            batch_size=batch_size,
            z_mix_ratio=z_mix_ratio,
            gaussian_actor=gaussian_actor,
            std_dev_clip=std_dev_clip,
            std_dev_schedule=std_dev_schedule,
            tau=tau,
            device=device,
            name=name,
        )

        # total_action_samples must be divisible by 4
        assert (ood_action_weight % 0.25 == 0) & (
            0 < ood_action_weight <= 1
        )  # ood_action_weight must be divisible by 0.25
        self.total_action_samples = total_action_samples
        self.ood_action_samples = int(self.total_action_samples * ood_action_weight)
        self.actor_action_samples = int(
            (self.total_action_samples - self.ood_action_samples) / 3
        )
        assert (
            self.ood_action_samples + (3 * self.actor_action_samples)
            == self.total_action_samples
        )

        self.alpha = alpha
        self.target_conservative_penalty = target_conservative_penalty

        # lagrange multiplier
        self.lagrange = lagrange
        self.critic_log_alpha = torch.zeros(1, requires_grad=True, device=self._device)

        # optimizer
        self.critic_alpha_optimizer = torch.optim.Adam(
            [self.critic_log_alpha], lr=critic_learning_rate
        )

    def update_fb(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        discounts: torch.Tensor,
        zs: torch.Tensor,
        step: int,
    ) -> Dict[str, float]:
        """
        Calculates the loss for the forward-backward representation network.
        Loss contains two components:
            1. Forward-backward representation (core) loss: a Bellman update
               on the successor measure (equation 24, Appendix B)
            2. Conservative loss: penalises out-of-distribution actions
        Args:
            observations: observation tensor of shape [batch_size, observation_length]
            actions: action tensor of shape [batch_size, action_length]
            next_observations: next observation tensor of
                                shape [batch_size, observation_length]
            discounts: discount tensor of shape [batch_size, 1]
            zs: policy tensor of shape [batch_size, z_dimension]
            step: current training step
        Returns:
            metrics: dictionary of metrics for logging
        """

        # update step common to all FB models
        (
            core_loss,
            core_metrics,
            F1,
            F2,
            B_next,
            M1_next,
            M2_next,
            _,
            _,
            actor_std_dev,
        ) = self._update_fb_inner(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            discounts=discounts,
            zs=zs,
            step=step,
        )

        # calculate MC or VC penalty
        if self.mcfb:
            (
                conservative_penalty,
                conservative_metrics,
            ) = self._measure_conservative_penalty(
                observations=observations,
                next_observations=next_observations,
                zs=zs,
                actor_std_dev=actor_std_dev,
                F1=F1,
                F2=F2,
                B_next=B_next,
                M1_next=M1_next,
                M2_next=M2_next,
            )
        # VCFB
        else:
            (
                conservative_penalty,
                conservative_metrics,
            ) = self._value_conservative_penalty(
                observations=observations,
                next_observations=next_observations,
                zs=zs,
                actor_std_dev=actor_std_dev,
                F1=F1,
                F2=F2,
            )

        # tune alpha from conservative penalty
        alpha, alpha_metrics = self._tune_alpha(
            conservative_penalty=conservative_penalty
        )
        conservative_loss = alpha * conservative_penalty

        total_loss = core_loss + conservative_loss

        # step optimizer
        self.FB_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        for param in self.FB.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.FB_optimizer.step()

        metrics = {
            **core_metrics,
            **conservative_metrics,
            **alpha_metrics,
            "train/forward_backward_total_loss": total_loss,
        }

        return metrics

    def _measure_conservative_penalty(
        self,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        zs: torch.Tensor,
        actor_std_dev: torch.Tensor,
        F1: torch.Tensor,
        F2: torch.Tensor,
        B_next: torch.Tensor,
        M1_next: torch.Tensor,
        M2_next: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates the measure conservative penalty.
        Args:
            observations: observation tensor of shape [batch_size, observation_length]
            next_observations: next observation tensor of shape
                                                        [batch_size, observation_length]
            zs: task tensor of shape [batch_size, z_dimension]
            actor_std_dev: standard deviation of the actor
            F1: forward embedding no. 1
            F2: forward embedding no. 2
            B_next: backward embedding
            M1_next: successor measure no. 1
            M2_next: successor measure no. 2
        Returns:
            conservative_penalty: the measure conservative penalty
            metrics: dictionary of metrics for logging
        """

        with torch.no_grad():
            # repeat observations, next_observations, zs, and Bs
            # we fold the action sample dimension into the batch dimension
            # to allow the tensors to be passed through F and B; we then
            # reshape the output back to maintain the action sample dimension
            repeated_observations_ood = observations.repeat(
                self.ood_action_samples, 1, 1
            ).reshape(self.ood_action_samples * self.batch_size, -1)
            repeated_zs_ood = zs.repeat(self.ood_action_samples, 1, 1).reshape(
                self.ood_action_samples * self.batch_size, -1
            )
            ood_actions = torch.empty(
                size=(self.ood_action_samples * self.batch_size, self.action_length),
                device=self._device,
            ).uniform_(-1, 1)

            if self.actor_action_samples > 0:
                repeated_observations_actor = observations.repeat(
                    self.actor_action_samples, 1, 1
                ).reshape(self.actor_action_samples * self.batch_size, -1)
                repeated_next_observations_actor = next_observations.repeat(
                    self.actor_action_samples, 1, 1
                ).reshape(self.actor_action_samples * self.batch_size, -1)
                repeated_zs_actor = zs.repeat(self.actor_action_samples, 1, 1).reshape(
                    self.actor_action_samples * self.batch_size, -1
                )
                actor_current_actions, _ = self.actor(
                    repeated_observations_actor,
                    repeated_zs_actor,
                    std=actor_std_dev,
                    sample=True,
                )  # [actor_action_samples * batch_size, action_length]

                actor_next_actions, _ = self.actor(
                    repeated_next_observations_actor,
                    z=repeated_zs_actor,
                    std=actor_std_dev,
                    sample=True,
                )  # [actor_action_samples * batch_size, action_length]

        # get cml Fs
        ood_F1, ood_F2 = self.FB.forward_representation(
            repeated_observations_ood, ood_actions, repeated_zs_ood
        )  # [ood_action_samples * batch_size, latent_dim]

        if self.actor_action_samples > 0:
            actor_current_F1, actor_current_F2 = self.FB.forward_representation(
                repeated_observations_actor, actor_current_actions, repeated_zs_actor
            )  # [actor_action_samples * batch_size, latent_dim]
            actor_next_F1, actor_next_F2 = self.FB.forward_representation(
                repeated_next_observations_actor, actor_next_actions, repeated_zs_actor
            )  # [actor_action_samples * batch_size, latent_dim]
            repeated_F1, repeated_F2 = F1.repeat(
                self.actor_action_samples, 1, 1
            ).reshape(self.actor_action_samples * self.batch_size, -1), F2.repeat(
                self.actor_action_samples, 1, 1
            ).reshape(
                self.actor_action_samples * self.batch_size, -1
            )
            cat_F1 = torch.cat(
                [
                    ood_F1,
                    actor_current_F1,
                    actor_next_F1,
                    repeated_F1,
                ],
                dim=0,
            )
            cat_F2 = torch.cat(
                [
                    ood_F2,
                    actor_current_F2,
                    actor_next_F2,
                    repeated_F2,
                ],
                dim=0,
            )
        else:
            cat_F1 = ood_F1
            cat_F2 = ood_F2

        cml_cat_M1 = torch.einsum("sd, td -> st", cat_F1, B_next).reshape(
            self.total_action_samples, self.batch_size, -1
        )
        cml_cat_M2 = torch.einsum("sd, td -> st", cat_F2, B_next).reshape(
            self.total_action_samples, self.batch_size, -1
        )

        cml_logsumexp = (
            torch.logsumexp(cml_cat_M1, dim=0).mean()
            + torch.logsumexp(cml_cat_M2, dim=0).mean()
        )

        conservative_penalty = cml_logsumexp - (M1_next + M2_next).mean()

        metrics = {
            "train/cml_penalty": conservative_penalty.item(),
            "train/cml_cat_M1": cml_cat_M1.mean().item(),
            "train/cml_cat_M2": cml_cat_M2.mean().item(),
        }

        return conservative_penalty, metrics

    def _value_conservative_penalty(
        self,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        zs: torch.Tensor,
        actor_std_dev: torch.Tensor,
        F1: torch.Tensor,
        F2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates the value conservative penalty. See section 3
        and appendix B.1.3.
        Args:
            observations: observation tensor of shape [batch_size, observation_length]
            next_observations: next observation tensor of shape
                                                     [batch_size, observation_length]
            zs: task tensor of shape [batch_size, z_dimension]
            actor_std_dev: standard deviation of the actor
            F1: forward embedding no. 1
            F2: forward embedding no. 2
        Returns:
            conservative_penalty: the value conservative penalty
            metrics: dictionary of metrics for logging
        """

        with torch.no_grad():
            # repeat observations, next_observations, zs, and Bs
            # we fold the action sample dimension into the batch dimension
            # to allow the tensors to be passed through F and B; we then
            # reshape the output back to maintain the action sample dimension
            repeated_observations_ood = observations.repeat(
                self.ood_action_samples, 1, 1
            ).reshape(self.ood_action_samples * self.batch_size, -1)
            repeated_zs_ood = zs.repeat(self.ood_action_samples, 1, 1).reshape(
                self.ood_action_samples * self.batch_size, -1
            )
            ood_actions = torch.empty(
                size=(self.ood_action_samples * self.batch_size, self.action_length),
                device=self._device,
            ).uniform_(-1, 1)

            if self.actor_action_samples > 0:
                repeated_observations_actor = observations.repeat(
                    self.actor_action_samples, 1, 1
                ).reshape(self.actor_action_samples * self.batch_size, -1)
                repeated_next_observations_actor = next_observations.repeat(
                    self.actor_action_samples, 1, 1
                ).reshape(self.actor_action_samples * self.batch_size, -1)
                repeated_zs_actor = zs.repeat(self.actor_action_samples, 1, 1).reshape(
                    self.actor_action_samples * self.batch_size, -1
                )
                actor_current_actions, _ = self.actor(
                    repeated_observations_actor,
                    repeated_zs_actor,
                    std=actor_std_dev,
                    sample=True,
                )  # [actor_action_samples * batch_size, action_length]

                actor_next_actions, _ = self.actor(
                    repeated_next_observations_actor,
                    z=repeated_zs_actor,
                    std=actor_std_dev,
                    sample=True,
                )  # [actor_action_samples * batch_size, action_length]

        # get cml Fs
        ood_F1, ood_F2 = self.FB.forward_representation(
            repeated_observations_ood, ood_actions, repeated_zs_ood
        )  # [ood_action_samples * batch_size, latent_dim]

        if self.actor_action_samples > 0:
            actor_current_F1, actor_current_F2 = self.FB.forward_representation(
                repeated_observations_actor, actor_current_actions, repeated_zs_actor
            )  # [actor_action_samples * batch_size, latent_dim]
            actor_next_F1, actor_next_F2 = self.FB.forward_representation(
                repeated_next_observations_actor, actor_next_actions, repeated_zs_actor
            )  # [actor_action_samples * batch_size, latent_dim]
            repeated_F1, repeated_F2 = F1.repeat(
                self.actor_action_samples, 1, 1
            ).reshape(self.actor_action_samples * self.batch_size, -1), F2.repeat(
                self.actor_action_samples, 1, 1
            ).reshape(
                self.actor_action_samples * self.batch_size, -1
            )
            cat_F1 = torch.cat(
                [
                    ood_F1,
                    actor_current_F1,
                    actor_next_F1,
                    repeated_F1,
                ],
                dim=0,
            )
            cat_F2 = torch.cat(
                [
                    ood_F2,
                    actor_current_F2,
                    actor_next_F2,
                    repeated_F2,
                ],
                dim=0,
            )
        else:
            cat_F1 = ood_F1
            cat_F2 = ood_F2

        repeated_zs = zs.repeat(self.total_action_samples, 1, 1).reshape(
            self.total_action_samples * self.batch_size, -1
        )

        # convert to Qs
        cql_cat_Q1 = torch.einsum("sd, sd -> s", cat_F1, repeated_zs).reshape(
            self.total_action_samples, self.batch_size, -1
        )
        cql_cat_Q2 = torch.einsum("sd, sd -> s", cat_F2, repeated_zs).reshape(
            self.total_action_samples, self.batch_size, -1
        )

        cql_logsumexp = (
            torch.logsumexp(cql_cat_Q1, dim=0).mean()
            + torch.logsumexp(cql_cat_Q2, dim=0).mean()
        )

        # get existing Qs
        Q1, Q2 = [torch.einsum("sd, sd -> s", F, zs) for F in [F1, F2]]

        conservative_penalty = cql_logsumexp - (Q1 + Q2).mean()

        metrics = {
            "train/cql_penalty": conservative_penalty.item(),
            "train/cql_cat_Q1": cql_cat_Q1.mean().item(),
            "train/cql_cat_Q2": cql_cat_Q2.mean().item(),
        }

        return conservative_penalty, metrics

    def _tune_alpha(
        self,
        conservative_penalty: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Tunes the conservative penalty weight (alpha) w.r.t. target penalty.
        Discussed in Appendix B.1.4
        Args:
            conservative_penalty: the current conservative penalty
        Returns:
            alpha: the updated alpha
            alpha_loss: the alpha loss
        """

        # alpha auto-tuning
        if self.lagrange:
            alpha = torch.clamp(self.critic_log_alpha.exp(), min=0.0, max=1e6)
            alpha_loss = (
                -0.5 * alpha * (conservative_penalty - self.target_conservative_penalty)
            )

            self.critic_alpha_optimizer.zero_grad()
            alpha_loss.backward(retain_graph=True)
            self.critic_alpha_optimizer.step()
            alpha = torch.clamp(self.critic_log_alpha.exp(), min=0.0, max=1e6).detach()
            alpha_loss = alpha_loss.detach().item()

        # fixed alpha
        else:
            alpha = self.alpha
            alpha_loss = 0.0

        metrics = {
            "train/alpha": alpha,
            "train/alpha_loss": alpha_loss,
        }

        return alpha, metrics
