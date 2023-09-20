from agents.fb.models import ForwardBackwardRepresentation as FB
from agents.fb.base import ForwardModel


class ForwardBackwardRepresentation(FB):
    """Combined Forward-backward representation network,
    with additional Forward network for exploration policy mu."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # NOTE: reappropriate the forward model structure (w/o preprocessor)
        self.forward_mu, self.forward_mu_target = (
            ForwardModel(
                preprocessor_feature_space_dimension=kwargs["observation_length"],
                number_of_preprocessed_features=1,  # i.e. just the observation
                z_dimension=kwargs["z_dimension"],
                hidden_dimension=kwargs["forward_hidden_dimension"],
                hidden_layers=kwargs["forward_hidden_layers"],
                device=kwargs["device"],
                activation=kwargs["forward_activation"],
            )
            for _ in range(2))  # to create both main network and target
