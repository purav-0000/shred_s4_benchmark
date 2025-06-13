from mamba_ssm import Mamba, Mamba2
from torch import nn


class MambaModel(nn.Module):

    def __init__(
        self,
        d_input,
        d_output,
        d_conv=4,
        d_model=256,
        d_state=64,
        expand=2,
        n_layers=2,
        dropout=0,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.mamba_layers.append(Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            ))

            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, model_dim)

        for mamba, norm, dropout in zip(self.mamba_layers, self.norm_layers, self.dropout_layers):
            residual = x

            if self.use_prenorm:
                x = norm(x)

            x = mamba(x)  # Mamba expects (B, L, D)
            x = dropout(x)
            x = x + residual  # Residual connection

            if not self.use_prenorm:
                x = norm(x)

        x = x[:, -1, :]  # Select last output
        return self.decoder(x) # -> (B, output_dim)


class MambawDecoder(nn.Module):

    def __init__(
            self,
            d_input,
            d_output,
            d_conv=4,
            d_model=256,
            d_state=64,
            expand=2,
            l1=350,
            l2=400,
            n_layers=2,
            dropout=0,
            dropout_decoder=0,
            prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.mamba_layers.append(Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            ))

            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))

        # Linear decoder
        self.linear1 = nn.Linear(d_model, l1)
        self.linear2 = nn.Linear(l1, l2)
        self.linear3 = nn.Linear(l2, d_output)
        self.dropout_decoder = nn.Dropout(dropout_decoder)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, model_dim)

        for mamba, norm, dropout in zip(self.mamba_layers, self.norm_layers, self.dropout_layers):
            residual = x

            if self.use_prenorm:
                x = norm(x)

            x = mamba(x)  # Mamba expects (B, L, D)
            x = dropout(x)
            x = x + residual  # Residual connection

            if not self.use_prenorm:
                x = norm(x)

        # Select last output
        x = x[:, -1, :]

        # Decode the output
        x = self.linear1(x)
        x = self.dropout_decoder(x)
        x = nn.functional.relu(x)

        x = self.linear2(x)
        x = self.dropout_decoder(x)
        x = nn.functional.relu(x)

        x = self.linear3(x)

        return x
