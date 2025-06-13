try:
    from shred_s4_benchmark.external.s4.models.s4.s4d import S4D
except ImportError:
    print("Please run setup_external.py with python -m shred_s4_benchmark.setup_external before running any other "
          "Python files")
    exit(1)

from torch import nn


class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output,
        lr,
        d_model=256,
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
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, lr))
            )

            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        """
        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)
        """

        # Select last output
        x = x[:, -1, :]

        # Decode the output
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x


class S4wDecoder(nn.Module):

    def __init__(
            self,
            d_input,
            d_output,
            lr,
            d_model=256,
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
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, lr))
            )

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
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        """
        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)
        """

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
