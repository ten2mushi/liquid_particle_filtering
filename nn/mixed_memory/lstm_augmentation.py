"""LSTM augmentation for particle filter cells.

Provides mixed memory architecture by combining LSTM preprocessing
with any particle filter cell. The LSTM provides deterministic
memory while the PF cell provides probabilistic inference.
"""

from typing import Tuple, Optional, Union, Any
import math

import torch
import torch.nn as nn
from torch import Tensor


class LSTMAugmentedPFCell(nn.Module):
    """LSTM-augmented particle filter cell.

    Combines an LSTM for deterministic memory with any particle filter
    cell for probabilistic inference. The LSTM preprocesses inputs
    and its output is fed to the PF cell.

    Architecture:
        x_t -> LSTM -> lstm_out -> PF_Cell -> output, particles

    This provides:
    - Deterministic long-term memory via LSTM
    - Probabilistic state inference via PF cell
    - Best of both worlds for complex sequential tasks

    Example:
        >>> from pfncps.torch.state_level import PFCfCCell
        >>> pf_cell = PFCfCCell(input_size=64, hidden_size=64, n_particles=32)
        >>> cell = LSTMAugmentedPFCell(
        ...     pf_cell=pf_cell,
        ...     input_size=20,
        ...     lstm_hidden_size=64,
        ... )
        >>> x = torch.randn(8, 20)
        >>> output, hx = cell(x)
    """

    def __init__(
        self,
        pf_cell: nn.Module,
        input_size: int,
        lstm_hidden_size: int,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.0,
        bidirectional: bool = False,
        fusion_mode: str = "concat",
    ):
        """Initialize LSTM-augmented PF cell.

        Args:
            pf_cell: Any particle filter cell (state-level, param-level, dual, SDE)
            input_size: Dimension of input
            lstm_hidden_size: Hidden size of LSTM
            lstm_layers: Number of LSTM layers
            lstm_dropout: LSTM dropout
            bidirectional: Use bidirectional LSTM (only for offline processing)
            fusion_mode: How to combine LSTM output with PF ('concat', 'add', 'gate')
        """
        super().__init__()

        self.pf_cell = pf_cell
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.fusion_mode = fusion_mode
        self.bidirectional = bidirectional

        # LSTM cell for step-by-step processing
        # Note: Using LSTMCell for compatibility with PF cell's step-by-step interface
        self.lstm_cell = nn.LSTMCell(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
        )

        # Additional LSTM layers if needed
        self.additional_lstm_cells = nn.ModuleList()
        for i in range(1, lstm_layers):
            self.additional_lstm_cells.append(
                nn.LSTMCell(lstm_hidden_size, lstm_hidden_size)
            )

        # Fusion layer
        if fusion_mode == "concat":
            # PF cell receives [lstm_out, input]
            expected_pf_input = lstm_hidden_size + input_size
        elif fusion_mode == "add":
            # Sizes must match
            if lstm_hidden_size != input_size:
                self.input_proj = nn.Linear(input_size, lstm_hidden_size)
            expected_pf_input = lstm_hidden_size
        elif fusion_mode == "gate":
            self.gate = nn.Linear(lstm_hidden_size + input_size, lstm_hidden_size)
            expected_pf_input = lstm_hidden_size
        else:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")

        # Verify PF cell input size matches
        pf_input_size = getattr(pf_cell, 'input_size', None)
        if pf_input_size is not None and pf_input_size != expected_pf_input:
            # Add projection if needed
            self.pf_input_proj = nn.Linear(expected_pf_input, pf_input_size)
        else:
            self.pf_input_proj = None

        # Dropout
        self.dropout = nn.Dropout(lstm_dropout) if lstm_dropout > 0 else None

    def init_lstm_hidden(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Initialize LSTM hidden state.

        Returns:
            (h, c): LSTM hidden and cell states, each [layers, batch, hidden]
        """
        h = torch.zeros(
            self.lstm_layers, batch_size, self.lstm_hidden_size,
            device=device, dtype=dtype,
        )
        c = torch.zeros(
            self.lstm_layers, batch_size, self.lstm_hidden_size,
            device=device, dtype=dtype,
        )
        return h, c

    def _fuse_inputs(
        self,
        lstm_out: Tensor,
        input: Tensor,
    ) -> Tensor:
        """Fuse LSTM output with input for PF cell.

        Args:
            lstm_out: LSTM output [batch, lstm_hidden]
            input: Original input [batch, input_size]

        Returns:
            fused: Fused input for PF cell
        """
        if self.fusion_mode == "concat":
            fused = torch.cat([lstm_out, input], dim=-1)
        elif self.fusion_mode == "add":
            if hasattr(self, 'input_proj'):
                input = self.input_proj(input)
            fused = lstm_out + input
        elif self.fusion_mode == "gate":
            combined = torch.cat([lstm_out, input], dim=-1)
            gate = torch.sigmoid(self.gate(combined))
            fused = gate * lstm_out
        else:
            fused = lstm_out

        if self.pf_input_proj is not None:
            fused = self.pf_input_proj(fused)

        return fused

    def forward(
        self,
        input: Tensor,
        hx: Optional[Tuple[Any, Tuple[Tensor, Tensor]]] = None,
        timespans: Optional[Tensor] = None,
        observation: Optional[Tensor] = None,
        return_all_particles: bool = False,
    ) -> Tuple[Tensor, Tuple[Any, Tuple[Tensor, Tensor]]]:
        """Forward pass through LSTM-augmented PF cell.

        Args:
            input: Input tensor [batch, input_size]
            hx: Tuple of (pf_state, lstm_state)
                - pf_state: PF cell hidden state (varies by cell type)
                - lstm_state: (h, c) LSTM states
            timespans: Optional time deltas
            observation: Optional observation for weight update
            return_all_particles: Return all particle outputs

        Returns:
            output: Cell output
            (pf_state, lstm_state): Updated hidden states
        """
        batch_size = input.shape[0]
        device = input.device
        dtype = input.dtype

        # Initialize states
        if hx is None:
            pf_state = None
            h, c = self.init_lstm_hidden(batch_size, device, dtype)
        else:
            pf_state, (h, c) = hx

        # Process through LSTM layers
        lstm_input = input
        new_h = []
        new_c = []

        for layer in range(self.lstm_layers):
            if layer == 0:
                h_l, c_l = self.lstm_cell(lstm_input, (h[0], c[0]))
            else:
                h_l, c_l = self.additional_lstm_cells[layer - 1](
                    lstm_input, (h[layer], c[layer])
                )

            new_h.append(h_l)
            new_c.append(c_l)

            lstm_input = h_l
            if self.dropout is not None and layer < self.lstm_layers - 1:
                lstm_input = self.dropout(lstm_input)

        lstm_out = new_h[-1]
        h = torch.stack(new_h, dim=0)
        c = torch.stack(new_c, dim=0)

        # Fuse LSTM output with input for PF cell
        pf_input = self._fuse_inputs(lstm_out, input)

        # Forward through PF cell
        output, pf_state = self.pf_cell(
            pf_input,
            pf_state,
            timespans=timespans,
            observation=observation,
            return_all_particles=return_all_particles,
        )

        return output, (pf_state, (h, c))

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, lstm_hidden={self.lstm_hidden_size}, "
            f"lstm_layers={self.lstm_layers}, fusion={self.fusion_mode}"
        )


class MixedMemoryPFCell(nn.Module):
    """Mixed memory cell with parallel LSTM and PF pathways.

    Unlike LSTMAugmentedPFCell which is sequential (LSTM -> PF),
    this processes inputs through both pathways in parallel and
    combines their outputs.

    Architecture:
        x_t -> [LSTM -> lstm_out]
            -> [PF_Cell -> pf_out]
            -> Combine -> output

    Example:
        >>> from pfncps.torch.state_level import PFCfCCell
        >>> pf_cell = PFCfCCell(input_size=20, hidden_size=64, n_particles=32)
        >>> cell = MixedMemoryPFCell(
        ...     pf_cell=pf_cell,
        ...     input_size=20,
        ...     lstm_hidden_size=64,
        ...     combine_mode='concat',
        ... )
    """

    def __init__(
        self,
        pf_cell: nn.Module,
        input_size: int,
        lstm_hidden_size: int,
        lstm_layers: int = 1,
        combine_mode: str = "concat",
        output_size: Optional[int] = None,
    ):
        """Initialize mixed memory PF cell.

        Args:
            pf_cell: Any particle filter cell
            input_size: Dimension of input
            lstm_hidden_size: Hidden size of LSTM
            lstm_layers: Number of LSTM layers
            combine_mode: How to combine outputs ('concat', 'add', 'weighted')
            output_size: Final output size (required for 'concat' mode)
        """
        super().__init__()

        self.pf_cell = pf_cell
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.combine_mode = combine_mode

        # LSTM pathway
        self.lstm_cell = nn.LSTMCell(input_size, lstm_hidden_size)
        self.additional_lstm_cells = nn.ModuleList([
            nn.LSTMCell(lstm_hidden_size, lstm_hidden_size)
            for _ in range(lstm_layers - 1)
        ])

        # Get PF output size
        pf_hidden = getattr(pf_cell, 'hidden_size', lstm_hidden_size)
        pf_output_size = getattr(pf_cell, 'output_size', pf_hidden)

        # Combination layer
        if combine_mode == "concat":
            if output_size is None:
                output_size = lstm_hidden_size + pf_output_size
            self.output_proj = nn.Linear(
                lstm_hidden_size + pf_output_size, output_size
            )
        elif combine_mode == "add":
            # Project to same size
            self.lstm_proj = nn.Linear(lstm_hidden_size, pf_output_size)
            output_size = pf_output_size
        elif combine_mode == "weighted":
            self.weight_net = nn.Linear(lstm_hidden_size + pf_output_size, 2)
            output_size = max(lstm_hidden_size, pf_output_size)
            if lstm_hidden_size != pf_output_size:
                self.lstm_proj = nn.Linear(lstm_hidden_size, pf_output_size)

        self.output_size = output_size

    def init_hidden(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[None, Tuple[Tensor, Tensor]]:
        """Initialize hidden states."""
        h = torch.zeros(
            self.lstm_layers, batch_size, self.lstm_hidden_size,
            device=device, dtype=dtype,
        )
        c = torch.zeros(
            self.lstm_layers, batch_size, self.lstm_hidden_size,
            device=device, dtype=dtype,
        )
        return None, (h, c)

    def forward(
        self,
        input: Tensor,
        hx: Optional[Tuple[Any, Tuple[Tensor, Tensor]]] = None,
        timespans: Optional[Tensor] = None,
        observation: Optional[Tensor] = None,
        return_all_particles: bool = False,
    ) -> Tuple[Tensor, Tuple[Any, Tuple[Tensor, Tensor]]]:
        """Forward pass."""
        batch_size = input.shape[0]
        device = input.device
        dtype = input.dtype

        if hx is None:
            pf_state, (h, c) = self.init_hidden(batch_size, device, dtype)
        else:
            pf_state, (h, c) = hx

        # LSTM pathway
        lstm_input = input
        new_h, new_c = [], []

        for layer in range(self.lstm_layers):
            if layer == 0:
                h_l, c_l = self.lstm_cell(lstm_input, (h[0], c[0]))
            else:
                h_l, c_l = self.additional_lstm_cells[layer - 1](
                    lstm_input, (h[layer], c[layer])
                )
            new_h.append(h_l)
            new_c.append(c_l)
            lstm_input = h_l

        lstm_out = new_h[-1]
        h = torch.stack(new_h, dim=0)
        c = torch.stack(new_c, dim=0)

        # PF pathway
        pf_out, pf_state = self.pf_cell(
            input, pf_state, timespans=timespans,
            observation=observation, return_all_particles=False,
        )

        # Combine outputs
        if self.combine_mode == "concat":
            combined = torch.cat([lstm_out, pf_out], dim=-1)
            output = self.output_proj(combined)
        elif self.combine_mode == "add":
            lstm_proj = self.lstm_proj(lstm_out)
            output = lstm_proj + pf_out
        elif self.combine_mode == "weighted":
            combined = torch.cat([lstm_out, pf_out], dim=-1)
            weights = torch.softmax(self.weight_net(combined), dim=-1)
            if hasattr(self, 'lstm_proj'):
                lstm_out = self.lstm_proj(lstm_out)
            output = weights[:, 0:1] * lstm_out + weights[:, 1:2] * pf_out

        return output, (pf_state, (h, c))
