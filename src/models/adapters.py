import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GATConv, GlobalAttention, GraphNorm, GINEConv,
                                GCNConv, BatchNorm, AttentionalAggregation,
                                global_mean_pool, global_add_pool,
                                global_max_pool, SAGPooling)


class GATAdapter_old(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.prefix_length = kwargs['prefix_length']
        self.mbart_dim = kwargs['mbart_dim']
        clip_dim = kwargs['clip_dim']
        edge_attr_dim = clip_dim

        hidden_channels = kwargs.get('hidden_channels', 1024)
        num_heads = kwargs.get('num_heads', 8)

        self.gat1 = GATConv(in_channels=clip_dim,
                            out_channels=hidden_channels,
                            heads=num_heads,
                            concat=False,
                            dropout=0.1,
                            edge_dim=edge_attr_dim)
        self.prelu1 = nn.PReLU()

        self.gat2 = GATConv(in_channels=hidden_channels,
                            out_channels=self.prefix_length * self.mbart_dim,
                            heads=num_heads,
                            concat=False,
                            dropout=0.1,
                            edge_dim=edge_attr_dim)
        self.prelu2 = nn.PReLU()

        # Commented out SAGPooling
        # self.sag_pool2 = SAGPooling(
        #     self.prefix_length * self.mbart_dim, ratio=0.5, nonlinearity='tanh'
        # )

        # AttentionalAggregation for final graph-level aggregation
        self.attention_pool = AttentionalAggregation(gate_nn=nn.Sequential(
            nn.Linear(self.prefix_length * self.mbart_dim, hidden_channels),
            nn.PReLU(), nn.Linear(hidden_channels, hidden_channels),
            nn.PReLU(), nn.Linear(hidden_channels, 1)))

    def forward(self, data, clip):
        x, edge_index, edge_attr, batch = (data.x, data.edge_index,
                                           data.edge_attr, data.batch)

        x = self.gat1(x, edge_index, edge_attr)
        x = self.prelu1(x)

        x = self.gat2(x, edge_index, edge_attr)
        x = self.prelu2(x)

        # Commented out SAGPooling
        # x, edge_index, edge_attr, batch, _, _ = self.sag_pool2(
        #     x, edge_index, edge_attr, batch
        # )

        # Apply AttentionalAggregation
        x = self.attention_pool(x, batch)

        # Remove global_mean_pool if using attention_pool
        # x = global_mean_pool(x, batch)

        return x.view(-1, self.prefix_length, self.mbart_dim)


class GATAdapter(nn.Module):

    def __init__(self, **kwargs):
        super(GATAdapter, self).__init__()
        self.prefix_length = kwargs['prefix_length']
        self.mbart_dim = kwargs['mbart_dim']
        clip_dim = kwargs['clip_dim']

        self.use_fusion = kwargs.get('use_fusion', False)
        self.use_gate = kwargs.get('use_gate', True)

        edge_attr_dim = clip_dim

        hidden_channels = kwargs.get('hidden_channels', 1024)
        num_heads = kwargs.get('num_heads', 8)

        self.gat1 = GATConv(in_channels=clip_dim,
                            out_channels=hidden_channels,
                            heads=num_heads,
                            concat=False,
                            dropout=0.1,
                            edge_dim=edge_attr_dim)
        self.prelu1 = nn.PReLU()

        self.gat2 = GATConv(in_channels=hidden_channels,
                            out_channels=self.prefix_length * self.mbart_dim,
                            heads=num_heads,
                            concat=False,
                            dropout=0.1,
                            edge_dim=edge_attr_dim)
        self.prelu2 = nn.PReLU()

        # AttentionalAggregation for final graph-level aggregation
        self.attention_pool = AttentionalAggregation(gate_nn=nn.Sequential(
            nn.Linear(self.prefix_length * self.mbart_dim, hidden_channels),
            nn.PReLU(), nn.Linear(hidden_channels, hidden_channels),
            nn.PReLU(), nn.Linear(hidden_channels, 1)))

        if self.use_fusion:
            self.fusion = CrossAttentionFusionLayer(hidden_size=self.mbart_dim,
                                                    num_heads=kwargs.get(
                                                        'fusion_num_heads',
                                                        16),
                                                    dropout=0.1,
                                                    use_gate=self.use_gate)

    def forward(self, data, encoder_outputs=None):
        x, edge_index, edge_attr, batch = (data.x, data.edge_index,
                                           data.edge_attr, data.batch)

        x = self.gat1(x, edge_index, edge_attr)
        x = self.prelu1(x)

        x = self.gat2(x, edge_index, edge_attr)
        x = self.prelu2(x)

        # Apply AttentionalAggregation
        prefix_emb = self.attention_pool(x, batch)
        prefix_emb = prefix_emb.view(-1, self.prefix_length, self.mbart_dim)

        # Cross-attention fusion
        if self.use_fusion and encoder_outputs is not None:
            # Cross-attention fusion
            fused_hidden_states = self.fusion(
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                prefix_emb=prefix_emb)
            encoder_outputs.last_hidden_state = fused_hidden_states
            return encoder_outputs
        else:
            return prefix_emb


class GATAdapterLarge(nn.Module):

    def __init__(self, **kwargs):
        super(GATAdapterLarge, self).__init__()
        self.prefix_length = kwargs['prefix_length']
        self.mbart_dim = kwargs['mbart_dim']
        clip_dim = kwargs['clip_dim']

        self.use_fusion = kwargs.get('use_fusion', True)
        self.use_gate = kwargs.get('use_gate', True)

        edge_attr_dim = clip_dim

        num_heads = kwargs.get('num_heads', 8)
        self.num_layers = kwargs.get('num_layers', 4)  # Number of GAT layers

        # Instead of fixed hidden_channels, compute adaptive hidden dimensions
        input_dim = clip_dim
        output_dim = self.prefix_length * self.mbart_dim

        # Compute hidden dimensions for each layer
        self.hidden_dims = self.compute_hidden_dims(input_dim, output_dim,
                                                    self.num_layers)

        self.gat_layers = nn.ModuleList()
        self.graphnorm_layers = nn.ModuleList()
        self.prelu_layers = nn.ModuleList()

        self.residual_projs = nn.ModuleList()

        for i in range(self.num_layers):
            in_channels = self.hidden_dims[i]
            out_channels = self.hidden_dims[i + 1]

            # GATConv
            gat_layer = GATConv(in_channels=in_channels,
                                out_channels=out_channels,
                                heads=num_heads,
                                concat=False,
                                dropout=0.1,
                                edge_dim=edge_attr_dim)
            self.gat_layers.append(gat_layer)

            # GraphNorm
            graphnorm_layer = GraphNorm(out_channels)
            self.graphnorm_layers.append(graphnorm_layer)

            # PReLU
            prelu_layer = nn.PReLU()
            self.prelu_layers.append(prelu_layer)

            if in_channels != out_channels:
                self.residual_projs.append(nn.Linear(in_channels,
                                                     out_channels))
            else:
                self.residual_projs.append(nn.Identity())

        # AttentionalAggregation for final graph-level aggregation
        hidden_channels = self.hidden_dims[-1]  # Last hidden dimension
        self.attention_pool = AttentionalAggregation(gate_nn=nn.Sequential(
            nn.Linear(output_dim, hidden_channels), nn.PReLU(),
            nn.Linear(hidden_channels, hidden_channels), nn.PReLU(),
            nn.Linear(hidden_channels, 1)))

        if self.use_fusion:
            self.fusion = CrossAttentionFusionLayer(hidden_size=self.mbart_dim,
                                                    num_heads=kwargs.get(
                                                        'fusion_num_heads',
                                                        16),
                                                    dropout=0.1,
                                                    use_gate=self.use_gate)

    def compute_hidden_dims(self,
                            input_dim,
                            output_dim,
                            num_layers,
                            mode='fixed',
                            fixed_dim=1024):
        hidden_dims = [input_dim]
        if mode == 'arithmetic':
            for i in range(1, num_layers):
                # arithmetic progression from input_dim to output_dim
                hidden_dim = int(input_dim +
                                 (output_dim - input_dim) * i / num_layers)
                hidden_dims.append(hidden_dim)
        elif mode == 'fixed':
            if fixed_dim is None:
                fixed_dim = output_dim
            for i in range(1, num_layers):
                hidden_dims.append(fixed_dim)
        else:
            raise ValueError("Invalid mode. Choose 'arithmetic' or 'fixed'.")
        hidden_dims.append(output_dim)
        return hidden_dims

    def forward(self, data, encoder_outputs=None):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch


        for i in range(self.num_layers):

            out = self.gat_layers[i](x, edge_index, edge_attr)

            out = self.graphnorm_layers[i](out, batch)

            out = out + self.residual_projs[i](x)

            out = self.prelu_layers[i](out)

            x = out

        prefix_emb = self.attention_pool(x, batch)

        prefix_emb = prefix_emb.view(-1, self.prefix_length, self.mbart_dim)

        if self.use_fusion and encoder_outputs is not None:
            fused_hidden_states = self.fusion(
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                prefix_emb=prefix_emb)
            encoder_outputs.last_hidden_state = fused_hidden_states
            return encoder_outputs
        else:
            return prefix_emb


class CrossAttentionFusionLayer(nn.Module):

    def __init__(self, hidden_size, num_heads, dropout=0.1, use_gate=True):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                batch_first=True)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.use_gate = use_gate

        # Gate mechanism
        if use_gate:
            self.gate = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size),
                                      nn.ReLU(), nn.Linear(hidden_size, 1),
                                      nn.Sigmoid())
            self.layer_norm2 = nn.LayerNorm(hidden_size)
            self.dropout2 = nn.Dropout(dropout)

    def forward(self, encoder_hidden_states, prefix_emb):
        """
        encoder_hidden_states: (batch_size, seq_length, hidden_size) from MEnc
        prefix_emb: (batch_size, prefix_length, hidden_size) from GEnc
        """
        # Cross-Attention: queries from MEnc, keys and values from GEnc
        attn_output, _ = self.cross_attn(query=encoder_hidden_states,
                                         key=prefix_emb,
                                         value=prefix_emb,
                                         need_weights=False)

        # Add & Norm
        attn_output = self.dropout1(attn_output)
        attn_output = self.layer_norm1(encoder_hidden_states + attn_output)

        # Gate
        if self.use_gate:
            gate_input = torch.cat(
                [attn_output, encoder_hidden_states],
                dim=-1)  # (batch_size, seq_length, hidden_size * 2)
            gate_weight = self.gate(gate_input)  # (batch_size, seq_length, 1)

            fused_output = gate_weight * attn_output + (
                1 - gate_weight) * encoder_hidden_states

            # Add & Norm
            fused_output = self.dropout2(fused_output)
            fused_output = self.layer_norm2(fused_output)

            return fused_output
        else:
            return attn_output


class MLPFusionLayer(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, encoder_hidden_states, prefix_emb):
        # encoder_hidden_states: (batch_size, seq_length, hidden_size)
        # prefix_emb: (batch_size, prefix_length, hidden_size)

        # Concatenate encoder_hidden_states and prefix_emb
        fusion_output = torch.cat([prefix_emb, encoder_hidden_states], dim=1)
        fusion_output = self.linear(fusion_output)
        fusion_output = F.relu(fusion_output)

        # Add & Norm
        fusion_output = self.dropout(fusion_output)
        fusion_output = self.layer_norm(fusion_output)

        return fusion_output


class GraphAdapterLarge(nn.Module):

    def __init__(self, **kwargs):
        super(GraphAdapterLarge, self).__init__()
        self.prefix_length = kwargs['prefix_length']
        self.mbart_dim = kwargs['mbart_dim']
        clip_dim = kwargs['clip_dim']

        # Choose the mapping network: supports 'gatl', 'ginel', and 'gcnl'
        self.mapping_network = kwargs.get('mapping_network', 'gatl')
        self.use_fusion = kwargs.get('use_fusion', True)
        self.use_gate = kwargs.get('use_gate', True)

        # For GATConv, edge_attr is used; other networks will ignore this parameter
        edge_attr_dim = clip_dim

        num_heads = kwargs.get('num_heads', 8)
        self.num_layers = kwargs.get('num_layers',
                                     4)  # Number of graph convolution layers

        # Define input and output dimensions
        input_dim = clip_dim
        output_dim = self.prefix_length * self.mbart_dim

        # Compute hidden dimensions for each layer (either fixed or arithmetic progression)
        self.hidden_dims = self.compute_hidden_dims(input_dim, output_dim,
                                                    self.num_layers)

        # Define graph convolution layers, GraphNorm, PReLU, and residual projection layers
        # The list name "gat_layers" is kept for consistency even though it might include GAT/GCN/GINE layers
        self.gat_layers = nn.ModuleList()
        self.graphnorm_layers = nn.ModuleList()
        self.prelu_layers = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        for i in range(self.num_layers):
            in_channels = self.hidden_dims[i]
            out_channels = self.hidden_dims[i + 1]

            if self.mapping_network == 'gatl':
                conv_layer = GATConv(in_channels=in_channels,
                                     out_channels=out_channels,
                                     heads=num_heads,
                                     concat=False,
                                     dropout=0.1,
                                     edge_dim=edge_attr_dim)
            elif self.mapping_network == 'gcnl':
                conv_layer = GCNConv(in_channels, out_channels)
            elif self.mapping_network == 'ginel':
                # Construct a simple 2-layer MLP for GINEConv with edge_dim
                mlp = nn.Sequential(nn.Linear(in_channels, out_channels),
                                    nn.ReLU(),
                                    nn.Linear(out_channels, out_channels))
                conv_layer = GINEConv(mlp, edge_dim=edge_attr_dim)
            else:
                raise ValueError(
                    "Invalid mapping_network type. Choose from 'gatl', 'ginel', 'gcnl'."
                )
            self.gat_layers.append(conv_layer)

            # GraphNorm layer
            self.graphnorm_layers.append(GraphNorm(out_channels))
            # PReLU activation
            self.prelu_layers.append(nn.PReLU())
            # Use a linear projection for residual connection if input and output dimensions differ
            if in_channels != out_channels:
                self.residual_projs.append(nn.Linear(in_channels,
                                                     out_channels))
            else:
                self.residual_projs.append(nn.Identity())

        # Graph-level aggregation: AttentionalAggregation aggregates node features into a graph representation
        hidden_channels = self.hidden_dims[-1]
        self.attention_pool = AttentionalAggregation(gate_nn=nn.Sequential(
            nn.Linear(output_dim, hidden_channels), nn.PReLU(),
            nn.Linear(hidden_channels, hidden_channels), nn.PReLU(),
            nn.Linear(hidden_channels, 1)))

        # Keep the original Fusion module unchanged
        if self.use_fusion:
            self.fusion = CrossAttentionFusionLayer(hidden_size=self.mbart_dim,
                                                    num_heads=kwargs.get(
                                                        'fusion_num_heads',
                                                        16),
                                                    dropout=0.1,
                                                    use_gate=self.use_gate)

    def compute_hidden_dims(self,
                            input_dim,
                            output_dim,
                            num_layers,
                            mode='fixed',
                            fixed_dim=1024):
        hidden_dims = [input_dim]
        if mode == 'arithmetic':
            for i in range(1, num_layers):
                # Arithmetic progression from input_dim to output_dim
                hidden_dim = int(input_dim +
                                 (output_dim - input_dim) * i / num_layers)
                hidden_dims.append(hidden_dim)
        elif mode == 'fixed':
            if fixed_dim is None:
                fixed_dim = output_dim
            for i in range(1, num_layers):
                hidden_dims.append(fixed_dim)
        else:
            raise ValueError("Invalid mode. Choose 'arithmetic' or 'fixed'.")
        hidden_dims.append(output_dim)
        return hidden_dims

    def forward(self, data, encoder_outputs=None):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Apply graph convolution layers layer by layer (calling different forward methods depending on the mapping network)
        for i in range(self.num_layers):
            if self.mapping_network in ['gatl', 'ginel']:
                out = self.gat_layers[i](x, edge_index, edge_attr)
            elif self.mapping_network == 'gcnl':
                out = self.gat_layers[i](x, edge_index)

            out = self.graphnorm_layers[i](out, batch)
            out = out + self.residual_projs[i](x)
            out = self.prelu_layers[i](out)
            x = out

        # Graph-level aggregation
        prefix_emb = self.attention_pool(x, batch)
        # Reshape to [batch_size, prefix_length, mbart_dim]
        prefix_emb = prefix_emb.view(-1, self.prefix_length, self.mbart_dim)

        if self.use_fusion and encoder_outputs is not None:
            fused_hidden_states = self.fusion(
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                prefix_emb=prefix_emb)
            encoder_outputs.last_hidden_state = fused_hidden_states
            return encoder_outputs
        else:
            return prefix_emb


class GINEAdapterLarge(nn.Module):

    def __init__(self, **kwargs):
        super(GINEAdapterLarge, self).__init__()
        self.prefix_length = kwargs['prefix_length']
        self.mbart_dim = kwargs['mbart_dim']
        clip_dim = kwargs['clip_dim']

        self.use_fusion = kwargs.get('use_fusion', True)
        self.use_gate = kwargs.get('use_gate', True)

        edge_attr_dim = clip_dim
        self.num_layers = kwargs.get('num_layers', 4)  # Number of GINE layers

        # Input and output dimensions
        input_dim = clip_dim
        output_dim = self.prefix_length * self.mbart_dim

        # Compute hidden dimensions for each layer
        self.hidden_dims = self.compute_hidden_dims(input_dim, output_dim,
                                                    self.num_layers)

        # Define GINE + GraphNorm + PReLU
        self.gine_layers = nn.ModuleList()
        self.graphnorm_layers = nn.ModuleList()
        self.prelu_layers = nn.ModuleList()

        # Define residual projection layers (to handle dimension mismatch)
        self.residual_projs = nn.ModuleList()

        for i in range(self.num_layers):
            in_channels = self.hidden_dims[i]
            out_channels = self.hidden_dims[i + 1]

            # Build MLP for each layer
            mlp_layer = nn.Sequential(nn.Linear(in_channels, out_channels),
                                      nn.PReLU(),
                                      nn.Linear(out_channels, out_channels))

            # GINEConv
            gine_layer = GINEConv(nn=mlp_layer, edge_dim=edge_attr_dim)
            self.gine_layers.append(gine_layer)

            # GraphNorm
            graphnorm_layer = GraphNorm(out_channels)
            self.graphnorm_layers.append(graphnorm_layer)

            # PReLU
            prelu_layer = nn.PReLU()
            self.prelu_layers.append(prelu_layer)

            # Add projection layer if dimensions don't match, otherwise use identity
            if in_channels != out_channels:
                self.residual_projs.append(nn.Linear(in_channels,
                                                     out_channels))
            else:
                self.residual_projs.append(nn.Identity())

        # AttentionalAggregation for final graph-level aggregation
        hidden_channels = self.hidden_dims[-1]  # Last hidden dimension
        self.attention_pool = AttentionalAggregation(gate_nn=nn.Sequential(
            nn.Linear(output_dim, hidden_channels), nn.PReLU(),
            nn.Linear(hidden_channels, hidden_channels), nn.PReLU(),
            nn.Linear(hidden_channels, 1)))

        if self.use_fusion:
            self.fusion = CrossAttentionFusionLayer(hidden_size=self.mbart_dim,
                                                    num_heads=kwargs.get(
                                                        'fusion_num_heads',
                                                        16),
                                                    dropout=0.1,
                                                    use_gate=self.use_gate)

    def compute_hidden_dims(self,
                            input_dim,
                            output_dim,
                            num_layers,
                            mode='fixed',
                            fixed_dim=1024):
        hidden_dims = [input_dim]
        if mode == 'arithmetic':
            for i in range(1, num_layers):
                # arithmetic progression from input_dim to output_dim
                hidden_dim = int(input_dim +
                                 (output_dim - input_dim) * i / num_layers)
                hidden_dims.append(hidden_dim)
        elif mode == 'fixed':
            if fixed_dim is None:
                fixed_dim = output_dim
            for i in range(1, num_layers):
                hidden_dims.append(fixed_dim)
        else:
            raise ValueError("Invalid mode. Choose 'arithmetic' or 'fixed'.")
        hidden_dims.append(output_dim)
        return hidden_dims

    def forward(self, data, encoder_outputs=None):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Layer-wise processing: GINE + GraphNorm + Residual + Activation
        for i in range(self.num_layers):
            # 1) GINEConv
            out = self.gine_layers[i](x, edge_index, edge_attr)
            # 2) GraphNorm
            out = self.graphnorm_layers[i](out, batch)
            # 3) Residual connection: out + projected(x)
            out = out + self.residual_projs[i](x)
            # 4) Activation
            out = self.prelu_layers[i](out)
            # Update x
            x = out

        # Graph-level aggregation
        prefix_emb = self.attention_pool(x, batch)

        # Reshape to [batch_size, prefix_length, mbart_dim]
        prefix_emb = prefix_emb.view(-1, self.prefix_length, self.mbart_dim)

        if self.use_fusion and encoder_outputs is not None:
            fused_hidden_states = self.fusion(
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                prefix_emb=prefix_emb)
            encoder_outputs.last_hidden_state = fused_hidden_states
            return encoder_outputs
        else:
            return prefix_emb
