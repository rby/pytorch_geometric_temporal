import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from ..signal import StaticGraphTemporalSignal
from pathlib import Path


class GenericDatasetLoader(object):
    """Loads data from 2 files: node_values.npy and adj_mat.npy"""

    def __init__(self, data_dir: Path | str):
        data_dir = Path(data_dir)
        assert data_dir.exists(), "data directory doesn't exist"
        node_values_path = data_dir / "node_values.npy"
        assert node_values_path.exists(), f"missing node_values.npy in {data_dir}"
        adj_mat_path = data_dir / "adj_mat.npy"
        assert node_values_path.exists(), f"missing adj_mat.npy in {data_dir}"
        self._nodes_values_path = node_values_path
        self._adj_mat_path = adj_mat_path

    def _load_data(self):
        A = np.load(self._adj_mat_path)
        X = np.load(self._nodes_values_path)
        self.A = torch.from_numpy(A)
        self.X = torch.from_numpy(X)

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def _generate_task(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
        """Uses the node features of the graph and generates a feature/target
        relationship of the shape
        (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
        predicting the average traffic speed using num_timesteps_in to predict the
        traffic conditions in the next num_timesteps_out

        Args:
            num_timesteps_in (int): number of timesteps the sequence model sees
            num_timesteps_out (int): number of timesteps the sequence model has to predict
        """
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, :, i : i + num_timesteps_in]).numpy())
            target.append((self.X[:, 0, i + num_timesteps_in : j]).numpy())

        self.features = features
        self.targets = target

    def get_dataset(
        self, num_timesteps_in: int = 12, num_timesteps_out: int = 12
    ) -> StaticGraphTemporalSignal:
        """Returns data iterator for METR-LA dataset as an instance of the
        static graph temporal signal class.

        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The METR-LA traffic
                forecasting dataset.
        """
        self._load_data()
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = StaticGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )

        return dataset
