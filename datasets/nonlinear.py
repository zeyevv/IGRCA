import numpy as np
import random
import igraph as ig
import os
from tqdm import tqdm

class Nonlinear:
    def __init__(self, options):
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.n = options['training_size'] + options['testing_size']
        self.t = options['T']
        self.m = options['m']
        self.num_vars = options['num_vars']
        self.num_envs = options['num_envs']
        self.data_dir = options['data_dir']
        self.mul = options['mul']
        self.adlength = options['adlength']
        self.adtype = options['adtype']
        self.noise_scale = options['noise_scale']
        self.dependent_features = options['dependent_features']
        self.generate_er_graph()

    def generate_er_graph(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        ig.set_random_number_generator(random.Random(self.seed))
        G_und = ig.Graph.Erdos_Renyi(n=self.num_vars, m=self.m, directed=True, loops=False)
        causal_map = np.array(G_und.get_adjacency().data).T
        np.fill_diagonal(causal_map, 1)
        print(causal_map)
        self.data_dict['causal_struct'] = causal_map

        ones_positions = np.argwhere(causal_map == 1)
        total_ones = len(ones_positions)
        int_map = np.zeros((self.num_envs, self.num_vars, self.num_vars))

        for i in range(self.num_envs-1):
            selected_indices = random.sample(range(total_ones), 5)
            selected_positions = ones_positions[selected_indices]

            for pos in selected_positions:
                int_map[i, pos[0], pos[1]] = 1
        self.data_dict['intervention_struct'] = int_map

        self.data_dict['signed_causal_struct'] = None

    def generate_example(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        x_n_list = []
        x_ab_list = []
        eps_n_list = []
        eps_ab_list = []
        label_list = []

        coefficients = np.random.uniform(low=0.1, high=2.0, size=(self.num_vars, self.num_vars, 5))

        for i in tqdm(range(self.n)):

            if self.dependent_features == 1:

                covariance_matrix = np.array([
                    [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
                    [0.8, 1.0, 0.7, 0.5, 0.3, 0.2],
                    [0.6, 0.7, 1.0, 0.6, 0.4, 0.3],
                    [0.4, 0.5, 0.6, 1.0, 0.5, 0.4],
                    [0.2, 0.3, 0.4, 0.5, 1.0, 0.6],
                    [0.1, 0.2, 0.3, 0.4, 0.6, 1.0]
                ])
                mean = np.zeros(self.num_vars)
                eps = self.noise_scale * np.random.multivariate_normal(mean, covariance_matrix, size=self.t)
            else:
                eps = self.noise_scale * np.random.randn(self.t, self.num_vars)

            eps_normal = eps.copy()
            eps_anom = eps.copy()

            x = np.zeros((self.t, self.num_vars))
            x[:5] = np.random.randn(5, self.num_vars)
            x_ab = np.zeros((self.t, self.num_vars))
            x_ab[:5] = x[:5].copy()

            A_list = [self.data_dict['causal_struct'] * coefficients[:, :, lag] for lag in range(5)]

            int_list = []
            for e in range(self.num_envs-1):
                int_list_e = [(self.data_dict['causal_struct'] + 0.5 * self.data_dict['intervention_struct'][e]) * coefficients[:, :, lag] for lag in range(5)]
                int_list.append(int_list_e)
            int_list = np.stack(int_list, axis=0)

            t_p = np.random.randint(int(0.2 * self.t), int(0.8 * self.t), size=1)
            if self.adlength > 1:
                t_p = np.array([t_p[0] + j for j in range(self.adlength)])
            t_p_set = set(t_p)
            feature_count = np.random.randint(1, min(10, self.num_vars) + 1)
            feature_p = np.random.permutation(np.arange(self.num_vars))[:feature_count]
            ab = np.zeros(self.num_vars)
            ab[feature_p] += self.mul
            temp_label = np.zeros((self.t, self.num_vars))
            temp_label[t_p, feature_p] = 1

            interval = self.t // self.num_envs
            for t in range(5, self.t):

                env_idx = min(t // interval, self.num_envs - 1)

                if env_idx ==0:
                    coeff_list = A_list
                else:
                    coeff_list = int_list[env_idx - 1]
                contributions = sum(
                        coeff_list[lag].dot(np.cos(x[t - lag - 1, :] + 1)) for lag in range(5)
                    )
                x[t, :] = contributions + eps_normal[t, :]

            for t in range(5, self.t):

                if t in t_p_set:
                    if self.adtype == 'non_causal':
                        eps_anom[t, :] += ab
                    elif self.adtype == 'causal':
                        raise NotImplementedError("Causal anomaly not implemented for this dataset.")
                    else:
                        raise NotImplementedError("Invalid adtype. Expected 'non_causal' or 'causal'.")

                env_idx = min(t // interval, self.num_envs - 1)
                if env_idx ==0:
                    coeff_list = A_list
                else:
                    coeff_list = int_list[env_idx - 1]
                contributions_ab = sum(
                        coeff_list[lag].dot(np.cos(x[t - lag - 1, :] + 1)) for lag in range(5)
                    )
                x_ab[t, :] = contributions_ab + eps_anom[t, :]

            x_n_list.append(x)
            eps_n_list.append(eps_normal)
            x_ab_list.append(x_ab)
            eps_ab_list.append(eps_anom)
            label_list.append(temp_label)

        self.data_dict['x_n_list'] = np.array(x_n_list)
        self.data_dict['x_ab_list'] = np.array(x_ab_list)
        self.data_dict['eps_n_list'] = np.array(eps_n_list)
        self.data_dict['eps_ab_list'] = np.array(eps_ab_list)
        self.data_dict['label_list'] = np.array(label_list)

    def save_data(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        np.save(os.path.join(self.data_dir, 'x_n_list'), self.data_dict['x_n_list'])
        np.save(os.path.join(self.data_dir, 'x_ab_list'), self.data_dict['x_ab_list'])
        np.save(os.path.join(self.data_dir, 'eps_n_list'), self.data_dict['eps_n_list'])
        np.save(os.path.join(self.data_dir, 'eps_ab_list'), self.data_dict['eps_ab_list'])
        np.save(os.path.join(self.data_dir, 'causal_struct'), self.data_dict['causal_struct'])
        np.save(os.path.join(self.data_dir, 'label_list'), self.data_dict['label_list'])

    def load_data(self):
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'))
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'))
        self.data_dict['eps_n_list'] = np.load(os.path.join(self.data_dir, 'eps_n_list.npy'))
        self.data_dict['eps_ab_list'] = np.load(os.path.join(self.data_dir, 'eps_ab_list.npy'))
        self.data_dict['causal_struct'] = np.load(os.path.join(self.data_dir, 'causal_struct.npy'))

        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'))
        self.data_dict['signed_causal_struct'] = None
