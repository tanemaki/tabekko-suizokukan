import copy

import dimod
import numpy as np


def softmax(xs: np.ndarray):
    """
    ソフトマックス関数
    """
    factors = np.exp(xs - np.max(xs))
    return factors / np.sum(factors)


def energies2probabilities(energies, beta):
    """
    エネルギーを確率に変換
    """
    xs = -beta * np.array(energies)
    return softmax(xs)


class TabekkoSuizokukanEstimator:
    def __init__(
        self,
        qubo,
        unique_class_ids,
        nodes_per_class,
        edge_length,
        tabekko_df,
        beta=0.1,
    ):
        self.qubo = qubo
        self.unique_class_ids = unique_class_ids
        self.nodes_per_class = nodes_per_class
        self.edge_length = edge_length
        self.tabekko_df = tabekko_df
        self.beta = beta

        self.xtick_labels = [
            f"{i}: {self.tabekko_df.loc[i]['japanese_name']}"
            for i in self.unique_class_ids
        ]

        # 使用可能な動物ID（クラスID）の数（学習時に使われた動物の数）
        self.C = len(unique_class_ids)

        self.bqm = dimod.BinaryQuadraticModel.from_qubo(self.qubo)

    def estimate(self, image):
        binary_vector = self.get_binary_vector(image)
        energies = self.get_energies(binary_vector)
        probabilities = self.get_probabilities(energies)
        estimated_class_index = np.argmax(probabilities)
        estimated_class_id = self.unique_class_ids[estimated_class_index]
        estimated_class_name = self.tabekko_df.loc[estimated_class_id]["japanese_name"]
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_class_ids = self.unique_class_ids[sorted_indices]
        sorted_class_names = self.tabekko_df.loc[sorted_class_ids]["japanese_name"]
        sorted_probabilities = probabilities[sorted_indices]

        return {
            "image": image,
            "binary_vector": binary_vector,
            "energies": energies,
            "probabilities": probabilities,
            "estimated_class_id": estimated_class_id,
            "estimated_class_name": estimated_class_name,
            "estimated_class_index": estimated_class_index,
            "sorted_indices": sorted_indices,
            "sorted_class_ids": sorted_class_ids,
            "sorted_class_names": sorted_class_names,
            "sorted_probabilities": sorted_probabilities,
        }

    def get_class_index(self, class_id):
        """
        クラスIDからクラスインデックスを取得
        """
        return int(np.where(self.unique_class_ids == class_id)[0][0])

    def get_binary_vector(self, image):
        x = image.flatten()

        return np.where(x >= 128, 1, 0)

    def get_energies(self, selected_binary_input):
        fixed_input_variables = {
            i: np.int8(pixel_value)
            for i, pixel_value in enumerate(selected_binary_input)
        }

        energies = []

        for c in range(self.C):
            selected_binary_output = np.zeros(self.C * self.nodes_per_class)
            selected_binary_output[
                c * self.nodes_per_class : (c + 1) * self.nodes_per_class
            ] = 1

            fixed_output_variables = {
                i + self.edge_length * self.edge_length: np.int8(pixel_value)
                for i, pixel_value in enumerate(selected_binary_output)
            }

            fixed_variables = copy.deepcopy(fixed_input_variables)

            fixed_variables = {**fixed_input_variables, **fixed_output_variables}

            energies.append(self.bqm.energy(fixed_variables))

        return energies

    def get_probabilities(self, energies):
        probabilities = energies2probabilities(energies, beta=self.beta)
        return probabilities
