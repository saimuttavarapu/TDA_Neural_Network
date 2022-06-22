"""
   Copyright 2021 Sai Muttavarapu
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
       
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import onnx
import torch
import numpy as np
import onnxruntime
import constants as cnts


def load_network(file_path):
    """
    Load the network.
    """
    model = onnx.load(file_path)
    onnx.checker.check_model(model)
    return model


def load_datafile(file_path):
    """
    Load the numoy data file.
    """
    data_val = np.load(file_path)
    return data_val


def activation_extraction(data, model):
    print("Nodes:")

    print(model.graph.node)


def main():
    data_val = load_datafile("BasicShapeSavedModels/generated_data.npy")

    model = load_network("BasicShapeSavedModels/n_dim_sphere20210608_175419.onnx")

    activation_extraction(data_val, model)

    session = onnxruntime.InferenceSession(
        "BasicShapeSavedModels/n_dim_sphere20210608_175419.onnx"
    )

    single_val = data_val[0][0 : cnts.DIMENSION]
    input = {session.get_inputs()[0].name: [single_val]}
    output = session.run(None, input)

    print(f"input: {input}")
    print(f"Output: {np.round(output)}")


if __name__ == "__main__":
    main()
