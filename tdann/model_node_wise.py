"""
   Copyright 2021 Sai Muttavarapu
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0 \

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import argparse
import onnx
import onnxruntime
import numpy as np
import constants as cnts
import loading_pre_req as lp


def node_data(model: onnx.onnx_ml_pb2.ModelProto) -> str:
    """
    Return the node inputs and outputs.
    """
    node_inputs = []
    node_outputs = []
    for i, _ in enumerate(model.graph.node):
        node_inputs.append(str(model.graph.node[i].input[0]))
        node_outputs.append(str(model.graph.node[i].output[0]))
    return node_inputs, node_outputs


def save_node_models(
    model_path: str, node_input_names: int, node_output_names: int
) -> None:
    """
    Save the node wise models in pre-existing folder.
    """
    for i, _ in enumerate(node_input_names):
        input_path = model_path
        output_path = (
            "BasicShapeSavedModels/save/model_node_" + "%d" + ".onnx") % i
        input_names = [node_input_names[i]]
        output_names = [node_output_names[i]]
        onnx.utils.extract_model(
            input_path, output_path, input_names, output_names)


def extract_node_outputs(model, batch_data):
    """
    extract the outputs of each node buy passing previous node data as input.
    """
    model_node_output = []
    for i, _ in enumerate(model.graph.node):
        model_node_path = (
            "BasicShapeSavedModels/save/model_node_" + "%d" + ".onnx"
        ) % i
        session = onnxruntime.InferenceSession(model_node_path)
        input_model = {session.get_inputs()[0].name: batch_data}
        output = session.run(None, input_model)
        model_node_output.append(output)
        batch_data = output[0]
        del session
    return model_node_output


def averages(model_node_output):
    """
    averages all data for the batch.
    """
    avg_ouputs = []
    for data in model_node_output:
        avg_val = np.mean(data[0], axis=0)
        avg_ouputs.append(avg_val)
    return avg_ouputs


def save_data(data, outpu_path) -> None:
    """
    save the data in numpy format.
    """
    np.save(outpu_path + "/" + "activations.npy", data)


def main():
    """
    main method.
    """
    activations = []
    neurons_batch_data = []

    parser = argparse.ArgumentParser(description="TDANN")
    parser.add_argument(
        "-m",
        "--model_loc",
        action="store",
        dest="model_path",
        help="The model location.",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--data_loc",
        action="store",
        dest="data_path",
        help="The data file location.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_loc",
        action="store",
        dest="output_path",
        help="the output file location.",
        required=True,
    )

    args = parser.parse_args()
    model = lp.load_network(args.model_path)
    data_val = lp.load_datafile(args.data_path)

    node_input_names, node_output_names = node_data(model)

    save_node_models(args.model_path, node_input_names, node_output_names)

    for i in range(int(data_val.shape[0] / cnts.BATCH_SIZE)):

        batch_data = data_val[
            i * cnts.BATCH_SIZE: cnts.BATCH_SIZE * (i + 1), 0: cnts.DIMENSION
        ]

        model_layers_output = extract_node_outputs(model, batch_data)

        layer_data = []
        model_layers_output = averages(model_layers_output)

        layer_data = [
            data_val for data in model_layers_output for data_val in data
            ]

        activations.append(layer_data)

    neurons_batch_data = np.transpose(np.array(activations), axes=None)

    save_data(neurons_batch_data, args.output_path)


if __name__ == "__main__":

    main()
