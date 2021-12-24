"""

This script provides
various exploration on onnx model
such as those operations:

- summary:
    this command will summarize model info mation such as:
    1. model opset version;
    2. if check pass;
    3. whether can be simplifiered or not;
    4. inputs node and outputs node;
    5. all nodes number;
    6. Initializers tensornumber and their index;

- search:
    1. search all nodes by OpType;
    2. search node by node name (ID);

- list:
    1. print out all nodes id
    2. list -hl will print all nodes and it's attribute

"""
import os
import sys
import argparse
import onnx
from onnx import ModelProto, TensorProto, NodeProto
from typing import Optional, cast, Text, IO
from google import protobuf
from colorama import init, Fore, Style, Back
from .core import *
from .totrt import convert_onnx_to_tensorrt

from alfred.utils.log import logger as logging

try:
    import onnxruntime as ort
except ImportError:
    logging.error("onnxexp needs onnxruntime, you need install onnxruntime first.")
    exit(0)

init()


def arg_parse():
    """
    parse arguments
    :return:
    """
    parser = argparse.ArgumentParser(prog="onnxexp")
    parser.add_argument(
        "--version", "-v", action="store_true", help="show version info."
    )

    main_sub_parser = parser.add_subparsers(dest="subparser_name")

    # =============== glance part ================
    vision_parser = main_sub_parser.add_parser(
        "glance", help="Take a glance at your onnx model."
    )
    vision_parser.add_argument("--model", "-m", help="onnx model path")
    vision_parser.add_argument(
        "--verbose", "-v", action="store_true", help="verbose info"
    )

    # =============== totrt part ================
    trt_parser = main_sub_parser.add_parser(
        "totrt", help="Convert your onnx model to tensorrt engine."
    )
    trt_parser.add_argument("--model", "-m", help="onnx model path")
    trt_parser.add_argument(
        "--data_type", "-d", type=int, default=32, help="32, 16, 8 presents fp32, fp16, int8"
    )
    trt_parser.add_argument("--min_shapes", nargs='+', help="min_shapes of opt_params")
    trt_parser.add_argument("--opt_shapes", nargs='+', help="opt_shapes of opt_params")
    trt_parser.add_argument("--max_shapes", nargs='+', help="max_shapes of opt_params")

    # =============== check part ================
    check_parser = main_sub_parser.add_parser(
        "check", help="Check your onnx model is valid or not."
    )
    check_parser.add_argument("--model", "-m", help="onnx model path")
    check_parser.add_argument("--print", action="store_true")

    return parser.parse_args()


__VERSION__ = "👍    0.2.0"
__AUTHOR__ = "😀    Lucas Jin"
__CONTACT__ = "😍    telegram: lucasjin"
__DATE__ = "👉    2022.01.01, since 2019.11.11"
__LOC__ = "👉    Shenzhen, China"
__git__ = "👍    http://github.com/jinfagang/onnxexplorer"


def print_welcome_msg():
    print("-" * 70)
    print(
        Fore.BLUE
        + Style.BRIGHT
        + "              onnxexp "
        + Style.RESET_ALL
        + Fore.WHITE
        + "- Your free Onnx explorer, debug AI."
        + Style.RESET_ALL
    )
    print(
        "         Author : " + Fore.CYAN + Style.BRIGHT + __AUTHOR__ + Style.RESET_ALL
    )
    print(
        "         Contact: " + Fore.BLUE + Style.BRIGHT + __CONTACT__ + Style.RESET_ALL
    )
    print(
        "         At     : "
        + Fore.LIGHTGREEN_EX
        + Style.BRIGHT
        + __DATE__
        + Style.RESET_ALL
    )
    print(
        "         Loc    : "
        + Fore.LIGHTMAGENTA_EX
        + Style.BRIGHT
        + __LOC__
        + Style.RESET_ALL
    )
    print(
        "         Star   : " + Fore.MAGENTA + Style.BRIGHT + __git__ + Style.RESET_ALL
    )
    print(
        "         Ver.   : " + Fore.GREEN + Style.BRIGHT + __VERSION__ + Style.RESET_ALL
    )
    print("-" * 70)
    print("\n")


class ONNXExplorer(object):
    def __init__(self):
        args = arg_parse()
        if args.version:
            print_welcome_msg()
        else:
            print(args)
            if args.subparser_name == None:
                print("should provide at least one sub command, -h for detail.")
                exit(-1)
            self.model_path = args.model
            if args.model != None and os.path.exists(args.model):
                print(
                    Style.BRIGHT
                    + "Exploring on onnx model: "
                    + Style.RESET_ALL
                    + Fore.GREEN
                    + self.model_path
                    + Style.RESET_ALL
                )
                # self.model_proto = load_onnx_model(self.model_path, ModelProto())
                self.model_proto = onnx.load(self.model_path)
                if args.subparser_name == "glance":
                    summary(self.model_proto, self.model_path, args.verbose)
                elif args.subparser_name == "totrt":
                    convert_onnx_to_tensorrt(self.model_path, args.data_type, args.min_shapes, args.opt_shapes, args.max_shapes)
                elif args.subparser_name == "check":
                    self.check(args.print)

            else:
                print(
                    "{} does not exist or you should provide model path like `onnxexp glance -m model.onnx`.".format(
                        args.model
                    )
                )

    def ls(self):
        parser = argparse.ArgumentParser(description="ls at any level about the model.")
        # prefixing the argument with -- means it's optional
        parser.add_argument("-hl", action="store_true")
        args = parser.parse_args(sys.argv[3:])
        print("Running onnxexp ls, hl=%s" % args.hl)
        if args.hl:
            list_nodes_hl(self.model_proto)
        else:
            list_nodes(self.model_proto)

    def search(self):
        parser = argparse.ArgumentParser(
            description="search model node by name or type"
        )
        # NOT prefixing the argument with -- means it's not optional
        parser.add_argument("--name", "-n", help="name of to search node")
        parser.add_argument("--type", "-t", help="type of to search node")
        args = parser.parse_args(sys.argv[3:])
        if args.name != None:
            search_node_by_id(self.model_proto, args.name)
        elif args.type != None:
            search_nodes_by_type(self.model_proto, args.type)
        else:
            print("search should provide type name or id to search.")

    def check(self, p):
        logging.info("checking on model: {}".format(self.model_path))
        logging.info("this command will check an onnx model is valid or not.")
        onnx.checker.check_model(self.model_proto)
        if p:
            a = onnx.helper.printable_graph(self.model_proto.graph)
            print(a)


def main():
    ONNXExplorer()


if __name__ == "__main__":
    main()
