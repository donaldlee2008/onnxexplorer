from alfred.deploy.tensorrt.common import (
    build_engine_onnx_v2,
    check_engine,
)
from alfred.utils.log import logger
import os


def convert_onnx_to_tensorrt(onnx_f, datatype, min_shapes=None, opt_shapes=None, max_shapes=None):
    if min_shapes is not None and opt_shapes is not None and max_shapes is not None:
        names = [i.split(':')[0] for i in min_shapes]
        mins_shapes = [[int(ii) for ii in i.split(':')[1].split('x')] for i in min_shapes]
        opt_shapes = [[int(ii) for ii in i.split(':')[1].split('x')] for i in opt_shapes]
        max_shapes = [[int(ii) for ii in i.split(':')[1].split('x')] for i in max_shapes]
        
        opt_params = dict()
        for i in range(len(min_shapes)):
            opt_params[names[i]] = [mins_shapes[i], opt_shapes[i], max_shapes[i]]

    fp16 = False
    int8 = False
    if datatype == 16:
        fp16 = True
    elif datatype == 8:
        print("int8 not supported.")
    save_p = os.path.join(os.path.dirname(onnx_f), onnx_f.replace(".onnx", ".engine"))
    engine = build_engine_onnx_v2(
        onnx_file_path=onnx_f, engine_file_path=save_p, fp16_mode=fp16, save_engine=True, opt_params=opt_params
    )
    check_engine(engine)
    logger.info('engine saved into: {}'.format(save_p))

