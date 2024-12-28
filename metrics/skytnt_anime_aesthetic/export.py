import argparse
import torch

from anime_aesthetic import AnimeAesthetic, model_cfgs


def export_onnx(model, img_size, path):
    import onnx
    from onnxsim import simplify
    torch.onnx.export(model,  # model being run
                      torch.randn(1, 3, img_size, img_size),  # model input (or a tuple for multiple inputs)
                      path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=["img"],  # the model's input names
                      output_names=["score"],  # the model's output names
                      verbose=True
                      )
    onnx_model = onnx.load(path)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, path)
    print('finished exporting onnx')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument(
        "--cfg",
        type=str,
        default="tiny",
        choices=list(model_cfgs.keys()),
        help="model configure",
    )
    parser.add_argument('--ckpt', type=str, default='lightning_logs/version_11/checkpoints/last.ckpt',
                        help='model checkpoint path')
    parser.add_argument('--out', type=str, default='model.onnx',
                        help='output path')
    parser.add_argument('--to', type=str, default='onnx', choices=["onnx"],
                        help='export to ()')
    parser.add_argument('--img-size', type=int, default=768,
                        help='input image size')
    opt = parser.parse_args()
    print(opt)

    model = AnimeAesthetic.load_from_checkpoint(opt.ckpt, cfg=opt.cfg, ema_decay=0.999, map_location="cpu",strict=False)
    model = model.eval()
    if opt.to == "onnx":
        export_onnx(model, opt.img_size, opt.out)
