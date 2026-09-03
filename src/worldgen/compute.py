import taichi as ti


def initialize_compute() -> None:
    ti.init(arch=ti.gpu)
