# Introduction 
Suzuran Renderer is a real-time neural renderer based on OpenGL. It contains three parts: a standard deferred renderer, a python bind to generate datasets to train the neural renderer, and the neural renderer itself. The renderer is developed at [Simo-Serra Lab.](https://esslab.jp/en/) in [Waseda University](https://www.waseda.jp/top/en/), Japan.

Currently, this project is under construction for refactoring from a bunch of hacky research code to a good quality open source project.

# Compile and Requirements

Execute the following commands

```
mkdir build
cd build
cmake -GNinja ..
ninja
```

Requirements:

```
LibTorch >= 2.0.0
CUDA >= 11.7, <= 11.8
glfw >= 3.3.8
```

# Current Progress
The code itself can be compiled and executed. However, because I wrote this renderer for my Ph.D. [research project](https://github.com/ziyangz5/NeuralBakingTransparency), 
all common problems of research code (messy, hacky, and lack of documentation) currently present in this project. I plan to refactor the code by the following plan:
1. Remove the repeated part of main.cpp (a standard deferred renderer), suzuran_py.cpp (python bind of the renderer), and neural_main.cpp (the real-time neural renderer).
2. Better CMake configuration.
3. Currently, the scene variable configuration is hard coded for BATHROOM. I will move this to config files.
4. Better code readability.
5. Documentation

This procedure will be finished within 3~6 months.
