# Autograd Engine: From Micrograd to Advanced Computational Graphs 🚀

An enhanced automatic-differentiation engine inspired by **Andrej Karpathy**’s
[Micrograd](https://github.com/karpathy/micrograd).  
I started with his minimalist `Value` class and grew it into a robust `Scalar`
class that supports tensors, extra operations, and AI-generated tests.  
Perfect both as a learning tool and as a playground for experimenting with
computational graphs in Python.

---

## 🎯 Purpose
Explore and demystify *automatic differentiation*—the backbone of deep-learning
frameworks such as PyTorch—while keeping the code short, readable, and
beginner-friendly.

---

## ✨ Features
| Category | Details |
|----------|---------|
| **Core Autograd** | Builds the computation graph and runs back-prop to compute gradients |
| **Operations** | `+`, `*`, `**`, `relu`, `tanh`, `sigmoid`, `exp`, `log`, `sin`, `cos` |
| **Tensor Support** | 1-D vectors and 2-D matrices, including element-wise ops & dot products |
| **Gradient Clipping** | Optional cap to stop exploding gradients |
| **Graph Visualization** | Render the computation graph with Graphviz |
| **Custom Ops** | Plug-in your own forward & backward functions |
| **AI-Generated Tests** | Unit tests cover ops, gradients, edge cases, and numerical checks |

---
## 📚 Inspiration
This project builds on Andrej Karpathy's Micrograd, which provides a minimalist implementation of automatic differentiation. 
I extended the original Value class to include tensor support, advanced operations, and robust testing, making it 
a versatile tool for learning and experimentation.

## 🤝 Contributing
Feel free to open issues or submit pull requests! Ideas for new operations, optimizations, or test cases are welcome.



## 🙌 Acknowledgments

Andrej Karpathy for Micrograd and his inspiring educational content.
AI tools for generating robust test cases to ensure reliability.

Happy coding and exploring the world of autograd! 🚀
