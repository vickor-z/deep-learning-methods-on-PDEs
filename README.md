# deep learning methods on PDEs
深度学习方法解微分方程

Python版本：3.7.0

Tensorflow版本：1.14.0

本库目前包含了以下三种用于求解微分方程的深度学习方法

* **Deep Galerkin Method**
* **Deep Ritz Method**
* **Local Deep Galerkin Method**



*eg_heat*文件夹包含了一个最简单的用残差型损失函数和全连接网络解热方程的例子，并将其与传统有限差分方法相比较。该例子基本包含了机器学习解微分方程编程过程中大部分需要注意的事项，可用于入门。

*gradient_scaling*程序显示了DGM方法在求解含有高阶导数方程时会出现多尺度现象。更为直观的比较可以参考CH方程中的例子，当界面参数$\epsilon=0.1$时，两种方法几乎没有差异；而当$\epsilon=0.01$时，DGM在给定迭代次数下几乎无效。

若您使用了该仓库中的程序或需要更为详细的关于求解高阶导数过程中出现的参数尺度化问题，请参阅并引用以下文献：

* **J Yang and Q Zhu. A Local Deep Learning Method for Solving High Order Partial Differential Equations. Numer. Math. Theor. Meth. Appl., doi:10.4208/nmtma.OA-2021-0035.**

  

注：该文档主要包含以下方程的数值算例

* kdv: modified Korteweg-de Vries equation

$$
u_t-6u^2u_x+u_{xxx}=0
$$

* heat: heat equation

$$
u_t=\Delta u
$$

* AC: Allen-Cahn equation

$$
u_t = \epsilon^2\Delta u+u-u^3
$$

* CH: Cahn-Hilliard equation

$$
u_t +\epsilon^2u_{xxxx}+(u-u^3)_{xx}=0
$$

* forth: an equation with 4-th order derivative

$$
u_t + u_{xxxx} +\sqrt{1+u^2} = 0
$$

* possion: Possion's equation

$$
-\Delta u = f
$$

若方程有外力项（source term），需要相应地在损失函数中加入$f(x,t)$.





