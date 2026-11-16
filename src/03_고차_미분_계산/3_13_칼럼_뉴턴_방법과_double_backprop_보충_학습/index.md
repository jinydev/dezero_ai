---
layout: docs
title: "칼럼 뉴턴 방법과 double backprop 보충 학습"
---

# 칼럼 뉴턴 방법과 double backprop 보충 학습

이번 칼럼은 제3고지의 내용을 보충합니다. 먼저 입력이 벡터인 경우의 뉴턴 방법에 대해 설명하고, 이어서 뉴턴 방법을 대체할 수 있는 또 다른 방법을 소개합니다. 마지막으로 `double backprop`의 실용적인 쓰임 예를 추가합니다. 이번 칼럼은 다소 깊이 있는 정보를 다루다 보니 수식이 많이 등장합니다. 어렵게 느껴지면 건너뛰어도 무방합니다(이번 칼럼의 내용은 이후의 단계들과 큰 관련은 없습니다).

---

## 다변수 함수의 뉴턴 방법

제3고지에서 뉴턴 방법을 구현하면서 $y = x^4 - 2x^2$ 이라는 수식의 최솟값을 뉴턴 방법으로 구했습니다. 보는 바와 같이 이 식에서 입력 변수는 $x$ 뿐입니다. 따라서 우리가 한 일은 정확히 '입력 변수가 하나(스칼라)인 경우의 뉴턴 방법을 구현'한 것입니다.

그렇다면 입력이 다차원 배열일 경우의 뉴턴 방법은 어떻게 다를까요? 여기에서는 입력 변수를 벡터 $\mathbf{x}$ 로 바꾸고 함수 $y = f(\mathbf{x})$ 에 대해 생각해봅시다. 이때 $\mathbf{x}$ 는 벡터이며 $\mathbf{x} = (x_1, x_2, \cdots, x_n)$ 형태로 $n$ 개의 원소를 갖습니다.

> **CAUTION**
> 이 책에서는 수식을 표기할 때 변수가 스칼라가 아닌 경우 $\mathbf{x}$ 처럼 굵은 글씨로 표기하고, 스칼라라면 $x$ 처럼 보통 굵기로 표기합니다.

그러면 $y = f(\mathbf{x})$ 에 대한 뉴턴 방법은 다음과 같아집니다.

$$
\mathbf{x} \leftarrow \mathbf{x} - [\nabla^2 f(\mathbf{x})]^{-1} \nabla f(\mathbf{x})
$$

 \hfill [식 C.1]

우선 기호의 뜻부터 설명하겠습니다. [식 C.1]의 $\nabla f(\mathbf{x})$ 는 기울기(gradient)를 나타냅니다. 기울기는 $\mathbf{x}$ 의 각 원소에 대한 미분으로, 원소까지 표기하면 다음과 같습니다.

$$
\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}
$$

 \hfill [식 C.2]

그리고 $\nabla^2 f(\mathbf{x})$ 는 헤세 행렬(Hessian matrix)입니다. 헤세 행렬의 식은 다음과 같습니다.

$$
\nabla^2 f(\mathbf{x}) = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}
$$

 \hfill [식 C.3]

[식 C.3]과 같이 헤세 행렬은 $\mathbf{x}$ 의 두 원소에 대한 미분입니다. 두 원소의 조합이 이루어지기 때문에 행렬 형태로 정의됩니다.

> **NOTE**
> 기울기 $\nabla f(\mathbf{x})$ 는 $\frac{\partial f}{\partial \mathbf{x}}$ 라고 표기할 수도 있고, 헤세 행렬 $\nabla^2 f(\mathbf{x})$ 는 $\frac{\partial^2 f}{\partial \mathbf{x} \partial \mathbf{x}^T}$ 라고 표기할 수도 있습니다.

[식 C.1]에서는 기울기와 헤세 행렬을 사용하여 $\mathbf{x}$ 를 갱신했습니다([식 C.1]의 $[\nabla^2 f(\mathbf{x})]^{-1}$ 은 헤세 행렬 $\nabla^2 f(\mathbf{x})$ 의 역행렬을 뜻합니다). 이때 $\mathbf{x}$ 를 기울기 방향으로 갱신하고 그 진행 거리를 헤세 행렬의 역행렬을 사용하여 조정합니다. 헤세 행렬이라는 2차 미분 정보를 이용함으로써 더 공격적으로 진행할 수 있어서 목적지에 더 빠르게 도달할 수 있는 것입니다. 그러나 아쉽게도 머신러닝, 특히 신경망에서는 뉴턴 방법이 좀처럼 사용되지 않습니다.

### 뉴턴 방법의 문제와 대안

머신러닝 같은 문제에서는 뉴턴 방법에 큰 단점이 있습니다. 매개변수가 많아지면 뉴턴 방법의 헤세 행렬, 정확히는 헤세 행렬의 역행렬 계산에 자원이 너무 많이 소모되는 것입니다. 매개변수가 $n$ 개면 $n^2$ 만큼의 메모리를 사용하며, $n \times n$ 의 역행렬 계산에는 $n^3$ 만큼 사용합니다.

> **NOTE**
> 신경망에서는 매개변수가 100만 개를 넘는 일이 다반사입니다. 만약 100만 개의 매개변수를 뉴턴 방법으로 갱신한다면 $100\text{만} \times 100\text{만}$ 크기의 헤세 행렬이 필요한데, 이렇게 큰 메모리를 장착한 장비는 현실적이지 않습니다.

이처럼 뉴턴 방법은 현실적인 해결책이 아닌 경우가 많기 때문에 대안들이 제안되고 있습니다. 대표적인 예가 준 뉴턴 방법(Quasi-Newton Method, QNM)입니다. 준 뉴턴 방법은 뉴턴 방법 중 '헤세 행렬의 역행렬'을 근사해 사용하는 방법의 총칭입니다(준 뉴턴 방법이라는 구체적인 방법이 존재하는 것은 아닙니다).

준 뉴턴 방법은 지금까지 몇 가지가 제안되었습니다. 그중에서 유명한 것이 L-BFGS입니다. L-BFGS는 기울기만으로 헤세 행렬을 근사합니다. (정확한 값이 아닌) 근사값을 구하여 계산 비용과 메모리 공간을 절약하는 전략입니다. 파이토치에는 L-BFGS<sup>[20]</sup>가 구현되어 있으니 부담 없이 시도해볼 수 있습니다. 다만 지금까지 딥러닝 분야에서 주류는 기울기만을 사용한 최적화(SGD, Momentum, Adam 등)라서 L-BFGS 등의 준 뉴턴 방법을 사용한 예는 많지 않습니다.

### double backprop의 용도: 헤세 행렬과 벡터의 곱

마지막으로 double backprop에 대해 보충하겠습니다. double backprop의 사용 예로는 헤세 행렬과 벡터의 곱(Hessian-vector product) 계산이 있습니다. 앞에서 말한 것처럼 원소 수가 늘어나면 헤세 행렬을 계산하는 비용이 매우 커집니다. 그러나 헤세 행렬과 벡터의 곱의 '결과'만 필요하다면 double backprop을 사용하여 효율적으로 구할 수 있습니다.

예를 들어 $y = f(\mathbf{x})$ 와 임의의 $n$ 차원 벡터 $\mathbf{v}$ 에서 헤세 행렬은 $\nabla^2 f(\mathbf{x})$ 입니다. 이때 $\nabla^2 f(\mathbf{x})\mathbf{v}$ (헤세 행렬 $\nabla^2 f(\mathbf{x})$ 와 벡터 $\mathbf{v}$ 의 곱)를 구해봅시다. 그러기 위해 다음 식으로 변환합니다.

$$
\nabla^2 f(\mathbf{x})\mathbf{v} = \nabla (\mathbf{v}^T \nabla f(\mathbf{x}))
$$

 \hfill [식 C.4]

이 변환이 가능하다는 것은 좌변과 우변의 원소를 표기한 후 전개해보면 알 수 있습니다. 다음은 벡터의 원소 수가 2개인 경우로 한정하여 식을 전개한 예입니다.

$$
\nabla^2 f(\mathbf{x})\mathbf{v} = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix}
$$

$$
= \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} v_1 + \frac{\partial^2 f}{\partial x_1 \partial x_2} v_2 \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} v_1 + \frac{\partial^2 f}{\partial x_2^2} v_2 \end{bmatrix}
$$

$$
\nabla (\mathbf{v}^T \nabla f(\mathbf{x})) = \nabla \left( \begin{bmatrix} v_1 & v_2 \end{bmatrix} \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \end{bmatrix} \right)
$$

$$
= \nabla \left( \frac{\partial f}{\partial x_1} v_1 + \frac{\partial f}{\partial x_2} v_2 \right)
$$

$$
= \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} v_1 + \frac{\partial^2 f}{\partial x_1 \partial x_2} v_2 \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} v_1 + \frac{\partial^2 f}{\partial x_2^2} v_2 \end{bmatrix}
$$

여기에서는 원소 수가 2개인 벡터로 한정했지만, $n$ 개짜리 벡터로도 쉽게 확장할 수 있습니다. 이상으로 [식 C.4]가 성립함을 알 수 있습니다.

이제 다시 [식 C.4]로 눈을 돌려보죠. [식 C.4]의 우변이 의미하는 것은 벡터 $\mathbf{v}$ 와 기울기 $\nabla f(\mathbf{x})$ 의 곱, 즉 벡터의 내적을 먼저 구하고, 그 결과로부터 다시 한 번 기울기를 구한다는 뜻입니다. 따라서 헤세 행렬을 만들지 않아도 되므로 계산 효율이 좋아집니다.

그러면 DeZero를 사용하여 헤세 행렬과 벡터의 곱을 구해보겠습니다. 원소 수가 2개인 벡터를 사용한 계산 예입니다(행렬 곱을 구하기 위해 나중에 소개할 `F.matmul` 함수를 미리 이용했습니다).

```python
import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([1.0, 2.0]))
v = Variable(np.array([4.0, 5.0]))

def f(x):
    t = x ** 2
    y = F.sum(t)
    return y

y = f(x)
y.backward(create_graph=True)

gx = x.grad
x.cleargrad()
```

```python
z = F.matmul(v, gx) # ❶
z.backward() # ❷
print(x.grad)
```

**실행 결과**
```text
variable([ 8. 10.])
```

이 코드를 수식으로 표현하면 $\nabla (\mathbf{v}^T \nabla f(\mathbf{x}))$ 에 해당합니다. $\mathbf{v}^T \nabla f(\mathbf{x})$ 의 계산이 `z = F.matmul(v, gx)`에 대응하고(❶), `z.backward()`에서 $z$ 에 대해 다시 한 번 기울기를 구하는 것이죠(❷). 이로써 헤세 행렬과 벡터의 곱이 구해집니다. 참고로 앞의 출력 결과는 올바른 값입니다. 이상으로 이번 칼럼을 마칩니다.
