---
layout: docs
title: "46단계 Optimizer로 수행하는 매개변수 갱신"
---

# 46단계 Optimizer로 수행하는 매개변수 갱신

지금까지는 경사하강법으로 매개변수를 갱신했습니다만, 딥러닝 분야에서는 경사하강법 외에도 다양한 최적화 기법이 제안되고 있습니다. 이번 단계에서는 매개변수 갱신 작업(갱신 코드)을 모듈화하고 쉽게 다른 모듈로 대체할 수 있는 구조를 만들겠습니다.

---

## 46.1 Optimizer 클래스

이번 절에서는 매개변수 갱신을 위한 기반 클래스인 `Optimizer`를 구현합니다. `Optimizer`가 기초를 제공하고, 구체적인 최적화 기법은 `Optimizer` 클래스를 상속한 곳에서 구현하게 할 것입니다. 먼저 `Optimizer`를 만나보시죠.

**dezero/optimizers.py**
```python
class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        # None 이외의 매개변수를 리스트에 모아둠
        params = [p for p in self.target.params() if p.grad is not None]
```

```python
        # 전처리(옵션)
        for f in self.hooks:
            f(params)

        # 매개변수 갱신
        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)
```

`Optimizer` 클래스의 초기화 메서드에서는 `target`과 `hooks`라는 두 개의 인스턴스 변수를 초기화합니다. 그리고 `setup` 메서드는 매개변수를 갖는 클래스(`Model` 또는 `Layer`)를 인스턴스 변수인 `target`으로 설정합니다.

한편 `update` 메서드는 모든 매개변수를 갱신합니다. 하지만 인스턴스 변수 `grad`가 `None`인 매개변수는 갱신을 건너뜁니다. 그리고 구체적인 매개변수 갱신은 `update_one` 메서드에서 수행하는데, 바로 이 메서드를 `Optimizer`의 자식 클래스에서 재정의해야 합니다.

또한 `Optimizer` 클래스는 매개변수 갱신에 앞서 전체 매개변수를 전처리해주는 기능도 갖췄습니다. 원하는 전처리가 있다면 `add_hook` 메서드를 사용하여 전처리를 수행하는 함수를 추가합니다. 이 구조 덕분에 가중치 감소(Weight Decay)와 기울기 클리핑(Gradient Clipping)<sup>[26]</sup> 같은 기법을 이용할 수 있습니다(구현 예는 `example/mnist.py` 등을 참고하세요).

---

## 46.2 SGD 클래스 구현

그러면 경사하강법으로 매개변수를 갱신하는 클래스를 구현해보죠. 다음은 `SGD`라는 이름의 클래스 코드입니다.

**dezero/optimizers.py**
```python
class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr
```

```python
    def update_one(self, param):
        param.data -= self.lr * param.grad.data
```

`SGD` 클래스는 `Optimizer` 클래스를 상속합니다. `__init__` 메서드는 학습률을 받아 초기화하고 `update_one` 메서드에서 매개변수 갱신 코드를 구현합니다. 이것으로 매개변수 갱신을 `SGD` 클래스에 맡길 수 있습니다. `SGD` 클래스는 `dezero/optimizers.py`에 담겨 있으니, 외부 파일에서는 `from dezero.optimizers import SGD`로 임포트할 수 있습니다.

> **NOTE**
> `SGD`는 **확률적경사하강법(Stochastic Gradient Descent)**의 약자입니다. 여기에서 말하는 '확률적'은 대상 데이터 중에서 무작위로(확률적으로) 선별한 데이터에 대해 경사하강법을 수행한다는 뜻입니다. 딥러닝에서는 이처럼 원래의 데이터에서 무작위로 골라 경사하강법을 수행하는 것이 일반적입니다.

---

## 46.3 SGD 클래스를 사용한 문제 해결

`SGD` 클래스를 사용하여 이전 단계와 똑같은 문제를 풀어봅시다. 마찬가지로 이전 단계에서 달라진 부분은 음영으로 표시했습니다.

**steps/step46.py**
```python
import numpy as np
from dezero import Variable
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
max_iter = 10000
hidden_size = 10

model = MLP((hidden_size, 1))
optimizer = optimizers.SGD(lr)
optimizer.setup(model)
# 또는 다음처럼 한 줄로 합칠 수 있다.
# optimizer = optimizers.SGD(lr).setup(model)
```

```python
    for i in range(max_iter):
        y_pred = model(x)
        loss = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()

        optimizer.update()
        if i % 1000 == 0:
            print(loss)
```

이번에는 `MLP` 클래스를 사용하여 모델을 생성했습니다(이전 단계에서는 `TwoLayerNet`을 사용). 그리고 `SGD` 클래스로 매개변수를 갱신합니다. 구체적인 매개변수 갱신 코드는 `SGD` 클래스에 담겨 있기 때문에 단순히 `optimizer.update()`를 호출하는 것으로 매개변수 갱신을 완료할 수 있습니다.

> **NOTE**
> `Optimizer` 클래스의 `setup` 메서드는 반환값으로 자신(`self`)을 돌려줍니다. 따라서 `my_optimizer = SGD(...).setup(...)`처럼 한 줄로 적을 수도 있습니다.

---

## 46.4 SGD 이외의 최적화 기법

기울기를 이용한 최적화 기법은 다양합니다. 대표적인 기법으로는 Momentum, AdaGrad<sup>[27]</sup>, AdaDelta<sup>[28]</sup>, Adam<sup>[29]</sup> 등을 들 수 있죠. `Optimizer` 클래스를 도입한 첫 번째 목표는 이처럼 다양한 최적화 기법을 필요에 따라 손쉽게 전환하기 위해서였습니다. 그래서 기반 클래스인 `Optimizer`를 상속하여 다양한 최적화 기법을 구현해보려 합니다. 이번 절에서는 `SGD` 이외의 최적화 기법으로 Momentum을 구현하겠습니다.

> **NOTE**
> AdaGrad, AdaDelta, Adam 등의 최적화 기법은 `dezero/optimizers.py`에 구현되어 있습니다. 지면에서는 따로 설명하지 않으니, 관심 있는 분은 『밑바닥부터 시작하는 딥러닝』의 '6.1 매개변수 갱신' 절을 참고하세요.

`Momentum` 기법을 수식으로 표현하면 다음과 같습니다.

$$
\mathbf{v} \leftarrow \alpha \mathbf{v} - \eta \frac{\partial L}{\partial \mathbf{W}} \tag{식 46.1}
$$

$$
\mathbf{W} \leftarrow \mathbf{W} + \mathbf{v} \tag{식 46.2}
$$

여기서 $\mathbf{W}$ 는 갱신할 가중치 매개변수, $\frac{\partial L}{\partial \mathbf{W}}$ 은 기울기($\mathbf{W}$ 에 관한 손실 함수 $L$ 의 기울기), $\eta$ 는 학습률을 뜻합니다. 또한 $\mathbf{v}$ 는 물리에서 말하는 '속도'에 해당합니다. 그래서 [식 46.1]은 물체가 기울기 방향으로 힘을 받아 가속된다는 물리 법칙을 나타냅니다. 그리고 [식 46.2]에 의해 속도만큼 위치(매개변수)가 이동합니다.

> **NOTE**
> [식 46.1]의 $\alpha \mathbf{v}$ 항은 물체가 아무런 힘을 받지 않을 때 서서히 감속시키는 역할을 합니다($\alpha$ 의 값을 0.9 등으로 설정합니다).

그럼 `Momentum` 기법을 `MomentumSGD`라는 이름으로 구현하겠습니다. 코드는 다음과 같습니다.

**dezero/optimizers.py**
```python
import numpy as np

class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v
```

여기서 각 매개변수에는 '속도'에 해당하는 데이터가 있습니다. 이 데이터들을 딕셔너리 타입의 인스턴스 변수 `self.vs`에 유지합니다. 초기화 시에는 `vs`에 아무것도 담겨 있지 않지만, `update_one()`이 처음 호출될 때 매개변수와 같은 타입의 데이터를 생성합니다. 그다음 코드들은 [식 46.1]과 [식 46.2]를 코드로 옮겨 적은 것입니다.

> **NOTE**
> `dezero/optimizers.py`의 `MomentumSGD` 클래스 코드는 앞의 코드와 일부 다른 부분이 있습니다. GPU 대응 로직이 반영된 것으로, `np.ones_like` 메서드 부분이 데이터 타입에 따라 CuPy의 `cupy.ones_like` 메서드를 호출하도록 작성되어 있습니다. GPU 대응은 52단계에서 다룹니다.

이상이 `Momentum`의 구현입니다. 이제 앞서 구현한 학습 코드에서 손쉽게 `Momentum`으로 전환할 수 있습니다. `optimizer = SGD(lr)`을 `optimizer = MomentumSGD(lr)`로 바꿔주기만 하면 끝이죠. 다른 코드는 아무것도 손댈 게 없습니다. 이상으로 다양한 최적화 기법을 간편하게 전환할 수 있게 되었습니다!
