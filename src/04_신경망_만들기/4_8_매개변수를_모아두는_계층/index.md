---
layout: docs
title: "44단계 매개변수를 모아두는 계층"
---

# 44단계 매개변수를 모아두는 계층

이전 단계에서는 DeZero를 사용하여 신경망을 구현했습니다. 단순하지만 진짜 신경망이었습니다. 이제 DeZero를 '신경망 프레임워크'라고 말할 수 있을 것입니다. 하지만 '사용 편의성' 면에서는 여전히 몇 가지 문제가 있습니다. 앞으로 우리는 DeZero에 신경망을 위한 기능을 추가할 것입니다. 이를 통해 신경망, 그리고 딥러닝을 더 간단하고 직관적으로 구현할 수 있게 됩니다.

이번 단계에서는 매개변수를 다룹니다. 이전 단계에서는 매개변수의 기울기를 재설정할 때(혹은 매개변수를 갱신할 때) 다소 단조로운 코드를 작성해야 했습니다. 앞으로는 더욱 복잡한 신경망을 구현할 텐데, 그렇게 되면 매개변수를 다루는 일도 그만큼 복잡해집니다.

> **NOTE**
> 매개변수는 경사하강법 등의 최적화 기법에 의해 갱신되는 변수입니다. 이전 단계의 예에서는 선형 변환에 사용되는 '가중치'와 '편향'이 매개변수에 해당합니다.

이번 단계에서는 매개변수를 담는 구조를 만듭니다. 이를 위해 `Parameter`와 `Layer`라는 클래스를 구현합니다. 이 두 클래스를 사용하면 매개변수 관리를 자동화할 수 있습니다.

---

## 44.1 Parameter 클래스 구현

우선 `Parameter` 클래스입니다. `Parameter` 클래스는 `Variable` 클래스와 똑같은 기능을 갖게

합니다. 그래서 다음과 같이 구현합니다.

**dezero/core.py**
```python
class Parameter(Variable):
    pass
```

이것이 `Parameter` 클래스입니다. 보다시피 `Variable` 클래스를 상속한 게 다라서 기능도 `Variable` 클래스와 동일합니다.

> **NOTE**
> `Parameter` 클래스는 `dezero/core.py`에 추가합니다. 그리고 `dezero/__init__.py`에 `from dezero.core import Parameter`라는 한 줄을 추가합니다. 그러면 DeZero를 사용하는 사람은 `from dezero import Parameter`로 임포트할 수 있습니다.

`Parameter` 인스턴스와 `Variable` 인스턴스는 기능은 같지만 구별할 수는 있습니다. 구체적인 예를 볼까요?

```python
import numpy as np
from dezero import Variable, Parameter

x = Variable(np.array(1.0))
p = Parameter(np.array(2.0))
y = x * p

print(isinstance(p, Parameter))
print(isinstance(x, Parameter))
print(isinstance(y, Parameter))
```

**실행 결과**
```text
True
False
False
```

이와 같이 `Parameter` 인스턴스와 `Variable` 인스턴스를 조합하여 계산할 수 있습니다. 그리고 `isinstance` 함수로 구분할 수 있죠. 이 점을 이용하여 `Parameter` 인스턴스만을 담는 구조를 만들 수 있습니다.

---

## 44.2 Layer 클래스 구현

다음은 `Layer` 클래스입니다. `Layer`는 DeZero의 `Function` 클래스와 마찬가지로 변수를 변환하는 클래스입니다. 그러나 매개변수를 유지한다는 점이 다릅니다. `Layer` 클래스는 매개변수를 유지하고 매개변수를 사용하여 변환을 하는 클래스입니다.

> **NOTE**
> `Layer` 클래스를 기반 클래스로 두고 구체적인 변환은 자식 클래스에서 구현합니다. 예를 들어 선형 변환은 `Layer` 클래스를 상속한 `Linear` 클래스에서 구현합니다.

`Layer` 클래스의 구현을 보겠습니다. 우선 초기화와 `__setattr__` 특수 메서드를 살펴보시죠.

**dezero/layers.py**
```python
from dezero.core import Parameter

class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._params.add(name)
        super().__setattr__(name, value)
```

`Layer` 클래스에는 `_params`라는 인스턴스 변수가 있습니다. `_params`에는 `Layer` 인스턴스에 속한 매개변수를 보관합니다.

> **CAUTION**
> 인스턴스 변수 `_params`의 타입은 '집합(set)'입니다. 집합은 리스트와 달리 원소들에 순서가 없고, ID가 같은 객체는 중복 저장할 수 없습니다.

`__setattr__`은 인스턴스 변수를 설정할 때 호출되는 특수 메서드입니다. `__setattr__(self, name, value)`는 이름이 `name`인 인스턴스 변수에 값으로 `value`를 전달해줍니다. 이 메서드를 재정의(override)하면 인스턴스 변수를 설정할 때 여러분만의 로직을 추가할 수 있습니다.

여기에서는 `value`가 `Parameter` 인스턴스라면 `self._params`에 `name`을 추가합니다.* 이렇게 하여 `Layer` 클래스가 갖는 매개변수를 인스턴스 변수 `_params`에 모아둘 수 있습니다. 다음은 예시 코드입니다.

```python
layer = Layer()

layer.p1 = Parameter(np.array(1))
layer.p2 = Parameter(np.array(2))
layer.p3 = Variable(np.array(3))
layer.p4 = 'test'

print(layer._params)
print('------------')

for name in layer._params:
    print(name, layer.__dict__[name])
```

**실행 결과**
```text
{'p2', 'p1'}
------------
p2 variable(2)
p1 variable(1)
```

이와 같이 `layer` 인스턴스 변수를 설정하면 `Parameter` 인스턴스를 보유하고 있는 인스턴스 변수 이름만 `layer._params`에 추가됩니다. 또한 인스턴스 변수 `__dict__`에는 모든 인스턴스 변수가 딕셔너리(dictionary) 타입으로 저장되기 때문에 `Parameter` 인스턴스만 꺼낼 수 있습니다.

그런 다음 `Layer` 클래스에 다음 4개의 메서드를 추가합니다.

**dezero/layers.py**
```python
import weakref

class Layer:
    # ...

    def __call__(self, *inputs):
```

---
\* `self._params`에는 `value`가 아니라 `name`을 추가합니다. 매개변수를 외부 파일로 저장할 때 `name`을 유지하는 쪽이 편리하기 때문입니다. 해당 작업은 53단계에서 수행합니다.

```python
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            yield self.__dict__[name]

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()
```

`__call__` 메서드는 입력받은 인수를 건네 `forward` 메서드를 호출합니다. `forward` 메서드는 자식 클래스에서 구현할 것입니다. `__call__` 메서드는 출력이 하나뿐이라면 튜플이 아니라 그 출력을 직접 반환합니다(`Function` 클래스와 똑같은 관례를 따랐습니다). 또한 나중을 생각하여 입력과 출력 변수를 약한 참조로 유지하고 있습니다.

`params` 메서드는 `Layer` 인스턴스에 담겨 있는 `Parameter` 인스턴스들을 꺼내주고, `cleargrads` 메서드는 모든 매개변수의 기울기를 재설정합니다. 참고로 `cleargrads`의 이름 끝에 's'를 붙여 '복수'형으로 만들었는데, 이는 `Layer`가 가진 '모든' 매개변수에 대해 (단수형인) `cleargrad`를 호출한다는 사실을 명시한 것입니다.

> **NOTE**
> `params` 메서드는 `yield`를 사용하여 값을 반환합니다. `yield`는 `return`처럼 사용할 수 있습니다. 다만 `return`은 처리를 종료하고 값을 반환하는 반면 `yield`는 처리를 '일시 중지(suspend)'하고 값을 반환합니다. 따라서 `yield`를 사용하면 작업을 '재개(resume)'할 수 있습니다. 앞의 예로 말하자면, `params` 메서드를 호출할 때마다 일시 중지됐던 처리가 재개됩니다. 이런 식으로 (`yield`를 `for` 문과 함께 사용하여) 매개변수를 순차적으로 꺼낼 수 있습니다.

이상으로 `Layer` 클래스의 구현을 마칩니다. 이제부터 `Layer` 클래스를 상속하여 선형 변환 등의 구체적인 처리를 구현할 것입니다.

---

## 44.3 Linear 클래스 구현

이어서 선형 변환을 하는 `Linear` 클래스를 구현하겠습니다(함수로서의 `Linear` 클래스가 아니라 계층으로서의 `Linear` 클래스를 구현합니다). 먼저 간단한 `Linear` 클래스를 보여드린 후 개선하는 식으로 진행하겠습니다. 다음 코드를 살펴보시죠.

```python
import numpy as np
import dezero.functions as F
from dezero.core import Parameter

class Linear(Layer):
    def __init__(self, in_size, out_size, nobias=False, dtype=np.float32):
        super().__init__()

        I, O = in_size, out_size
        W_data = np.random.randn(I, O).astype(dtype) * np.sqrt(1 / I)
        self.W = Parameter(W_data, name='W')
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(O, dtype=dtype), name='b')

    def forward(self, x):
        y = F.linear(x, self.W, self.b)
        return y
```

`Linear` 클래스는 `Layer` 클래스를 상속하여 구현합니다. `__init__(self, in_size, out_size, nobias)`의 인수는 차례로 '입력 크기', '출력 크기', '편향 사용 여부 플래그'입니다. `nobias`가 `True`면 편향을 생략합니다.

이때 가중치와 편향은 `self.W = Parameter(...)`와 `self.b = Parameter(...)` 형태로, `Parameter` 인스턴스를 인스턴스 변수에 설정합니다. 이렇게 함으로써 두 `Parameter` 인스턴스 변수의 이름이 `self._params`에 추가됩니다.

> **NOTE**
> `Linear` 클래스의 가중치 초깃값은 무작위로 설정해야 합니다. 이전 단계에서는 무작위 초깃값의 스케일을 $0.01$ 로 설정했는데($0.01 * \text{np.random.randn}(\dots)$), 이번에는 참고문헌 [25]에서 제안한 방법에 따라 `np.sqrt(1 / in_size)`로 설정했습니다. 또한 신경망의 계산은 32비트 부동소수점을 써도 문제가 없다고 하니 매개변수의 데이터는 32비트 부동소수점을 기본값으로 설정했습니다.

이어서 `forward` 메서드로 선형 변환을 구현했습니다. 이 메서드는 DeZero의 `linear` 함수를 호출할 뿐입니다. 이상이 `Linear` 클래스의 구현입니다.

그런데 앞서 언급했듯이 `Linear` 클래스를 구현하는 더 나은 방법이 있습니다. 바로 가중치 `W`를 생성하는 시점을 늦추는 방식입니다. 구체적으로는 가중치를 (초기화 메서드가 아닌) `forward` 메서드에서 생성함으로써 `Linear` 클래스의 입력 크기(`in_size`)를 자동으로 결정할 수 있습니다(사용자가 지정하지 않아도 됩니다). 개선된 버전의 `Linear` 클래스를 만나보시죠.

**dezero/layers.py**
```python
import numpy as np
import dezero.functions as F
from dezero.core import Parameter

class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_size is not None:  # in_size가 지정되어 있지 않다면 나중으로 연기
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        # 데이터를 흘려보내는 시점에 가중치 초기화
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()

        y = F.linear(x, self.W, self.b)
        return y
```

이것이 개선된 `Linear` 클래스입니다. 주목할 점은 `__init__` 메서드에서 `in_size`를 지정하지 않아도 된다는 것입니다. 인수 `in_size`는 기본적으로 `None`으로 지정되어 있으며, `None`인 경우 `self.W.data` 초기화를 '연기'합니다. `forward(self, x)` 메서드에서 입력 `x`의 크기에 맞게 가중치 데이터를 생성하는 것이죠. 이제 `layer = Linear(100)`처럼 출력 크기만 지정해도 되므로 사용성이 좋아졌습니다. 이상이 `Linear` 클래스의 구현입니다.

---

## 44.4 Layer를 이용한 신경망 구현

다음은 `Linear` 클래스를 이용하여 신경망을 구현해볼 차례입니다. 이전 단계에서 푼 문제, 즉 `sin` 함수의 데이터셋에 대한 회귀 문제를 다시 풀어봅시다. 이전 단계에서 달라진 부분은 음영으로 표시했습니다.

**steps/step44.py**
```python
import numpy as np
from dezero import Variable
import dezero.functions as F
import dezero.layers as L  # L로 임포트

# 데이터셋
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

l1 = L.Linear(10)  # 출력 크기 지정
l2 = L.Linear(1)

def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)
```

```python
    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)
```

주목할 점은 매개변수 관리를 `Linear` 인스턴스가 맡고 있다는 것입니다. 그 덕분에 매개변수 기울기 재설정과 매개변수 갱신 작업이 전보다 깔끔해졌습니다.

그러나 `Linear` 클래스를 개별적으로 다루는 부분이 눈에 밟힙니다. 앞으로 더욱 '깊은' 신경망을 다룰 것을 생각하면 분명 걸림돌이 될 거 같습니다. 그래서 다음 단계에서는 여러 `Layer`를 하나의 클래스로 묶어서 관리하도록 개선하겠습니다.
