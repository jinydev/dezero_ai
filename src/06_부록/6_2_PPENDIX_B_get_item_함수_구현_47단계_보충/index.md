---
layout: docs
title: "APPENDIX B getitem 함수 구현(47단계 보충)"
---

# APPENDIX B get_item 함수 구현(47단계 보충)


47단계에서는 `get_item`이라는 DeZero 함수의 사용법만 소개했는데, 이제부터 그 내부를 설명하겠습니다. 다음은 `GetItem` 클래스와 `get_item` 함수의 코드입니다.

##### dezero/functions.py
```python
class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


def get_item(x, slices):
    return GetItem(slices)(x)
```

초기화 시 슬라이스 연산을 수행하는 인수 `slices`를 받아 인스턴스 변수에 저장해 두고, `forward(x)` 메서드에서는 단순히 이 변수를 이용해 원소를 추출하고 있습니다.

> [!CAUTION]
> DeZero에서 `forward(x)`의 `x`는 `ndarray` 인스턴스이지만, `backward(gy)`의 `gy`는 `Variable` 인스턴스입니다. 따라서 역전파를 구현할 때는 `Variable` 인스턴스도 다룰 수 있도록 구현해둔 DeZero 함수들을 사용하여 계산해야 합니다.

그런데 슬라이스 조작에 대응하는 역전파 계산은 DeZero 함수 중에는 없습니다. 그래서 별도로 `GetItemGrad`라는 새로운 DeZero 함수 클래스를 제공합니다. 즉, `GetItemGrad`의 순전파가 `GetItem`의 역전파에 대응하도록 구현합니다.

`GetItemGrad` 클래스의 코드는 어떻게 생겼을까요?

##### dezero/functions.py
```python
class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        gx = np.zeros(self.in_shape)
        np.add.at(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)
```

우선 초기화 메서드에서 슬라이스 연산 인수(`slices`)와 함께 입력 데이터의 모양(`in_shape`)을 받습니다. 그리고 주 계산(`forward`)에서는 입력용 기울기로서 원소가 모두 0인 다차원 배열 `gx`를 준비한 다음 `np.add.at(gx, self.slices, gy)`를 실행합니다. 그 결과 `gx`의 원소 중 `self.slices`로 지정한 위치에 `gy`가 더해집니다. `np.add.at` 함수의 사용법은 다음 예를 보면 명확해질 것입니다.

```python
>>> import numpy as np
>>> a = np.zeros((2, 3))
>>> a
array([[0., 0., 0.],
       [0., 0., 0.]])
>>> b = np.ones((3,))
>>> b
array([1., 1., 1.])
```

```python
>>> slices = 1
>>> np.add.at(a, slices, b)
>>> a
array([[0., 0., 0.],
       [1., 1., 1.]])
```

> [!CAUTION]
> 다차원 배열을 슬라이스하여 여러 원소가 한꺼번에 추출된 경우라면 역전파 시 대응하는 기울기를 더해줘야 합니다. 앞의 코드에서는 `np.add.at` 함수로 기울기를 더해줬습니다.

다음으로 `np.add.at` 함수에 대응하는 역전파를 구현해야 합니다. 그런데 재미있게도 지금 구현한 `get_item` 함수가 정확히 이 일을 해줍니다. 이것으로 `get_item` 함수를 완성했습니다.
