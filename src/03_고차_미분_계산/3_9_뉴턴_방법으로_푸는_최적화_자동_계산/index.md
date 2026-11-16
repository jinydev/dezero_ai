---
layout: docs
title: "33단계 뉴턴 방법으로 푸는 최적화(자동 계산)"
---

# 33단계 뉴턴 방법으로 푸는 최적화(자동 계산)

지금까지는 2차 미분을 수동으로 계산했습니다. 이번 단계에서는 새로운 DeZero를 사용하여 2차 미분도 자동으로 계산할 생각입니다. 우선 간단한 수식의 2차 미분을 계산합니다. 그리고 미분값이 제대로 구해졌음을 확인한 다음, 이어서 뉴턴 방법을 사용해 최적화하겠습니다.

---

## 33.1 2차 미분 계산하기

29단계에서 다룬 $y = x^4 - 2x^2$ 이라는 수식의 2차 미분을 계산해봅시다. DeZero를 사용하면 다음과 같이 구현할 수 있습니다(이 구현에는 사실 한 가지 문제가 있습니다).

```python
import numpy as np
from dezero import Variable

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

x = Variable(np.array(2.0))
y = f(x)
y.backward(create_graph=True) # 1
print(x.grad)
```

```python
# 두 번째 역전파 진행
gx = x.grad # 2
gx.backward() # 3
print(x.grad)
```

**실행 결과**
```text
variable(24.0)
variable(68.0)
```

우선 1의 `y.backward(create_graph=True)`에 의해 첫 번째 역전파가 진행됩니다. 이때 인수 `create_graph`를 `True`로 지정하여 역전파 계산에 대해서도 계산 그래프를 만들게 했습니다. 이제 역전파 계산 그래프에 다시 한 번 역전파할 차례입니다. $x$ 의 2차 미분을 계산해야 하므로 먼저 2의 `gx = x.grad` 코드로 $y$ 의 $x$ 에 대한 미분값을 꺼냅니다. 그리고 3에서는 방금 꺼낸 미분값인 `gx`에서 한 번 더 역전파합니다. 이렇게 하면 `gx`의 $x$ 에 대한 미분이 구해집니다. 이 두 번째 미분이 바로 2차 미분에 해당합니다.

코드를 실행하면 1차 미분이 24.0이고, 2차 미분이 68.0으로 나옵니다. 수식으로 확인하면 $y' = 4x^3 - 4x$ 이므로 $x = 2$ 일 때의 1차 미분은 24입니다. 구현 결과와 일치합니다. 그리고 2차 미분은 수식으로 $y'' = 12x^2 - 4$ 이므로 $x = 2$ 일 때는 44입니다. 이런! 우리의 결과와 다르군요.

앞의 코드가 내놓은 68이라는 잘못된 결과는 1차 미분 결과(24)에 2차 미분 결과(44)가 '더해진' 값입니다. 즉, `Variable`에 미분값이 남아 있는 상태에서 새로운 역전파를 수행했기 때문에 새로운 미분값이 '더해진' 것입니다. 이 문제를 해결하려면 새로운 계산을 하기 전에 `Variable`의 미분값을 '재설정'해야 합니다.

> **NOTE**
> 복습해봅시다. DeZero에서 역전파 시 `x.backward(retain_grad=False)`처럼 `retain_grad`라는 인수를 받습니다. `retain_grad`는 18단계에서 추가한 기능으로, 값이 `False`(기본값)면 계산 과정에서 중간 변수의 미분값(기울기)은 자동으로 재설정됩니다. 따라서 말단 변수(사용자가 제공한 변수)만 미분값을 갖게 됩니다. 예컨대 `x.backward()`를 호출하면 $x$ 만 미분값을 유지합니다.

이를 반영하여 이전 문제를 다시 풀어봅시다.

```python
x = Variable(np.array(2.0))
y = f(x)
y.backward(create_graph=True)
print(x.grad)

gx = x.grad
x.cleargrad() # 미분값 재설정
gx.backward()
print(x.grad)
```

**실행 결과**
```text
variable(24.0)
variable(44.0)
```

보다시피 `gx.backward()`를 부르기 전에 `x.cleargrad()`를 추가했습니다. 이 한 줄이 $x$ 의 미분값을 재설정하여 역전파가 올바르게 이루어지게 합니다. 실제로 앞의 코드를 실행하면 2차 미분의 결과는 44.0이 나오며, 수식으로 확인한 결과와 일치합니다.

---

## 33.2 뉴턴 방법을 활용한 최적화

이제 뉴턴 방법을 활용해 최적화를 해볼 차례입니다. 우선 뉴턴 방법을 활용한 최적화의 수식을 다시 확인해보죠.

$$
x \leftarrow x - \frac{f'(x)}{f''(x)}
$$

  [식 33.1]

[식 33.1]과 같이 함수 $f(x)$ 의 1차 미분과 2차 미분을 사용하여 $x$ 를 갱신하고 있습니다. 이 계산을 DeZero를 사용하여 자동으로 수행되도록 해봅시다.

**steps/step33.py**
```python
import numpy as np
from dezero import Variable

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y
```

```python
x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward(create_graph=True)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data
```

이 코드는 29단계에서 구현한 코드에 기초하고 있습니다. 지난번에는 2차 미분을 '손으로' 계산해 하드코딩했지만, 이번에는 `backward` 메서드를 두 번 실행하여 자동으로 계산하게 수정했습니다. 이 코드를 실행하면 $x$ 값의 갱신 과정이 다음과 같이 출력됩니다.

```text
0 variable(2.0)
1 variable(1.4545454545454546)
2 variable(1.1510467893775467)
3 variable(1.0253259289766978)
4 variable(1.0009084519430513)
5 variable(1.0000012353089454)
6 variable(1.000000000002289)
7 variable(1.0)
8 variable(1.0)
9 variable(1.0)
```

결과에서 알 수 있듯이 불과 7회 만에 최솟값 1에 도달합니다. 29단계 때와 똑같은 결과입니다. 즉, 뉴턴 방법을 활용한 최적화를 자동으로 진행하게끔 한 구현이 올바르게 작동함을 알 수 있습니다!
