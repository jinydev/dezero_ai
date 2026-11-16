---
layout: docs
title: "18단계 메모리 절약 모드"
---

# 18단계 메모리 절약 모드

이전 단계에서는 파이썬의 메모리 관리 방식에 대해 알아봤습니다. 이번 단계에는 DeZero의 메모리 사용을 개선할 수 있는 구조 두 가지를 도입합니다. 
---
첫 번째는 역전파 시 사용하는 메모리양을 줄이는 방법으로, 불필요한 미분 결과를 보관하지 않고 즉시 삭제합니다. 두 번째는 '역전파가 필요 없는 경우용 모드'를 제공하는 것입니다. 이 모드에서는 불필요한 계산을 생략합니다.

---

## 18.1 필요 없는 미분값 삭제

첫 번째로 DeZero의 역전파를 개선하겠습니다. 현재의 DeZero에서는 모든 변수가 미분값을 변수에 저장해두고 있습니다. 

---
다음 예를 보시죠.

```python
x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
t = add(x0, x1)
y = add(x0, t)
y.backward()

print(y.grad, t.grad)
print(x0.grad, x1.grad)
```
---

**실행 결과**
```
1.0 1.0
2.0 1.0
```
---

여기에서 사용자가 제공한 변수는 `x0`와 `x1`이며, 다른 변수 `t`와 `y`는 계산 결과로 만들어집니다. 그리고 `y.backward()`를 실행하여 미분하면 모든 변수가 미분 결과를 메모리에 유지합니다. 

---

그러나 많은 경우, 특히 머신러닝에서는 역전파로 구하고 싶은 미분값은 말단 변수(`x0`, `x1`)뿐일 때가 대부분입니다. 앞의 예에서는 `y`와 `t` 같은 중간 변수의 미분값은 필요하지 않습니다. 그래서 중간 변수에 대해서는 미분값을 제거하는 모드를 추가하겠습니다. 

---

현재의 `Variable` 클래스의 `backward` 메서드에 다음 음영 부분의 코드를 추가하면 됩니다.

**steps/step18.py**
```python
class Variable:
    # ...
    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y는 약한 참조(weakref)
```

---

우선 메서드의 인수에 `retain_grad`를 추가합니다. 이 `retain_grad`가 `True`면 지금까지처럼 모든 변수가 미분 결과(기울기)를 유지합니다. 반면 `retain_grad`가 `False`면(기본값) 중간 변수의 미분값을 모두 `None`으로 재설정합니다. 

---

그 원리는 앞의 코드에서 보듯 `backward` 메서드의 마지막 `for` 문으로, 각 함수의 출력 변수의 미분값을 유지하지 않도록 `y().grad = None`으로 설정하는 것입니다. 이렇게 하면 말단 변수 외에는 미분값을 유지하지 않습니다.
---

> **NOTE**  
> 앞 코드의 마지막 `y().grad = None`에서 `y`에 접근할 때 `y()`라고 한 이유는 `y`가 약한 참조이기 때문입니다(약한 참조 구조는 이전 단계에서 도입했습니다). `y().grad = None` 코드가 실행되면 참조 카운트가 0이 되어 미분값 데이터가 메모리에서 삭제됩니다.

---

이제 앞에서 실행했던 코드를 다시 실행해보죠.

**steps/step18.py**
```python
x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
t = add(x0, x1)
y = add(x0, t)
y.backward()

print(y.grad, t.grad)
print(x0.grad, x1.grad)
```

---

**실행 결과**
```
None None
2.0 1.0
```

이와 같이 중간 변수인 `y`와 `t`의 미분값이 삭제되어 그만큼의 메모리를 다른 용도로 사용할 수 있게 됩니다. 이렇게 DeZero의 메모리 사용에 관한 첫 번째 개선이 완성되었습니다. 다음은 두 번째 개선 차례지만, 그에 앞서 잠시 현재의 `Function` 클래스를 복습해보겠습니다.

---

## 18.2 Function 클래스 복습

DeZero에서 미분을 하려면 순전파를 수행한 뒤 역전파해주면 됩니다. 그리고 역전파 시에는 순전파의 계산 결과가 필요하기 때문에 순전파 때 결과값을 기억해둡니다. 

---
결과값을 보관하는 로직은 바로 `Function` 클래스의 다음 음영 부분입니다.

**steps/step17.py**
```python
class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
```

---

이와 같이 함수는 입력을 `inputs`라는 '인스턴스 변수'로 참조합니다. 그 결과 `inputs`가 참조하는 변수의 참조 카운트가 1만큼 증가하고, `__call__` 메서드에서 벗어난 뒤에도 메모리에 생존합니다. 만약 인스턴스 변수인 `inputs`로 참조하지 않았다면 참조 카운트가 0이 되어 메모리에서 삭제됐을 겁니다.

---

인스턴스 변수 `inputs`는 역전파 계산 시 사용됩니다. 따라서 역전파하는 경우라면 참조할 변수들을 `inputs`에 미리 보관해둬야 합니다. 하지만 때로는 미분값이 필요 없는 경우도 있습니다. 이런 경우라면 중간 계산 결과를 저장할 필요가 없고, 계산의 '연결' 또한 만들 이유가 없습니다.

---

> **CAUTION**  
> 신경망에는 학습<sup>training</sup>(혹은 훈련)과 추론<sup>inference</sup>이라는 두 가지 단계가 있습니다. 학습 시에는 미분값을 구해야 하지만, 추론 시에는 단순히 순전파만 하기 때문에 중간 계산 결과를 곧바로 버리면 메모리 사용량을 크게 줄일 수 있습니다.

---

## 18.3 Config 클래스를 활용한 모드 전환

이제부터 순전파만 할 경우를 위한 개선을 DeZero에 추가하겠습니다. 우선 두 가지 모드, 즉 '역전파 활성 모드'와 '역전파 비활성 모드'를 전환하는 구조가 필요합니다. 간단히 다음 `Config` 클래스를 이용할 것입니다.

**steps/step18.py**
```python
class Config:
    enable_backprop = True
```

보다시피 아주 간단한 클래스입니다. 이 클래스의 속성은 (현재) 불리언 타입인 `enable_backprop`만 존재합니다. `enable_backprop`은 역전파가 가능한지 여부를 뜻하고, 이 값이 `True`면 '역전파 활성 모드'입니다.

> **CAUTION**  
> 설정 데이터는 단 한 군데에만 존재하는 게 좋습니다. 그래서 `Config` 클래스는 인스턴스화하지 않고 '클래스' 상태로 이용합니다. 인스턴스는 여러 개 생성할 수 있지만 클래스는 항상 하나만 존재하기 때문이죠. 따라서 앞 코드에서 `Config` 클래스가 '클래스 속성'을 갖도록 설정했습니다.

`Config` 클래스를 정의했으니 `Function`에서 참조하게 하여 모드를 전환할 수 있게 하겠습니다. 코드로는 다음과 같습니다.

**steps/step18.py**
```python
class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])  # ❶ 세대 설정
            for output in outputs:
                output.set_creator(self)  # ❷ 연결 설정
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]
```

이와 같이 `Config.enable_backprop`이 `True`일 때만 역전파 코드가 실행됩니다. ❶에서 정하는 '세대'는 역전파 시 노드를 따라가는 순서를 정하는 데 사용됩니다. 따라서 '역전파 비활성 모드'에서는 필요하지 않습니다. 또한 ❷의 `output.set_creator(self)`는 계산들의 '연결'을 만드는데, 마찬가지로 '역전파 비활성 모드'에서는 필요 없습니다.

---

## 18.4 모드 전환

이상으로 역전파 활성/비활성을 구분 짓는 구조가 만들어졌습니다. 이 구조를 활용하면 다음과 같이 모드를 전환할 수 있습니다.

```python
Config.enable_backprop = True
x = Variable(np.ones((100, 100, 100)))
y = square(square(square(x)))
y.backward()

Config.enable_backprop = False
x = Variable(np.ones((100, 100, 100)))
y = square(square(square(x)))
```

일부러 큰 다차원 배열을 준비해보았습니다. 형상이 $(100, 100, 100)$ 인 텐서입니다. 이 텐서에 `square` 함수를 세 번 적용합니다(그러면 원소별 제곱이 이루어집니다). 이때 `Config.enable_backprop`이 `True`면 중간 계산 결과가 (적어도 역전파가 완료되기 전까지는) 계속 유지되어 그만큼 메모리를 차지합니다. 한편 `Config.enable_backprop`이 `False`면 중간 계산 결과는 사용 후 곧바로 삭제됩니다(정확하게는 다른 객체에서의 참조가 없어지는 시점에 메모리에서 삭제됩니다).

이상으로 역전파 모드를 전환하는 구조가 완성되었습니다. 이어서 모드 전환을 더 쉽게 해주는 구조를 만들어볼까 합니다.

---

## 18.5 with 문을 활용한 모드 전환

파이썬에는 `with`라고 하는, 후처리를 자동으로 수행하고자 할 때 사용할 수 있는 구문이 있습니다. 대표적인 예는 파일의 `open`과 `close`입니다. 예를 들어 `with` 문 없이 파일에 무언가를 쓰려면 다음과처럼 작성해야 합니다.

```python
f = open('sample.txt', 'w')
f.write('hello world!')
f.close()
```

보다시피 `open()`으로 파일을 열고, 무언가를 쓰고, `close()`로 파일을 닫습니다. 이때 매번 `close()`하기란 귀찮기도 하거니와 깜빡하고 잊어버릴 때도 있습니다. `with` 문은 이런 실수를 막아줍니다.

```python
with open('sample.txt', 'w') as f:
    f.write('hello world!')
```

이 코드에서는 `with` 블록에 들어갈 때 파일이 열립니다. `with` 블록 안에서 파일은 계속 열린 상태고 블록을 빠져나올 때 (사용자에게 보이지 않는 곳에서) 자동으로 닫힙니다. 이와 같이 `with` 문을 사용하는 것으로 '`with` 블록에 들어갈 때의 처리(전처리)'와 '`with` 블록을 빠져나올 때의 처리(후처리)'를 자동으로 할 수 있습니다.

이러한 `with` 문의 원리를 이용하여 '역전파 비활성 모드'로 전환하려 합니다. 구체적으로는 다음과 같은 식으로 사용하려 합니다(`using_config` 메서드의 구현은 조금 뒤에 설명합니다).

**steps/step18.py**
```python
with using_config('enable_backprop', False):
    x = Variable(np.array(2.0))
    y = square(x)
```

이와 같이 `with using_config('enable_backprop', False):` 안에서만 '역전파 비활성 모드'가 됩니다. 그리고 `with` 블록을 벗어나면 일반 모드, 즉 '역전파 활성 모드'로 돌아갑니다.

> **NOTE**  
> '역전파 비활성 모드'로 일시적으로 전환하는 방법은 실전에서 자주 사용됩니다. 예컨대 신경망 학습에서는 모델 평가를 (학습 도중에) 하기 위해 기울기가 필요 없는 모드를 사용하는 일이 자주 발생합니다.

그럼 `with` 문을 사용한 모드 전환을 구현해볼까요? `contextlib` 모듈을 사용하면 가장 쉽게 구현할 수 있습니다. 우선 `contextlib` 모듈 사용법을 설명하겠습니다.

```python
import contextlib

@contextlib.contextmanager
def config_test():
    print('start')  # 전처리
    try:
        yield
    finally:
        print('done')  # 후처리

with config_test():
    print('process...')
```

**실행 결과**
```
start
process...
done
```

앞의 코드처럼 `@contextlib.contextmanager` 데코레이터를 달면 문맥<sup>context</sup>을 판단하는 함수가 만들어집니다. 그리고 이 함수 안에서 `yield` 전에는 전처리 로직을, `yield` 다음에는 후처리 로직을 작성합니다. 그러면 `with config_test():` 형태의 구문을 사용할 수 있습니다. 이 구문을 사용하면 `with` 블록 안으로 들어갈 때 전처리가 실행되고 블록 범위를 빠져나올 때 후처리가 실행됩니다.

> **NOTE**  
> `with` 블록 안에서 예외가 발생할 수 있고, 발생한 예외는 `yield`를 실행하는 코드로도 전달됩니다. 따라서 `yield`는 `try/finally`로 감싸야 합니다.

이상을 바탕으로 `using_config` 함수를 다음과 같이 구현할 수 있습니다.

**steps/step18.py**
```python
import contextlib

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)
```

`using_config(name, value)`의 인수 중 `name`은 타입이 `str`이며, 사용할 `Config` 속성의 이름(클래스 속성 이름)을 가리킵니다. 그리고 `name`을 `getattr` 함수에 넘겨 `Config` 클래스에서 꺼내옵니다. 그런 다음 `setattr` 함수를 사용하여 새로운 값을 설정합니다.

이제 `with` 블록에 들어갈 때 `name`으로 지정한 `Config` 클래스 속성이 `value`로 설정됩니다. 그리고 `with` 블록을 빠져나오면서 원래 값(`old_value`)으로 복원됩니다. 그럼 `using_config` 함수를 실제로 사용해보죠.

**steps/step18.py**
```python
with using_config('enable_backprop', False):
    x = Variable(np.array(2.0))
    y = square(x)
```

이와 같이 역전파가 필요 없는 경우에는 `with` 블록에서 순전파 코드만 실행합니다. 이제 불필요한 계산을 생략하고 메모리를 절약할 수 있습니다. 그러나 `with using_config('enable_backprop', False):`라는 긴 코드를 매번 적어주기는 귀찮은 일이니 다음과 같이 `no_grad`라는 편의 함수를 준비했습니다.

**steps/step18.py**
```python
def no_grad():
    return using_config('enable_backprop', False)

with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)
```

`no_grad` 함수는 단순히 `using_config('enable_backprop', False)`를 호출하는 코드를 `return`으로 돌려줄 뿐입니다. 이제 기울기가 필요 없을 때는 `no_grad` 함수를 호출하면 됩니다. 이상으로 이번 단계를 마칩니다. 앞으로는 기울기 계산이 필요 없을 때, 즉 단순히 순전파 계산만 필요할 때는 방금 구현한 '모드 전환'을 사용하겠습니다.
