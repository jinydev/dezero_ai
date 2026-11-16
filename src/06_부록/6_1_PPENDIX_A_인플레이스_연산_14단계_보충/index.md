---
layout: docs
title: "APPENDIX A 인플레이스 연산(14단계 보충)"
---

# APPENDIX A 인플레이스 연산(14단계 보충)


이번 부록은 14단계의 설명을 보충하기 위해 준비했습니다. 14단계에서 '미분값을 더할 때 `+=`를 사용하지 않는다'라고 했는데, 그 이유를 함께 알아보겠습니다.

## A.1 문제 확인

우선 이번 문제를 정확하게 정리해보죠. 14단계에서 같은 변수를 반복해서 사용할 수 있도록 `Variable` 클래스의 `backward` 메서드를 다음처럼 수정했습니다.

```python
class Variable:
    ...

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(gys)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
```

```python
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

            if x.creator is not None:
                funcs.append(x.creator)
```

음영으로 표시한 부분이 수정된 코드입니다. 간단히 말하면 1차 미분의 결과(기울기)가 전파됐을 때는 `x.grad = gx`로 '대입'하고, 그 이후로는 `x.grad = x.grad + gx`로 '누적'합니다. 참고로 `gx`는 `ndarray` 인스턴스입니다. 14단계에서는 이 누적 연산을 `x.grad += gx`라고 쓰면 문제가 되는 경우가 있음을 지적했습니다. 이제부터 그 이유를 설명합니다.

## A.2 복사와 덮어 쓰기

우선 사전 지식으로 `ndarray` 인스턴스의 '복사<sup>copy</sup>'와 '덮어 쓰기<sup>overwrite</sup>'의 차이를 이해해야 합니다. 다음 코드를 살펴봅시다.

```python
>>> import numpy as np
>>> x = np.array(1)
>>> id(x)

>>> x += x # 덮어 쓰기
>>> id(x)

>>> x = x + x # 복사(새로 생성)
>>> id(x)
```

여기에서 `id(x)`의 결과를 보면(x는 `ndarray` 인스턴스), x가 메모리에서 덮어 써지는지 아니면 새로 생성되는지 알 수 있습니다. 보다시피 누적 대입 연산자인 `+=`를 사용하면 x의 객체 ID가 변하지 않습니다. 즉, 메모리 위치가 동일하다는 뜻으로, 값만 덮어 쓴 것입니다. 이처럼 복사하지 않고 메모리의 값을 직접 덮어 쓰는 연산을 인플레이스 연산<sup>in-place operation</sup>이라고 합니다.

한편 `x = x + x`를 실행하니 객체 ID가 달라졌습니다. 따라서 다른 위치에 새로운 `ndarray` 인스턴스가 생성되었음을(복사되었음을) 알 수 있습니다. 메모리 효율 측면에서는 인플레이스 연산을 사용하는 게 바람직하다고 할 수 있습니다(물론 인플레이스 연산을 사용해도 문제가 없는 상황에서만).

> [!CAUTION]
> `id` 함수는 파이썬 객체의 ID를 반환합니다. 구체적인 반환값은 실행 시점과 환경에 따라 달라집니다.

## A.3 DeZero의 역전파에서는

DeZero의 역전파에서는 미분값이 `ndarray` 인스턴스 형태로 전파됩니다. 따라서 만약 두 번째 이후의 역전파 시 미분값을 `x.grad += gx`라는 '인플레이스 연산'으로 덮어 쓰면 어떻게 될까요? 실제로 `steps/step14.py` 코드를 인플레이스 연산으로 바꾼 후 다음 코드를 실행해봅시다.

```python
x = Variable(np.array(3))
y = add(x, x)
y.backward()

print('y.grad: {}({})'.format(y.grad, id(y.grad)))
print('x.grad: {}({})'.format(x.grad, id(x.grad)))
```

**실행 결과**
```
y.grad: 2 (4427494384)
x.grad: 2 (4427494384)
```

x와 y의 미분 결과는 모두 2로 같은데, 오른쪽을 살펴보니 `ndarray`의 ID까지 똑같습니다. 같은 `ndarray`를 참조한다는 소리입니다. 여기에서 문제는 y의 미분값으로, 본래는 1이어야 합니다.

이와 같은 사태가 발생한 원인은 인플레이스 연산이 값을 덮어 썼기 때문입니다. `y.grad`와 `x.grad`가 같은 값을 참조하게 되어 결과적으로 `y.grad`에 잘못된 결과가 저장된 것입니다. 그

래서 코드를 `x.grad = x.grad + gx`로 고쳐서 (덮어 쓰지 않고) 복사하도록 했던 것입니다. 제대로 수정한 후 앞의 코드를 다시 실행하면 다음 결과를 얻을 수 있습니다.

**실행 결과**
```
y.grad: 1 (4755624944)
x.grad: 2 (4755710960)
```

이번에는 y와 x가 서로 다른 `ndarray`를 참조하고 있으며, 값도 올바릅니다. 문제가 해결되었군요. 이상이 14단계에서 `+=` 연산을 사용하지 않은 이유입니다.
