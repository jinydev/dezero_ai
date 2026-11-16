---
layout: docs
title: "19단계 변수 사용성 개선"
---

# 19단계 변수 사용성 개선

DeZero의 기초는 이미 완성되었습니다. 지금 상태로도 계산 그래프를 만들고 자동으로 미분을 계산할 수 있습니다. 그래서 앞으로 할 일은 DeZero를 더 쉽게 사용하도록 개선하는 작업입니다. 그 첫걸음으로, 이번 단계에서는 `Variable` 클래스를 더욱 쉽게 사용할 수 있게 해보겠습니다.

---

## 19.1 변수 이름 지정

앞으로 우리는 수많은 변수를 처리할 것이라서 변수들을 서로 구분할 필요가 있습니다. 변수에 '이름'을 붙여줄 수 있도록 설정하면 해결되겠군요. 그래서 다음과 같이 `Variable` 클래스에 `name`이라는 인스턴스 변수를 추가했습니다.

**steps/step19.py**
```python
class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0
```

이와 같이 초기화 인수 `name=None`을 추가하고 그 값을 인스턴스 변수 `name`에 설정합니다. 이제 예컨대 `x = Variable(np.array(1.0), 'input_x')`라고 작성하면 변수 `x`의 이름은 `input_x`가 됩니다. 아무런 이름도 주지 않으면 변수명으로 `None`이 할당됩니다.

> **NOTE**  
> 변수에 이름을 붙일 수 있다면, 예컨대 계산 그래프를 시각화할 때 변수 이름을 그래프에 표시할 수 있습니다. 계산 그래프 시각화는 25단계와 26단계를 참고하세요.

---

## 19.2 ndarray 인스턴스 변수

`Variable`은 데이터를 담는 '상자' 역할을 합니다. 그러나 사용하는 사람 입장에서 중요한 것은 상자가 아니라 그 안의 '데이터'입니다. 그래서 `Variable`이 데이터인 것처럼 보이게 하는 장치, 즉 상자를 투명하게 해주는 장치를 만들겠습니다.

> **NOTE**  
> 1단계에서 언급했듯이 수치 계산과 머신러닝 시스템은 다차원 배열(텐서)을 기본 데이터 구조로 사용합니다. 따라서 `Variable` 클래스는 (스칼라는 무시하고) `ndarray`만을 취급하기로 했습니다. 그래서 이번 절의 목표는 `Variable` 인스턴스를 `ndarray` 인스턴스처럼 보이게 하는 것입니다.

`Variable` 안에는 `ndarray` 인스턴스가 있습니다. 넘파이의 `ndarray` 인스턴스에는 다차원 배열용 인스턴스 변수가 몇 가지 제공됩니다. 다음은 그중 하나인 `shape` 인스턴스 변수를 사용하는 모습입니다.

```python
>>> import numpy as np
>>> x = np.array([[1, 2, 3], [4, 5, 6]])
>>> x.shape
(2, 3)
```

인스턴스 변수 `shape`는 다차원 배열의 형상을 알려줍니다. 참고로 앞의 결과에서 $(2, 3)$ 은 수학에서 말하는 $2 \times 3$ 행렬을 뜻합니다. 이제 똑같은 작업을 `Variable` 인스턴스에서도 할 수 있도록 확장하겠습니다.

**steps/step19.py**
```python
class Variable:
    ...

    @property
    def shape(self):
        return self.data.shape
```

`shape`라는 메서드를 추가한 후 실제 데이터의 `shape`를 반환하도록 했습니다. 여기서 중요한 부분은 `def shape(self):` 앞에 추가된 `@property`라는 한 줄입니다. 이 한 줄 덕분에 `shape` 메서드를 인스턴스 변수처럼 사용할 수 있게 됩니다. 확인해보겠습니다.

**steps/step19.py**
```python
x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
print(x.shape)  # x.shape() 대신 x.shape로 호출할 수 있다.
```

**실행 결과**
```
(2, 3)
```

이와 같이 메서드 호출이 아닌 인스턴스 변수로 데이터의 형상을 얻을 수 있습니다. 같은 방법으로 `ndarray`의 다른 인스턴스 변수들을 `Variable`에 추가할 수 있습니다. 여기에서는 다음 세 인스턴스 변수를 더 추가하겠습니다.

**steps/step19.py**
```python
class Variable:
    ...

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype
```

보다시피 `ndim`, `size`, `dtype`이라는 3개의 인스턴스 변수를 추가했습니다. `ndim`은 차원 수, `size`는 원소 수, `dtype`은 데이터 타입을 나타냅니다. 이상으로 `Variable`에 필요한 인스턴스 변수를 모두 추가했습니다. 이외에도 `ndarray`에는 많은 인스턴스 변수가 존재하며, 그 모두를 추가할 수도 있습니다. 하지만 단순한 작업이라서 지면으로는 설명하지 않겠습니다. 더 필요한 독자는 직접 추가해보기 바랍니다.

> **CAUTION**  
> 이 교제는 지금까지 `ndarray` 인스턴스의 데이터 타입인 `dtype`은 특별히 의식하지 않고 이야기를 진행했습니다. `dtype`을 지정하지 않으면 `ndarray` 인스턴스는 (환경에 따라) `float64` 또는 `int64`로 초기화됩니다. 한편 신경망에서는 `float32`를 사용하는 경우가 많습니다.

---

## 19.3 len 함수와 print 함수

이어서 `Variable` 클래스를 더 확장하여 파이썬의 `len` 함수와도 함께 사용할 수 있도록 하겠습니다. `len`은 객체 수를 알려주는 파이썬의 표준 함수입니다. 다음과 같이 사용할 수 있지요.

```python
>>> x = [1, 2, 3, 4]
>>> len(x)

>>> x = np.array([1, 2, 3, 4])
>>> len(x)

>>> x = np.array([[1, 2, 3], [4, 5, 6]])
>>> len(x)
```

이와 같이 리스트 등에 `len` 함수를 사용하면 그 안에 포함된 원소 수를 반환합니다. `ndarray` 인스턴스라면 첫 번째 차원의 원소 수를 반환합니다. 이제 이 `len` 함수가 `Variable` 안의 원소 수도 인식하도록 해보겠습니다.

**steps/step19.py**
```python
class Variable:
    ...

    def __len__(self):
        return len(self.data)
```

이와 같이 `__len__`이라는 특수 메서드를 구현하면 `Variable` 인스턴스에 대해서도 `len` 함수를 사용할 수 있게 됩니다. 이제 다음과 같은 코드를 작성할 수 있습니다.

```python
x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
print(len(x))
```

**실행 결과**
```
```

> **NOTE**  
> 파이썬에서 `__init__`와 `__len__` 등 특별한 의미를 지닌 메서드는 밑줄 두 개로 감싼 이름을 사용합니다.

마지막으로 `Variable`의 내용을 쉽게 확인할 수 있는 기능을 추가합니다. 바로 `print` 함수를 사용하여 `Variable` 안의 데이터 내용을 출력하는 기능입니다. 즉, 다음 예처럼 사용하고자 합니다.

```python
x = Variable(np.array([1, 2, 3]))
print(x)

x = Variable(None)
print(x)

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
print(x)
```

**실행 결과**
```
variable([1 2 3])
variable(None)
variable([[1 2 3]
          [4 5 6]])
```

이와 같이 `Variable` 인스턴스를 `print` 함수에 건네면 안에 담긴 `ndarray` 인스턴스의 내용을 출력하도록 하겠습니다. 이때 출력 결과는 `variable(...)` 형태로 통일하여 사용자에게 `Variable` 인스턴스임을 알려줍니다. 값이 `None`이거나 내용을 여러 줄로 출력해야 하는 경우도 지원합니다. 여러 줄일 때는 공백 문자로 시작 위치를 조정하여 보기 좋게 출력합니다. 다음은 이상의 조건을 만족하는 `Variable`의 `__repr__` 메서드 모습입니다.

**steps/step19.py**
```python
class Variable:
    ...

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
```

이처럼 `print` 함수가 출력해주는 문자열을 입맛에 맞게 정의하려면 `__repr__` 메서드를 재정의하면 됩니다. 반환값은 출력하고자 하는 문자열입니다. 앞의 코드에서는 `str(self.data)`를 이용하여 `ndarray` 인스턴스를 문자열로 변환했습니다. `str` 함수 안에서는 `ndarray` 인스턴스의 `__str__` 함수가 호출되고 숫자가 문자열로 변환됩니다. 줄바꿈(`\n`)이 있으면 줄을 바꾼 후 새로운 줄 앞에 공백 9개를 삽입하여 여러 줄에 걸친 출력도 숫자의 시작 위치가 가지런하게 표시되게 했습니다. 마지막으로 변환된 문자열을 '`variable(...)`' 형태로 감쌉니다.

이상으로 `Variable` 클래스를 '투명한 상자'로 만드는 작업을 일부 끝마쳤습니다. 다음 단계에서도 이 작업을 계속 이어갈 것입니다.
