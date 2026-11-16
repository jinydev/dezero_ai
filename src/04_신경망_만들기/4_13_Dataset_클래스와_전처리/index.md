---
layout: docs
title: "49단계 Dataset 클래스와 전처리"
---

# 49단계 Dataset 클래스와 전처리

이전 단계에서는 스파이럴 데이터셋을 읽어올 때 `x, t = dezero.datasets.get_spiral()` 코드를 사용했습니다. 이때 `x`와 `t`는 `ndarray` 인스턴스이며, 형상은 `x`가 `(300, 2)`이고 `t`가 `(300,)`이었습니다. 즉, 총 300개의 데이터를 하나의 `ndarray` 인스턴스에 담았던 것입니다.

스파이럴 데이터셋은 300개 정도의 작은 데이터셋이라서 `ndarray` 인스턴스 하나로 처리할 수 있었지만, 대규모 데이터셋을 처리할 때는 문제가 됩니다. 예를 들어 데이터가 100만 개라면 어떨까요? 이처럼 거대한 데이터를 하나의 `ndarray` 인스턴스로 처리하면 모든 원소를 한꺼번에 메모리에 올려야만 하죠. 그래서 이러한 문제에 대응할 수 있도록 데이터셋 전용 클래스인 `Dataset` 클래스를 만들려 합니다. `Dataset` 클래스에는 데이터를 전처리할 수 있는 구조도 추가할 것입니다.

---

## 49.1 Dataset 클래스 구현

`Dataset` 클래스는 기반 클래스로서의 역할을 하고, 사용자가 실제로 사용하는 데이터셋은 이를 상속하여 구현하게 할 것입니다. 그럼 `Dataset` 클래스의 구현을 보겠습니다.

**dezero/datasets.py**
```python
import numpy as np

class Dataset:
    def __init__(self, train=True):
        self.train = train
        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index):
        assert np.isscalar(index) # index는 정수(스칼라)만 지원
        if self.label is None:
            return self.data[index], None
        else:
            return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass
```

우선 초기화 때 `train` 인수를 받습니다. 이 인수는 '학습'이냐 '테스트'이냐를 구별하기 위한 플래그입니다. 인스턴스 변수 `data`와 `label`에는 각각 입력 데이터와 레이블을 보관합니다. 그리고 자식 클래스에서는 `prepare` 메서드가 데이터 준비 작업을 하도록 구현해야 합니다.

`Dataset` 클래스에서 중요한 메서드는 `__getitem__`과 `__len__`입니다. 이 두 메서드(인터페이스)를 제공해야만 'DeZero 데이터셋'이라고 할 수 있기 때문이죠. 이렇게 인터페이스를 통일하면 다양한 데이터셋을 교체해가며 사용할 수 있습니다.

`__getitem__`은 파이썬의 특수 메서드로, `x[0]`이나 `x[1]`처럼 괄호를 사용해 접근할 때의 동작을 정의합니다. `Dataset` 클래스의 `__getitem__` 메서드는 단순히 지정된 인덱스에 위치하는 데이터를 꺼냅니다. 레이블 데이터가 없다면 입력 데이터 `self.data[index]`의 레이블은 `None`을 반환합니다('정답 데이터 없음' 상황을 가정한 동작입니다). 그리고 데이터셋의 길이를 알려주는 `__len__` 메서드는 `len` 함수를 사용할 때 호출됩니다(예: `len(x)`).

> **CAUTION**
> `__getitem__` 메서드는 원래 '슬라이스' 조작도 처리할 수 있습니다. 슬라이스란 예를 들어 `x[1:3]` 같은 작업을 말합니다. 그러나 `DeZero`의 `Dataset` 클래스는 슬라이스에는 대응하지 않고 단순히 정수를 `index`로 사용하는 조작법만 지원하게 했습니다.

이상이 `Dataset` 클래스입니다. 그럼 `Dataset` 클래스를 상속하여 스파이럴 데이터셋을 구현하겠습니다. 클래스 이름은 `Spiral`로 하고 코드는 다음과 같습니다.

**dezero/datasets.py**
```python
class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(self.train)
```

이와 같이 `prepare` 메서드에서 인스턴스 변수인 `data`와 `label`에 데이터를 설정하는 게 다입니다. 이것으로 다음과 같이 `Spiral` 클래스를 사용하여 데이터를 추출할 수 있습니다. 물론 데이터의 길이도 얻을 수 있고요.

```python
import dezero

train_set = dezero.datasets.Spiral(train=True)
print(train_set[0])
print(len(train_set))
```

**실행 결과**
```text
(array([-0.13981389, -0.00721657], dtype=float32), 1)
```

이 예에서는 괄호를 사용하여 `train_set[0]` 형태로 데이터를 가져왔고, 그 결과 0번째 입력 데이터와 레이블이 튜플로 반환되었습니다.

---

## 49.2 큰 데이터셋의 경우

스파이럴 데이터셋 같은 작은 데이터셋이라면 `Dataset` 클래스의 인스턴스 변수인 `data`와 `label`에 직접 `ndarray` 인스턴스를 유지해도 무리가 없습니다. 하지만 데이터셋이 훨씬 크다면 이런 구현 방식은 사용할 수 없죠. 그래서 다음과 같은 구현을 생각해보았습니다.

```python
class BigData(Dataset):
    def __getitem__(index):
        x = np.load('data/{}.npy'.format(index))
        t = np.load('label/{}.npy'.format(index))
        return x, t

    def __len__():
        return 1000000
```

`data` 디렉터리와 `label` 디렉터리에 각각 100만 개의 데이터(레이블)가 저장되어 있다고 가정했습니다. 그리고 `BigData` 클래스를 초기화할 때는 데이터를 아직 읽지 않고, 데이터에 접근할 때 비로소 읽게 하는 것입니다. 구체적으로는 `__getitem__(self, index)`가 불리는 시점에서 `data` 디렉터리에 있는 데이터를 읽습니다(`np.load` 함수는 53단계에서 설명합니다). 앞에서도 이야기하였듯, 'DeZero 데이터셋'이 되기 위한 요건은 `__getitem__`과 `__len__` 두 메서드를 구현하는 것입니다. 앞의 `BigData` 클래스는 요건을 충족하므로 `Spiral` 클래스처럼 사용할 수 있습니다.

이어서 `Spiral` 클래스를 사용하여 학습 코드를 작성해봅시다. 그러려면 데이터를 '연결'해야 합니다.

---

## 49.3 데이터 이어 붙이기

신경망을 학습시킬 때는 데이터셋 중 일부를 미니배치로 꺼냅니다. 다음은 `Spiral` 클래스를 사용하여 데이터를 미니배치로 가져오는 코드입니다.

```python
train_set = dezero.datasets.Spiral()

batch_index = [0, 1, 2] # 0에서 2번째까지의 데이터를 꺼냄
batch = [train_set[i] for i in batch_index]
# batch = [(data_0, label_0), (data_1, label_1), (data_2, label_2)]
```

이와 같이 우선은 인덱스를 지정하여 여러 데이터(미니배치)를 꺼냅니다. 이 코드를 실행하면 `batch`에 여러 데이터가 리스트로 저장됩니다. 이제 이 데이터를 `DeZero`의 신경망에 입력하려면 하나의 `ndarray` 인스턴스로 변환해야 하는데, 코드로는 다음과 같습니다.

```python
x = np.array([example[0] for example in batch])
t = np.array([example[1] for example in batch])

print(x.shape)
print(t.shape)
```

**실행 결과**
```text
(3, 2)
(3,)
```

여기에서는 `batch`의 각 원소에서 데이터(혹은 레이블)만을 꺼내 하나의 `ndarray` 인스턴스로 변형했습니다(이어 붙였습니다). 이것으로 드디어 신경망에 입력할 수 있는 형태의 데이터가 마련되었습니다.

---

## 49.4 학습 코드

그럼 `Spiral` 클래스를 사용하여 학습을 해보겠습니다. 코드는 다음과 같습니다(파이썬의 임포트 코드는 생략했고, 이전 단계에서 달라진 부분은 음영으로 표시했습니다).

**steps/step49.py**
```python
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = dezero.datasets.Spiral()
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0
```

```python
    for i in range(max_iter):
        # 미니배치 꺼내기
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch = [train_set[i] for i in batch_index]
        batch_x = np.array([example[0] for example in batch])
        batch_t = np.array([example[1] for example in batch])

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    # 에포크마다 손실 출력
    avg_loss = sum_loss / data_size
    print('epoch %d, loss %.2f' % (epoch + 1, avg_loss))
```

**실행 결과**
```text
epoch 1, loss 1.35
epoch 2, loss 1.06
epoch 3, loss 0.98
epoch 4, loss 0.90
...
```

이전 단계에서 달라진 부분은 `Spiral` 클래스를 사용하는 내용입니다. 그에 따라 미니배치를 만드는 부분의 코드를 수정했습니다. 그 외에는 이전 단계와 같습니다. 이 코드를 실행하면 이전 단계와 마찬가지로 손실(loss)이 낮아지는 모습을 확인할 수 있습니다.

이제 `Dataset` 클래스를 사용하여 신경망을 학습할 수 있게 되었습니다. `Dataset` 클래스를 사용하는 이점은 다른 데이터셋으로 교체해 학습할 때 확인할 수 있는데, 예컨대 앞의 코드에서 `Spiral`을 `BigData`로 교체하는 것만으로 훨씬 큰 데이터셋에 대응할 수 있습니다. 데이터셋 인터페이스를 통일하여 다양한 데이터셋을 똑같은 코드로 처리할 수 있기 때문입니다.

마지막으로 `Dataset` 클래스에 전처리 기능을 추가하겠습니다.

---

## 49.5 데이터셋 전처리

머신러닝에서는 모델에 데이터를 입력하기 전에 데이터를 특정한 형태로 가공할 일이 많습니다. 예를 들어 데이터에서 특정 값을 제거하거나 데이터의 형상을 변형하는 처리입니다. 이미지를 회전 혹은 좌우 반전시키거나 데이터 수를 인위적으로 늘리는 기술(**데이터 확장(data augmentation)**)도 자주 이용됩니다. 이러한 전처리(및 데이터 확장)에 대응하기 위해 `Dataset` 클래스에도 전처리 기능을 추가하겠습니다. 추가할 코드는 다음과 같습니다.

**dezero/datasets.py**
```python
class Dataset:
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = lambda x: x
        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]),\
                   self.target_transform(self.label[index])

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass
```

이와 같이 초기화 시에 `transform`과 `target_transform`을 새롭게 받습니다. 이 인수들은 호출 가능한 객체(예를 들어 '파이썬 함수' 등)를 받습니다. `transform`은 입력 데이터 하나에 대한 변환을 처리하고, `target_transform` 레이블 하나에 대한 변환을 처리합니다. 이 값이 `None`이

라면 전처리 로직은 `lambda x : x`로 설정되어, 받은 인수를 그대로 반환합니다(아무런 전처리도 하지 않습니다). 이상의 전처리 설정 기능이 있으면 다음과 같은 코드를 작성할 수 있습니다.

```python
def f(x):
    y = x / 2.0
    return y

train_set = dezero.datasets.Spiral(transform=f)
```

이 코드는 입력 데이터를 1/2로 스케일 변환하는 전처리 예입니다. 이처럼 데이터셋에 대해 사용자가 원하는 전처리를 추가할 수 있습니다. 참고로 `DeZero`는 데이터 정규화(normalization)와 이미지 데이터(`PIL.Image` 인스턴스) 변환 등 전처리 시 자주 사용되는 변환들을 `dezero/transforms.py`에 준비해뒀습니다. 다음은 사용 예시입니다.

```python
from dezero import transforms

f = transforms.Normalize(mean=0.0, std=2.0)
train_set = dezero.datasets.Spiral(transform=f)
```

`transforms.Normalize(mean=0.0, std=2.0)`에 의해 입력을 `x`라 할 때 `(x - mean) / std`라는 변환이 이루어집니다. 또한 다음과 같이 여러 변환 처리를 연달아 수행할 수도 있습니다.

```python
f = transforms.Compose([transforms.Normalize(mean=0.0, std=2.0),
                        transforms.AsType(np.float64)])
```

`transforms.Compose` 클래스는 주어진 변환 목록을 앞에서부터 순서대로 처리합니다. 지금 예에서는 정규화를 먼저 하고, 연이어 데이터 타입을 `np.float64`로 변환하는 일을 수행합니다. 이처럼 `dezero/transforms.py`에는 편리한 변환 처리가 준비되어 있습니다. 각각의 구현은 간단하므로 설명은 생략하니, 관심 있는 분은 파일 내용을 참고하세요.
