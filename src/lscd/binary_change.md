```py
class BinaryThresholdModel(lscd.Model):
    def predict(self, targets):
        predictions = self.graded_model.predict(targets)
        return threshold_fn(predictions)
```


```yaml
_target_: src.lscd.BinaryThresholdModel
graded_model:
    _target_: src.lscd.ApdCompareAll
    wic: ???
threshold_fn:
    _target_: src.threshold.mean_std
    t: 0.1

```


```yaml
pairings: [COMPARE, EARLIER, LATER]
samplings: [all, all, all]
function: 
    _target_: src.function

then:
    _target_: np.entropy
    model: apd_compare_all
```

```py
def entropy():


```